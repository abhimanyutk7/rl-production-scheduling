from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from .env import FactoryEnv
from .config import ShopConfig


TRAIN_SEEDS = list(range(1000))
VALIDATION_SEEDS = list(range(10000, 10500))
TEST_SEEDS = list(range(20000, 20500))


class ScenarioPool(gym.Wrapper):
    def __init__(self, env, seeds, random_seed):
        super().__init__(env)
        self.seeds, self.rng = seeds, np.random.default_rng(random_seed)

    def reset(self, **kwargs):
        kwargs["seed"] = int(self.rng.choice(self.seeds))
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # The finite shift includes its final penalty, so PPO must not bootstrap beyond it.
        return observation, reward, terminated or truncated, False, info

    def action_masks(self):
        return self.env.action_masks()


class EpisodeLog(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.rows.append({"step": self.num_timesteps, **info["metrics"], **{f"cost_{k}": v for k, v in info["cost_components"].items()}})
        return True


class MaskedDQN:
    """Small PyTorch DQN; mask exploration, greedy selection AND Bellman targets."""
    def __init__(self, obs_size, actions, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, actions))
        self.target = nn.Sequential(nn.Linear(obs_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, actions))
        self.target.load_state_dict(self.net.state_dict())
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=3e-4)
        self.obs_size, self.actions = obs_size, actions
        self.losses = []

    def predict(self, obs, mask):
        with torch.no_grad():
            q = self.net(torch.as_tensor(obs, dtype=torch.float32))
            return int(q.masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())

    def learn(self, env, steps):
        replay = deque(maxlen=20000)
        obs, info = env.reset()
        logs = []
        for step in range(steps):
            mask = info["action_mask"]
            epsilon = max(0.05, 1 - step / max(1, steps * 0.6))
            action = int(self.rng.choice(np.flatnonzero(mask))) if self.rng.random() < epsilon else self.predict(obs, mask)
            nxt, reward, terminated, truncated, nxt_info = env.step(action)
            done = terminated or truncated
            replay.append((obs, action, reward, nxt, done, nxt_info["action_mask"]))
            if step >= 256 and step % 4 == 0:
                sample = [replay[i] for i in self.rng.choice(len(replay), 64, replace=False)]
                states, actions, rewards, next_states, dones, next_masks = map(np.asarray, zip(*sample))
                states, next_states = torch.tensor(states, dtype=torch.float32), torch.tensor(next_states, dtype=torch.float32)
                prediction = self.net(states).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
                with torch.no_grad():
                    q = self.target(next_states).masked_fill(~torch.tensor(next_masks), -1e9)
                    target = torch.tensor(rewards, dtype=torch.float32) + (~torch.tensor(dones)).float() * q.max(1).values
                loss = nn.functional.smooth_l1_loss(prediction, target)
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimiser.step()
                self.losses.append(float(loss.detach()))
            if step % 500 == 0:
                self.target.load_state_dict(self.net.state_dict())
            obs, info = nxt, nxt_info
            if done:
                logs.append({"step": step + 1, **info["metrics"], **{f"cost_{k}": v for k, v in info["cost_components"].items()}})
                obs, info = env.reset()
        return logs

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "obs_size": int(self.obs_size), "actions": int(self.actions)}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(data["obs_size"], data["actions"])
        model.net.load_state_dict(data["state"])
        return model


def train(config, output, algorithm="ppo", objective="joint", seed=0, steps=200000, instance=None):
    import pandas as pd
    from dataclasses import asdict
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = ScenarioPool(FactoryEnv(config, objective=objective, instance=instance), TRAIN_SEEDS, seed)
    started = time.perf_counter()
    if algorithm == "ppo":
        model = MaskablePPO("MlpPolicy", env, seed=seed, device="cpu", n_steps=256, batch_size=64, n_epochs=4,
                            gamma=1.0, ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]}, verbose=0)
        log = EpisodeLog()
        model.learn(total_timesteps=steps, callback=log)
        model.save(output / "model.zip")
        rows, actual_steps = log.rows, model.num_timesteps
    else:
        model = MaskedDQN(env.observation_space.shape[0], env.action_space.n, seed)
        rows = model.learn(env, steps)
        model.save(output / "model.pt")
        actual_steps = steps
    pd.DataFrame(rows).to_csv(output / "training.csv", index=False)
    metadata = {"algorithm": algorithm, "objective": objective, "seed": seed, "requested_steps": steps,
                "actual_steps": actual_steps, "training_seconds": time.perf_counter() - started,
                "train_scenario_range": [0, 999], "validation_scenario_range": [10000, 10499], "test_scenario_range": [20000, 20499],
                "instance": str(Path(instance).resolve()) if instance else None, "config": asdict(config), "discount": 1.0,
                "shift_end_is_terminal": True}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    env.close()
    return metadata


def load_policy(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["algorithm"] == "ppo":
        model = MaskablePPO.load(directory / "model.zip", device="cpu")
        return lambda env: int(model.predict(env._observation(), action_masks=env.action_masks(), deterministic=True)[0])
    model = MaskedDQN.load(directory / "model.pt")
    return lambda env: model.predict(env._observation(), env.action_masks())
