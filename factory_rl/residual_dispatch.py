"""Hybrid dispatch PPO: fixed public service rule and a shared residual job scorer.

This is a separate protocol. It changes neither the physical simulator nor the
meaning of the original inspection experiments. Service choices are not learned.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .inspection import InspectionConfig, InspectionEnv
from .priority_search import priority_action, feature_terms, _json, _csv, _table
from .inspection_study import source_fingerprint, paired_statistics, _run_lock
from .agents import ScenarioPool, TRAIN_SEEDS

# Setup lambda=8 and PM threshold=.55 were selected on the earlier validation
# pool, not on any test outcome. Score scaling matches the frozen priority rule.
PRIOR = np.array([1., 4., 0., 0., 0., 0., .55, 1.])
VALIDATION_START, TEST_START = 10200, 24500


class DispatchEnv(gym.Wrapper):
    schema_version = 'hybrid-dispatch-v1'

    def __init__(self, config, prior=PRIOR):
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (8,) or not np.isfinite(prior).all():
            raise ValueError('Eight finite public priority parameters required')
        super().__init__(InspectionEnv(config, strict_actions=True))
        self.prior = prior.copy()
        self.prior.flags.writeable = False
        self.observation_schema = [f'job{j}.{f}' for j in range(config.shortlist)
            for f in ['duration', 'setup', 'slack', 'age', 'ops', 'energy', 'prior_logit', 'present']]
        self.observation_schema += ['shift_time', 'episode_left', 'tariff', 'peak', 'queue',
            'late', 'health_mean', 'health_std', 'health_q05', 'health_q95', 'busy', 'remaining']
        self.observation_space = gym.spaces.Box(-20., 20., (len(self.observation_schema),), np.float32)

    def action_masks(self):
        state, config = self.env.policy_state(), self.env.config
        chosen = priority_action(state, config, self.prior)
        mask = np.zeros(config.shortlist + 3, dtype=bool)
        if chosen < config.shortlist:
            mask[:config.shortlist] = state['action_mask'][:config.shortlist]
        else:
            mask[chosen] = True
        return mask

    def observation(self):
        state, config = self.env.policy_state(), self.env.config
        jobs = np.zeros((config.shortlist, 8), dtype=np.float64)
        chosen = priority_action(state, config, self.prior)
        if state['candidates']:
            terms = np.array([feature_terms(j, state, config) for j in state['candidates']])
            scores = terms @ self.prior[:6]
            # Calculate the winner before float32 conversion; its explicit bonus
            # preserves the deterministic prior even for tied/rounded scores.
            logits = np.clip(-(scores-scores.min())/.2, -20, 0)
            if chosen < config.shortlist:
                logits[chosen] += 1.
            jobs[:len(terms), :6] = np.clip(terms, -4, 4)
            jobs[:len(terms), 6], jobs[:len(terms), 7] = logits, 1.
        machine = state['machines'][state['machine']] if state['machine'] >= 0 else None
        context = [state['time'] % 1440 / 1440, 1-state['time']/config.horizon,
            config.tariff(state['time'])/max(1, config.peak_rate), state['peak_kw']/300,
            state['waiting']/(config.jobs+state['waiting']), state['late']/(config.jobs+state['late']),
            *([machine[f] for f in ['mean', 'std', 'q05', 'q95']] if machine else [0.]*4),
            sum(m['mode'] != 'idle' for m in state['machines'])/config.machines,
            sum(m['remaining'] for m in state['machines'])/(config.machines*config.shift_minutes)]
        return np.concatenate([jobs.ravel(), np.clip(context, -4, 4)]).astype(np.float32)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.observation(), info

    def step(self, action):
        if not self.action_masks()[int(action)]:
            raise ValueError('Action outside the declared public service/dispatch gate')
        _, reward, terminal, truncated, info = self.env.step(action)
        return self.observation(), reward/self.env.config.days, terminal, truncated, info


class SharedScorer(nn.Module):
    def __init__(self, shortlist, use_prior=True):
        super().__init__()
        self.shortlist, self.use_prior = shortlist, use_prior
        self.latent_dim_pi, self.latent_dim_vf = shortlist+3, 64
        self.job_net = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_net = nn.Sequential(nn.Linear(18, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

    def split(self, x):
        return x[:, :self.shortlist*8].reshape(-1, self.shortlist, 8), x[:, self.shortlist*8:]

    def forward_actor(self, x):
        jobs, context = self.split(x)
        context = context[:, None, :].expand(-1, self.shortlist, -1)
        residual = 2*torch.tanh(self.job_net(torch.cat([jobs[:, :, :6], context], dim=-1)).squeeze(-1))
        logits = residual + jobs[:, :, 6] if self.use_prior else residual
        return torch.cat([logits, torch.zeros((len(x), 3), device=x.device)], dim=1)

    def forward_critic(self, x):
        jobs, context = self.split(x)
        present = jobs[:, :, 7:8]
        pooled = (jobs[:, :, :6]*present).sum(1)/present.sum(1).clamp_min(1)
        return self.value_net(torch.cat([pooled, context], dim=1))

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class ResidualPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, shortlist=10, use_prior=True, **kwargs):
        self.shortlist, self.use_prior = shortlist, use_prior
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SharedScorer(self.shortlist, self.use_prior).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        nn.init.zeros_(self.mlp_extractor.job_net[-1].weight)
        nn.init.zeros_(self.mlp_extractor.job_net[-1].bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), 'shortlist': self.shortlist, 'use_prior': self.use_prior}


class UpdateGPUOnlyPPO(MaskablePPO):
    """Batch gradients on CUDA; keep tiny per-decision inference on the CPU."""
    def collect_rollouts(self, *args, **kwargs):
        self.policy.to('cpu')
        self.device = torch.device('cpu')
        return super().collect_rollouts(*args, **kwargs)

    def train(self):
        self.policy.to('cuda')
        self.device = torch.device('cuda')
        super().train()
        self.gpu_update_calls = getattr(self, 'gpu_update_calls', 0)+1


def make_model(env, seed, device, use_prior=True):
    config = env.unwrapped.config if hasattr(env.unwrapped, 'config') else None
    shortlist = config.shortlist if config else 10
    algorithm = UpdateGPUOnlyPPO if device == 'cuda_updates' else MaskablePPO
    return algorithm(ResidualPolicy, env, seed=seed, device='cuda' if device == 'cuda_updates' else device, learning_rate=1e-4,
        n_steps=512, batch_size=256, n_epochs=4, gamma=1., gae_lambda=.98,
        ent_coef=0., clip_range=.1, target_kl=.015,
        policy_kwargs=dict(shortlist=shortlist, use_prior=use_prior), verbose=0)


def validate_rows(rows, policy, seed, scenarios):
    if (len({r['scenario_seed'] for r in rows}) != len(rows)
        or any(r['policy'] != policy or r['training_seed'] != seed or r['scenario_seed'] not in scenarios for r in rows)):
        raise ValueError('Evaluation cache has incompatible or duplicate rows')


def bind_identity(root, identity):
    path = Path(root)/'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Protocol differs; preserve the old stage and choose a new output')
    _json(path, identity)


def evaluation_policy(policy):
    # Cached action distributions hold non-leaf tensors after an update, so
    # deepcopy is unsafe. Rebuild only the inference module without changing RNG.
    with torch.random.fork_rng(devices=[]):
        actor = ResidualPolicy(**policy._get_constructor_parameters())
    actor.load_state_dict({k:v.detach().cpu() for k,v in policy.state_dict().items()})
    actor.set_training_mode(False)
    return actor


def _roll_dispatch(config, controller, scenario):
    env = DispatchEnv(config)
    obs, _ = env.reset(seed=int(scenario))
    changed, dispatch = 0, 0
    started = time.perf_counter()
    while not env.unwrapped.terminated:
        prior = priority_action(env.unwrapped.policy_state(), config, PRIOR)
        action = prior if controller is None else int(controller.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
        if action < config.shortlist:
            dispatch += 1
            changed += action != prior
        obs, _, _, _, _ = env.step(action)
    metrics = env.unwrapped.metrics()
    for key in ['ledger_daily', 'ledger_billing']:
        metrics[key] = json.dumps(metrics[key])
    metrics.update(changed_dispatches=int(changed), dispatch_decisions=int(dispatch), wall_seconds=time.perf_counter()-started)
    return metrics


def _vector(config, seed, n_envs):
    from stable_baselines3.common.vec_env import DummyVecEnv
    # The machine has 16GB RAM. Batched in-process environments avoid loading
    # several CUDA runtimes in separate Windows worker processes.
    return DummyVecEnv([lambda i=i: ScenarioPool(DispatchEnv(config), TRAIN_SEEDS, seed*100+i) for i in range(n_envs)])


def benchmark(root, config, n_envs=4):
    results = []
    torch.set_num_threads(1)
    for device in ['cpu', 'cuda', 'cuda_updates'] if torch.cuda.is_available() else ['cpu']:
        env = _vector(config, 982, n_envs)
        model = make_model(env, 982, device)
        started = time.perf_counter()
        model.learn(2048)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        results.append(dict(device=device, steps=int(model.num_timesteps), seconds=elapsed,
            steps_per_second=model.num_timesteps/elapsed, gpu_update_calls=getattr(model, 'gpu_update_calls', 0),
            final_parameter_device=str(next(model.policy.parameters()).device)))
        env.close()
        del model
    record = dict(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        n_envs=n_envs, measurement='2048 actual PPO decisions including updates; same config and seed, cold starts', results=results)
    _json(Path(root)/'device_benchmark.json', record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def _train(config, root, seed, condition, args):
    from stable_baselines3.common.callbacks import BaseCallback
    directory = root/'models'/f'{condition}_s{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory/'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['status'] == 'complete':
            assert hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest() == meta['model_sha256']
            return MaskablePPO.load(directory/'model.zip', device='cpu'), meta
    env = _vector(config, seed, args.envs)
    model = make_model(env, seed, args.device, use_prior=condition == 'residual')
    started = time.perf_counter()
    meta = dict(status='training', condition=condition, seed=seed, device=args.device,
        config=asdict(config), schema=DispatchEnv(config).observation_schema, service='fixed public prior gate; not learned',
        method='shared candidate neural scorer, bounded residual logits, MaskablePPO',
        reward_scale=1/config.days, initialisation='zero residual; deterministic prior exactly preserved' if condition == 'residual' else 'zero actor; no job priority prior')
    _json(meta_path, meta)

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history, self.best, self.selected = [], float('inf'), 0

        def validate(self):
            # CPU evaluation avoids one CUDA transfer per tiny decision.
            actor = evaluation_policy(model.policy)
            metrics = [_roll_dispatch(config, actor, args.validation_start+j) for j in range(args.validation)]
            cost = float(np.mean([r['full_objective'] for r in metrics]))
            row = dict(step=int(model.num_timesteps), mean_cost=cost,
                changed_dispatches=float(np.mean([r['changed_dispatches'] for r in metrics])))
            if hasattr(model, '_logger'):
                row.update({k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)})
            self.history.append(row)
            if cost < self.best:
                self.best, self.selected = cost, int(model.num_timesteps)
                model.save(directory/'model.zip')
            _json(directory/'validation.json', self.history)
            print(f'{condition} seed{seed} step{model.num_timesteps} validation {cost:.4f}', flush=True)

        def _on_rollout_end(self):
            if model.num_timesteps % 8192 == 0:
                self.validate()

        def _on_step(self):
            return True

    progress = Progress()
    progress.validate()
    model.learn(args.steps, callback=progress)
    progress.validate()
    meta.update(status='complete', selected_steps=progress.selected, validation_cost=progress.best,
        actual_steps=int(model.num_timesteps), gpu_update_calls=getattr(model, 'gpu_update_calls', 0),
        final_parameter_device=str(next(model.policy.parameters()).device),
        training_seconds=time.perf_counter()-started,
        model_sha256=hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest())
    _json(meta_path, meta)
    env.close()
    return MaskablePPO.load(directory/'model.zip', device='cpu'), meta


def run(args):
    import pandas as pd
    from .inspection_study import rollout
    from .priority_search import _roll, INITIAL
    torch.set_num_threads(1)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    base = Path('outputs/inspection_research/final_pilot')
    config = InspectionConfig(**json.loads((base/'run.json').read_text())['config'])
    if args.benchmark:
        return benchmark(root, config, args.envs)
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use the verified GPU Python or explicitly select CPU')
    artifacts = [Path('outputs/inspection_research/priority_audit/baseline_validation.json')]
    artifacts += [Path('outputs/inspection_research/priority_search')/f'policy_s{s}.json' for s in range(args.seeds)]
    artifacts += [base/'models'/f'adaptive_s{s}'/'model.zip' for s in range(args.seeds)]
    prior_file = artifacts[0]
    best_prior = min(json.loads(prior_file.read_text()), key=lambda r:r['validation_cost'])
    np.testing.assert_array_equal(PRIOR, best_prior['weights'])
    identity = dict(config=asdict(config), settings=vars(args), prior=PRIOR.tolist(),
        source_hash=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()+Path('factory_rl/priority_search.py').read_text()).encode()).hexdigest(),
        controls_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        conditions=['residual', 'no_prior'], schema=DispatchEnv(config).observation_schema)
    with _run_lock(root):
        bind_identity(root, identity)
        _json(root/'run.json', dict(status='running', identity=identity))
        scenarios = set(range(args.test_start, args.test_start+args.episodes))
        all_rows = []
        jobs = [(p, s) for s in range(args.seeds) for p in ['residual', 'no_prior', 'legacy_adaptive', 'learned_priority']]
        jobs += [('retuned_setup_spt', -1), ('spt', -1)]
        metadata = []
        for policy, seed in jobs:
            if policy in ['residual', 'no_prior']:
                controller, meta = _train(config, root, seed, policy, args)
                metadata.append(meta)
            elif policy == 'legacy_adaptive':
                controller = MaskablePPO.load(base/'models'/f'adaptive_s{seed}'/'model.zip', device='cpu')
            else:
                controller = None
            path = root/f'{policy}_s{seed}.csv'
            rows = pd.read_csv(path).to_dict('records') if path.exists() else []
            validate_rows(rows, policy, seed, scenarios)
            for scenario in sorted(scenarios-{r['scenario_seed'] for r in rows}):
                if policy in ['residual', 'no_prior', 'retuned_setup_spt']:
                    metrics = _roll_dispatch(config, controller, scenario)
                elif policy == 'legacy_adaptive':
                    metrics, _ = rollout(config, controller, scenario)
                else:
                    weights = INITIAL if policy == 'spt' else np.array(json.loads(artifacts[1+seed].read_text())['weights'])
                    metrics, _ = _roll(config, weights, scenario)
                rows.append(dict(policy=policy, training_seed=seed, scenario_seed=scenario, **metrics))
                _csv(path, pd.DataFrame(rows))
            all_rows.extend(rows)
            print(f'Finished {policy} seed{seed}: {len(all_rows)} rows', flush=True)
        frame = pd.DataFrame(all_rows)
        expected = {(p, s, j) for p,s in jobs for j in scenarios}
        actual = list(frame[['policy', 'training_seed', 'scenario_seed']].itertuples(index=False, name=None))
        assert len(actual) == len(set(actual)) and set(actual) == expected
        _csv(root/'episodes.csv', frame)
        fields = ['full_objective', 'completion_rate', 'on_time_rate', 'inspection_count', 'maintenance_count',
                  'breakdowns', 'belief_coverage90', 'changed_dispatches', 'dispatch_decisions']
        summary = frame.groupby('policy')[fields].mean().reset_index()
        _csv(root/'summary.csv', summary)
        paired = pd.DataFrame([paired_statistics(frame, 'residual', reference) for reference in
            ['no_prior', 'retuned_setup_spt', 'spt', 'legacy_adaptive', 'learned_priority']])
        _csv(root/'paired_comparisons.csv', paired)
        report = ['# Hybrid neural dispatch: separate research protocol', '',
            'Three-day synthetic episodes; accelerated two-day billing. Raw normalised objective is evaluated.',
            'The neural policy learns job dispatch only. A fixed public belief rule controls maintenance/WAIT.',
            'Shared candidate scorer, bounded residual to a previously validation-selected setup-aware SPT prior.',
            'These established ideas are adapted here; no new-algorithm or inspection-value claim.', '',
            f'{args.seeds} training seeds, {args.steps} requested PPO steps/condition, {args.episodes} fresh test sequences, {len(frame)} rows.',
            f'Training runtime: {args.device}. CUDA-update counts and selected steps are persisted per model.', '',
            _table(summary), '', 'Negative paired differences favour residual PPO; seed/scenario uncertainty:', '', _table(paired), '',
            '## Selected checkpoints', '', _table(pd.DataFrame([{k:m[k] for k in
                ['condition','seed','selected_steps','validation_cost','gpu_update_calls']} for m in metadata])), '',
            'Step zero means the initial rule/scorer was retained; never call that an added PPO gain.',
            'The initial deterministic residual policy equals its prior, but stochastic training explores alternative jobs.',
            'The no-prior arm uses the same service gate, bounded scorer and training budget, without job-priority logits.',
            'Legacy neural controls have different architectures and training protocols; that comparison cannot isolate a single cause.',
            'Low cost versus older policies does not establish improvement over the strong prior.',
            'Synthetic simulation only; normalized cost is neither accuracy nor money savings. Test outcomes must not tune this protocol.', '',
            'Sources: https://arxiv.org/abs/2010.12367 ; https://arxiv.org/abs/1812.03201 ; https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html']
        (root/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
        _json(root/'run.json', dict(status='complete', identity=identity, rows=len(frame), completed_models=len(metadata)))
        print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/inspection_research/residual_dispatch')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'cuda_updates'], default='cuda_updates')
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--validation', type=int, default=10)
    parser.add_argument('--validation-start', type=int, default=VALIDATION_START)
    parser.add_argument('--test-start', type=int, default=TEST_START)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.episodes, args.validation, args.envs) < 1 or args.seeds > 3:
        parser.error('Positive budgets and at most three matched legacy seeds required')
    run(args)


if __name__ == '__main__':
    main()
