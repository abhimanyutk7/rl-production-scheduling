from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .env import FactoryEnv
from .baselines import dispatch, CPSATPolicy
from .agents import load_policy, TEST_SEEDS, VALIDATION_SEEDS


def rollout(config, policy, scenario_seed, instance=None, record=False):
    env = FactoryEnv(config, instance=instance, record=record, strict_actions=True)
    env.reset(seed=scenario_seed)
    timings = []
    total_reward = 0.0
    while not (env.terminated or env.truncated):
        started = time.perf_counter()
        action = policy(env)
        timings.append((time.perf_counter() - started) * 1000)
        _, reward, _, _, _ = env.step(action)
        total_reward += reward
    metrics = {**env.metrics(), "reward": total_reward, "decision_ms_mean": float(np.mean(timings)),
               "decision_ms_p95": float(np.percentile(timings, 95)), "planning_seconds": sum(timings) / 1000,
               **{f"cost_{k}": v for k, v in env.cost_components().items()}}
    if isinstance(policy, CPSATPolicy):
        metrics["solver_calls"] = len(policy.statuses)
        metrics["solver_no_incumbent"] = sum(s not in ("FEASIBLE", "OPTIMAL") for s in policy.statuses)
    return metrics, env


def evaluate(config, output, models=None, episodes=20, split="test", solver_seconds=0.1, include_solver=True, instance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = (TEST_SEEDS if split == "test" else VALIDATION_SEEDS)[:episodes]
    if not seeds or episodes > 500:
        raise ValueError("episodes must be 1..500")
    rows = []
    methods = {name: (lambda name=name: lambda env: dispatch(env, name)) for name in ("fcfs", "spt", "edd", "health")}
    if include_solver:
        methods.update({"cp_static_oracle": lambda: CPSATPolicy(False, solver_seconds), "cp_rolling": lambda: CPSATPolicy(True, solver_seconds)})
    for label, directory in (models or {}).items():
        methods[label] = lambda directory=directory: load_policy(directory)
    for label, factory in methods.items():
        for i, seed in enumerate(seeds):
            metrics, env = rollout(config, factory(), seed, instance=instance, record=i == 0 and (label in ("health", "ppo_joint_s0") or label.startswith("joint")))
            rows.append({"policy": label, "scenario_seed": seed, **metrics})
            if env.record:
                pd.DataFrame(env.trace).to_csv(output / f"trace_{label}.csv", index=False)
                pd.DataFrame(env.events).to_csv(output / f"events_{label}.csv", index=False)
        print(f"Evaluated {label}: {len(seeds)} {split} scenarios", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "episodes.csv", index=False)
    summary = frame.groupby("policy").agg({name: ["mean", "std"] for name in frame.select_dtypes("number").columns if name != "scenario_seed"})
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output / "summary.csv")
    (output / "protocol.json").write_text(json.dumps({"split": split, "scenario_seeds": seeds, "solver_time_limit_seconds": solver_seconds,
        "model_directories": {k: str(Path(v).resolve()) for k, v in (models or {}).items()}, "instance": instance,
        "latency_scope": "policy call includes feature and mask extraction; excludes simulator integration and model load"}, indent=2))
    return frame


def paired_comparison(frame, reference="ppo_joint", seed=42):
    """Cluster by training seed, not thousands of correlated decisions."""
    work = frame.copy()
    work["method"] = work.policy.str.replace(r"_s\d+$", "", regex=True)
    work["training_seed"] = pd.to_numeric(work.policy.str.extract(r"_s(\d+)$")[0]).fillna(-1).astype(int)
    work = work[work.training_seed >= 0]
    averaged = work.groupby(["method", "training_seed"]).full_objective.mean().unstack(0)
    rows = []
    rng = np.random.default_rng(seed)
    if reference not in averaged:
        return pd.DataFrame()
    for method in averaged.columns:
        if method == reference:
            continue
        differences = (averaged[reference] - averaged[method]).dropna().to_numpy()
        if not len(differences):
            continue
        bootstrap = rng.choice(differences, size=(2000, len(differences)), replace=True).mean(axis=1)
        rows.append({"reference": reference, "other": method, "training_seeds": len(differences),
                     "mean_difference": differences.mean(), "ci_low": np.quantile(bootstrap, 0.025), "ci_high": np.quantile(bootstrap, 0.975),
                     "interpretation": "negative difference favours joint; pilot uncertainty with only five training seeds"})
    return pd.DataFrame(rows)


def sensitivity(config, models, output, episodes=10, instance=None):
    rows = []
    for wear in (0.0, 0.5, 1.0, 2.0):
        for demand in (0.0, 50.0, 100.0, 200.0):
            condition = replace(config, wear_strength=wear, demand_rate=demand)
            for label, directory in models.items():
                policy = load_policy(directory)
                for seed in TEST_SEEDS[:episodes]:
                    metrics, _ = rollout(condition, policy, seed, instance=instance)
                    rows.append({"wear_strength": wear, "demand_rate": demand, "policy": label, "scenario_seed": seed, **metrics})
    Path(output).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(output) / "sensitivity.csv", index=False)


def transfer(config, models, output, episodes=10, instance=None):
    if instance:
        raise ValueError("Larger-shop transfer requires generated scenarios, not a fixed-size instance")
    rows = []
    for machines, jobs in ((5, 20), (5, 50), (5, 80), (10, 60)):
        condition = replace(config, machines=machines, jobs=jobs, wear_strength=0.5 if jobs == 20 else 2.0 if jobs == 80 else 1.0)
        for label, directory in models.items():
            policy = load_policy(directory)
            for seed in TEST_SEEDS[:episodes]:
                metrics, _ = rollout(condition, policy, seed)
                rows.append({"machines": machines, "jobs": jobs, "policy": label, "scenario_seed": seed, **metrics})
    Path(output).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(output) / "transfer.csv", index=False)
