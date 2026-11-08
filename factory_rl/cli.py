import argparse
from dataclasses import replace
import importlib.metadata
import json
import math
from pathlib import Path
import pandas as pd
import torch
from .config import ShopConfig
from .agents import train, TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS
from .evaluate import evaluate, sensitivity, transfer, paired_comparison, rollout
from .distill import distill
from .instances import import_benchmark
from .plots import plot_results


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="Joint production/maintenance/demand-peak RL research")
    sub = parser.add_subparsers(dest="command", required=True)
    from .inspection_study import add_arguments, run_cli
    add_arguments(sub.add_parser('inspection', help='Persistent scheduling with uncertain health and paid inspection'))
    for command in ("train", "evaluate", "study", "distill"):
        p = sub.add_parser(command)
        p.add_argument("--config")
        p.add_argument("--instance")
        p.add_argument("--output", default=f"outputs/{command}")
        if command != "train":
            p.add_argument("--episodes", type=positive, default=500 if command == "study" else 20)
        if command in ("train", "study"):
            p.add_argument("--steps", type=positive, default=200000)
        if command == "study":
            p.add_argument("--seeds", type=positive, default=5)
        if command == "train":
            p.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
            p.add_argument("--objective", choices=["joint", "wear", "peak", "lateness"], default="joint")
            p.add_argument("--seed", type=int, default=0)
        if command in ("evaluate", "study"):
            p.add_argument("--solver-seconds", type=float, default=0.1)
            p.add_argument("--skip-solver", action="store_true")
        if command == "evaluate":
            p.add_argument("--models", nargs="*", default=[], metavar="LABEL=PATH")
            p.add_argument("--split", choices=["validation", "test"], default="test")
        if command == "study":
            p.add_argument("--extended", action="store_true", help="Run the 2x2 attribution training, parameter sweep, transfer, and mixture comparisons")
        if command == "distill":
            p.add_argument("--model", required=True)
            p.add_argument("--training-episodes", type=positive, default=50)
    p = sub.add_parser("import")
    p.add_argument("path")
    p.add_argument("--format", choices=["jssp", "brandimarte"], required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("plot")
    p.add_argument("directory")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'inspection':
        run_cli(args)
        return
    if args.command == "plot":
        plot_results(args.directory)
        return
    if args.command == "import":
        data, config = import_benchmark(args.path, args.format)
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "instance.json").write_text(json.dumps(data, indent=2))
        config.save(directory / "config.json")
        return
    if getattr(args, "episodes", 1) > 500:
        parser.error("episodes must not exceed the 500 held-out scenarios")
    if not math.isfinite(getattr(args, "solver_seconds", 1)) or getattr(args, "solver_seconds", 1) <= 0:
        parser.error("solver-seconds must be positive")
    config = ShopConfig.load(args.config)
    if args.command == "train":
        result = train(config, args.output, args.algorithm, args.objective, args.seed, args.steps, args.instance)
        print(f"Saved {args.algorithm} {args.objective}: {result['actual_steps']} steps, {result['training_seconds']:.1f}s")
    elif args.command == "evaluate":
        models = dict(item.split("=", 1) for item in args.models)
        evaluate(config, args.output, models, args.episodes, args.split, args.solver_seconds, not args.skip_solver, args.instance)
        plot_results(args.output)
    elif args.command == "distill":
        distill(config, args.model, args.output, args.training_episodes, args.episodes, args.instance)
    else:
        if args.seeds < 5:
            parser.error("study requires at least five training seeds; use train for a single-model smoke run")
        study(config, args)


def study(config, args):
    from .agents import load_policy
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    (output / "scenarios.json").write_text(json.dumps({"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS}, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib", "ortools", "scikit-learn")}
    (output / "run.json").write_text(json.dumps({"steps_per_model": args.steps, "training_seeds": list(range(args.seeds)), "evaluation_episodes": args.episodes,
        "extended": args.extended, "versions": versions, "status": "running", "provenance": "Controlled simulation with Taillard-derived duration values; no factory calibration."}, indent=2))
    models = {}
    for seed in range(args.seeds):
        for algorithm, objective in (("ppo", "joint"), ("ppo", "wear"), ("ppo", "peak"), ("dqn", "joint")):
            label = f"{algorithm}_{objective}_s{seed}"
            directory = output / "models" / label
            print(f"Training {label}: {args.steps} steps", flush=True)
            train(config, directory, algorithm, objective, seed, args.steps, args.instance)
            models[label] = directory
    evaluate(config, output / "validation", models, args.episodes, "validation", args.solver_seconds, False, args.instance)
    frame = evaluate(config, output, models, args.episodes, "test", args.solver_seconds, not args.skip_solver, args.instance)
    paired_comparison(frame).to_csv(output / "paired_comparison.csv", index=False)
    distill(config, models["ppo_joint_s0"], output / "distillation", training_episodes=20, test_episodes=args.episodes, instance=args.instance)
    if args.extended:
        ablations = []
        for seed in range(args.seeds):
            for visible in (False, True):
                for dependent in (False, True):
                    condition = replace(config, health_visible=visible, load_dependent_failure=dependent)
                    label = f"health{int(visible)}_dependent{int(dependent)}_s{seed}"
                    directory = output / "attribution" / label
                    if visible and dependent:
                        directory = models[f"ppo_joint_s{seed}"]
                    else:
                        print(f"Training attribution {label}", flush=True)
                        train(condition, directory, "ppo", "joint", seed, args.steps, args.instance)
                    policy = load_policy(directory)
                    for scenario in TEST_SEEDS[:args.episodes]:
                        metrics, _ = rollout(condition, policy, scenario, args.instance)
                        ablations.append({"health_visible": visible, "load_dependent_failure": dependent, "training_seed": seed, "scenario_seed": scenario, **metrics})
        pd.DataFrame(ablations).to_csv(output / "attribution.csv", index=False)
        print("Running wear/tariff sensitivity sweep", flush=True)
        sensitivity(config, models, output, min(5, args.episodes), args.instance)
        if not args.instance:
            print("Testing workload levels and ten-machine transfer", flush=True)
            transfer(config, models, output, min(5, args.episodes))
        mixtures = []
        for seed in range(args.seeds):
            wear, peak = load_policy(models[f"ppo_wear_s{seed}"]), load_policy(models[f"ppo_peak_s{seed}"])
            for weight in (0.25, 0.5, 0.75):
                for scenario in TEST_SEEDS[:args.episodes]:
                    import numpy as np
                    rng = np.random.default_rng(scenario + seed * 100000)
                    policy = lambda env, rng=rng, weight=weight: wear(env) if rng.random() < weight else peak(env)
                    metrics, _ = rollout(config, policy, scenario, args.instance)
                    mixtures.append({"training_seed": seed, "wear_policy_probability": weight, "scenario_seed": scenario, **metrics})
        pd.DataFrame(mixtures).to_csv(output / "mixtures.csv", index=False)
    plot_results(output)
    record = json.loads((output / "run.json").read_text())
    record["status"] = "complete"
    (output / "run.json").write_text(json.dumps(record, indent=2))
    print(f"Study finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
