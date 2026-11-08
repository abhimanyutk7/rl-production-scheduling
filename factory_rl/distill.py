import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from .agents import load_policy, TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS
from .env import FactoryEnv
from .evaluate import rollout


def rule_features(env, action):
    machine = env.machines[env.current_machine]
    candidates = env.candidates()
    values = [float(action == env.maintenance_action), machine.health if env.config.health_visible else 0.0,
              machine.continuous, env.config.tariff(env.time), env.peak_kw,
              env.window_kwh * 60 / env.config.demand_window, len(candidates)]
    if action == env.maintenance_action:
        return values + [0.0] * 5
    j = candidates[action]
    job, op = env.jobs[j], env.operation[j]
    work = sum(min(operation.values()) for operation in job.operations[op:])
    return values + [job.operations[op][env.current_machine], job.due - env.time - work,
                     env.setup_time(machine, job), len(job.operations) - op, machine.power * (0.8 + 0.2 * job.product)]


FEATURE_NAMES = ["is_maintenance", "machine_health", "continuous_minutes", "tariff", "billing_peak_kw", "window_load_kw", "candidate_count",
                 "processing_minutes", "slack_minutes", "setup_minutes", "remaining_operations", "job_power_kw"]


def readable_rule(tree):
    return ("Score every valid candidate with this depth-4 tree. Leaf weights are [negative, positive]; score = positive / (negative + positive).\n"
            "Choose the highest score. Maintenance is a candidate too; ties use EDD order.\n\n"
            + export_text(tree, feature_names=FEATURE_NAMES, show_weights=True, decimals=6))


class TreePolicy:
    def __init__(self, tree):
        self.tree = tree

    def __call__(self, env):
        actions = np.flatnonzero(env.action_masks())
        features = [rule_features(env, int(a)) for a in actions]
        probabilities = self.tree.predict_proba(features)
        positive = list(self.tree.classes_).index(1) if 1 in self.tree.classes_ else None
        if positive is None:
            return int(actions[0])
        return int(actions[np.argmax(probabilities[:, positive])])


def collect(config, teacher, seeds, instance):
    features, labels = [], []
    for seed in seeds:
        env = FactoryEnv(config, instance=instance, strict_actions=True)
        env.reset(seed=seed)
        while not (env.terminated or env.truncated):
            selected = teacher(env)
            for action in np.flatnonzero(env.action_masks()):
                features.append(rule_features(env, int(action)))
                labels.append(int(action == selected))
            env.step(selected)
    return features, labels


def distill(config, model_directory, output, training_episodes=50, test_episodes=20, instance=None):
    import joblib
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_policy(model_directory)
    features, labels = collect(config, teacher, TRAIN_SEEDS[:training_episodes], instance)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=0)
    tree.fit(features, labels)
    policy = TreePolicy(tree)
    agreement = []
    for seed in VALIDATION_SEEDS[:min(10, test_episodes)]:
        env = FactoryEnv(config, instance=instance)
        env.reset(seed=seed)
        while not (env.terminated or env.truncated):
            action = teacher(env)
            agreement.append(policy(env) == action)
            env.step(action)
    rows = []
    for seed in TEST_SEEDS[:test_episodes]:
        for label, chooser in (("network", teacher), ("tree", policy)):
            metrics, _ = rollout(config, chooser, seed, instance=instance)
            rows.append({"policy": label, "scenario_seed": seed, **metrics})
    pd.DataFrame(rows).to_csv(output / "compression.csv", index=False)
    joblib.dump(tree, output / "tree.joblib")
    (output / "rule.txt").write_text(readable_rule(tree))
    (output / "metadata.json").write_text(json.dumps({"teacher": str(Path(model_directory).resolve()), "validation_action_agreement": float(np.mean(agreement)),
        "training_scenarios": TRAIN_SEEDS[:training_episodes], "test_scenarios": TEST_SEEDS[:test_episodes], "depth": tree.get_depth()}, indent=2))
    return policy
