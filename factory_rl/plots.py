from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(directory):
    directory = Path(directory)
    episodes = pd.read_csv(directory / "episodes.csv")
    episodes["method"] = episodes.policy.str.replace(r"_s\d+$", "", regex=True)
    training_seed = pd.to_numeric(episodes.policy.str.extract(r"_s(\d+)$")[0])
    episodes["seed"] = training_seed.fillna(-1).astype(int)
    episodes.loc[episodes.seed < 0, "seed"] = episodes.loc[episodes.seed < 0, "scenario_seed"]
    # Error bars for RL are across training seeds, not correlated decisions.
    independent = episodes.groupby(["method", "seed"])[["full_objective", "incremental_electricity_cost", "average_tardiness_min", "completion_rate"]].mean().reset_index()
    means = independent.groupby("method").mean(numeric_only=True)
    stds = independent.groupby("method").std(numeric_only=True).fillna(0)
    labels = list(means.index)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, metric, title in zip(axes.ravel(), ["full_objective", "incremental_electricity_cost", "average_tardiness_min", "completion_rate"],
                                 ["Full normalised cost (lower is better)", "Incremental electricity cost (simulated INR)", "Mean tardiness (minutes; unfinished are censored)", "Fraction completed within shift"]):
        ax.bar(labels, means[metric], yerr=stds[metric], color="#216869", capsize=3)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    title = "Mean cost | RL spread across training seeds; baseline spread across scenarios" if training_seed.notna().any() else "Mean and spread across unseen scenarios"
    fig.suptitle(title, fontsize=13)
    fig.savefig(directory / "comparison.png", dpi=150)
    plt.close(fig)
    for path in directory.glob("trace_*.csv"):
        trace = pd.read_csv(path)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
        colors = {"process": "#216869", "setup": "#f4b942", "maintenance": "#7768ae", "repair": "#bc4749", "idle": "#eeeeee"}
        for machine, rows in trace.groupby("machine"):
            groups = (rows["mode"].ne(rows["mode"].shift()) | rows["job"].ne(rows["job"].shift()) | rows["operation"].ne(rows["operation"].shift())).cumsum()
            for _, segment in rows.groupby(groups):
                axes[0].broken_barh([(segment.time.min(), len(segment))], (machine - 0.35, 0.7), facecolors=colors[segment.iloc[0]["mode"]])
            axes[2].plot(rows.time, rows.health, label=f"M{machine}")
        for mode, color in colors.items():
            axes[0].plot([], [], color=color, linewidth=6, label=mode)
        axes[0].set_ylabel("Machine")
        axes[0].legend(ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
        power = trace.groupby("time").power_kw.sum()
        axes[1].plot(power.index, power.values, color="#216869", label="Instantaneous kW")
        windows = power.groupby(power.index // 15).sum() / 15
        axes[1].step(windows.index * 15, windows.values, where="post", color="#bc4749", label="15-minute average (last window zero-padded)")
        axes[1].set_ylabel("Power kW")
        axes[1].legend(fontsize=8)
        axes[2].set_ylabel("Health")
        axes[2].set_xlabel("Minutes from shift start")
        axes[2].legend(ncol=5, fontsize=8)
        fig.suptitle(path.stem.removeprefix("trace_") + " | processing, electricity and health")
        fig.savefig(directory / (path.stem.replace("trace_", "schedule_") + ".png"), dpi=150)
        plt.close(fig)
