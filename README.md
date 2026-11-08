# Reinforcement learning for production scheduling

This is our 22AIE401 term project. It studies a common factory decision: when a
machine becomes free, which waiting job should run next? A short job may need a
long setup, an urgent order may be close to its deadline, and running a worn
machine can cause a breakdown. Electricity prices and demand charges add another
reason to consider the whole schedule.

We built a simulator for this problem and compared a neural controller trained
with proximal policy optimisation (PPO), readable policy search and scheduling
rules. The newer neural model learns job priorities while
a declared rule handles service decisions. All reported factory outcomes come
from simulation.

## Team

| Member | University ID |
|---|---|
| Abhimanyu T K | AM.SC.U4AIE23102 |
| Adith Anish | AM.SC.U4AIE23103 |
| Vishnu J | AM.SC.U4AIE23160 |
| Swarag Padiyankunnath | AM.SC.U4AIE23058 |

## Submission files

- [Project report: 18 pages](outputs/term_submission/RL_Project_Report_IEEE.pdf)
- [LaTeX source, figures and class files](outputs/term_submission/RL_Project_Report_LaTeX.zip)
- [Final presentation: 21 slides](outputs/term_submission/RL_Production_Scheduling_Final_Review.pptx)
- [Detailed experiment results](outputs/inspection_research/report.md)

The report uses the supplied IEEE template for course formatting. It is an
unpublished academic report.

## How a decision is made

The environment identifies a machine that needs work. It gives the controller
the available jobs, machine information, time, tariff and condition estimates.
The shared network scores each legal candidate. The environment then applies
the selected action, updates the factory and calculates the added cost.

During training, PPO samples legal actions and uses these experiences to update
the actor and critic. During evaluation, the saved policy chooses its
highest-probability legal action.

The latest observation has 92 values: ten candidate slots with eight stored
values each, plus twelve context values. For one candidate, the actor uses six
job features and the twelve context values. Its network is
`18 -> 64 -> 64 -> 1`. The same weights score every candidate. A separate critic
pools the job information to estimate future return.

The full inspection environment can dispatch, inspect, maintain or wait. The
newer neural experiment narrows this problem: a fixed public service rule
handles service and the neural actor ranks jobs. The cross-entropy method (CEM) is a separate controller
that searches for readable job-feature weights and service thresholds.

## Data and factory assumptions

Processing durations come from the Taillard generator and published seeds
distributed through [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html).
We sample and scale those durations, then generate arrivals, deadlines, product
types, routes and eligible-machine alternatives. Wear, failures, sensor noise
and electricity settings are simulation assumptions. We did not collect real
factory sensor measurements.

Duration families are separated: ta01-ta06 for training, ta07-ta08 for validation,
and ta09-ta10 for testing. The final validation sequences are 10240-10249; the
final fresh test uses 24900-24949. Scenario seeds control factory generation;
training seeds control learning randomness.

The research configuration has five machines, three product types and 30 new
jobs per day. The [sample CSV](outputs/dataset_sample_scenario_20000.csv) comes
from an earlier 50-job configuration. Its 200 rows describe operation-machine
alternatives, so several rows can belong to the same job.

## What the experiments showed

We selected policies on validation before running the final 50 fresh three-day
sequences. Lower combined cost is better.

| Controller | Mean cost | Jobs completed | Jobs on time |
|---|---:|---:|---:|
| Selected neural PPO | 39.733 | 98.42% | 49.20% |
| Selected CEM | 34.669 | 98.87% | 61.84% |
| Retuned setup-aware SPT | 36.308 | 98.87% | 60.13% |
| Original SPT | 37.772 | 98.51% | 60.38% |

CEM had the lowest test mean in this comparison. Retuned SPT was the overall
validation winner and remains the recorded selection. The scenario intervals
for these fixed policies do not include variation across training seeds or
uncertainty from selecting a model.

In an earlier matched three-day comparison, the revised neural protocol reduced
mean cost from 46.339 to 37.965, about 18.1%. Tuned SPT still did better at 33.802.
That change included the architecture, inputs and service handling, so the gain
cannot be assigned to the shared network alone.

The original 60-day study gave adaptive neural cost 965.963 versus SPT 652.847.
Four adaptive seeds kept their pre-PPO imitation checkpoint. Neither this study
nor the inspection diagnostics established an inspection benefit. The
[research report](outputs/inspection_research/report.md) keeps these negative
results alongside the later improvements.

Costs are scaled simulation objectives, not accuracy percentages or measured
money savings. The base reward is negative added cost divided by `reward_scale`.
The newer dispatch wrapper also divides by episode days. Evaluation tables use
the full episode cost. Costs from different horizons should be read within
their own experiment.

## Install and run

Use Python 3.12 for a fresh installation. Dependencies are listed in
`pyproject.toml`: NumPy, Gymnasium, PyTorch, Stable-Baselines3, SB3-Contrib,
OR-Tools, pandas, Matplotlib and scikit-learn. The test extra includes pytest.

On Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Review the saved submission without retraining:

```powershell
# Check the model hash, final-test rows and mean results.
& '.\.venv\Scripts\python.exe' scripts/review_submission.py
# Run eight environment decisions with the saved model, including forced actions.
& '.\.venv\Scripts\python.exe' scripts/review_submission.py --replay
# Run the tests. The CUDA-only check can skip on CPU hardware.
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
& '.\.venv\Scripts\python.exe' -m factory_rl.cli --help
```

The basic review command uses only Python's standard library. The optional
replay loads the trained policy and runs the simulator on CPU. It leaves the
saved experiment records unchanged.

The newer training used an RTX 4050 Laptop GPU for batch updates and the CPU
for simulation and small inference calls. A GPU is not needed for review.
The manifests record the original package versions; repeating training exactly
also depends on those versions and the seeds.

To run a new experiment:

```powershell
# Original single-shift controller.
& '.\.venv\Scripts\python.exe' -m factory_rl.cli train --config configs/shop.json --steps 200000 --output outputs/new_joint
& '.\.venv\Scripts\python.exe' -m factory_rl.cli evaluate --config configs/shop.json --models joint=outputs/new_joint --episodes 20 --output outputs/new_joint_evaluation
# Shared scorer, using the preserved pilot and priority controls.
& '.\.venv\Scripts\python.exe' -m factory_rl.residual_dispatch --device cpu --steps 65536 --seeds 3 --episodes 50 --output outputs/inspection_research/new_shared_study
```

Training can take a long time. Use a new output directory when changing a
protocol; the frozen stages reject incompatible settings. The repository
includes the code, saved controllers, compact historical results and all 200
final-test rows. Large raw historical evaluation tables remain outside Git.

## Where to read the code

| File | What to look for |
|---|---|
| `factory_rl/inspection.py` | Factory transitions, actions, reward and costs |
| `factory_rl/belief.py` | Machine-condition estimates |
| `factory_rl/instances.py` | Duration and job generation |
| `factory_rl/residual_dispatch.py` | Shared actor, critic, PPO settings and training |
| `factory_rl/event_dispatch.py` | Grouping forced transitions |
| `factory_rl/priority_search.py` | Readable CEM policy search |
| `factory_rl/refinement.py` | Actor/critic pretraining comparison |
| `factory_rl/inspection_study.py` | Original study and paired statistics |
| `factory_rl/agents.py` | Scenario pools and earlier PPO/DQN |
| `factory_rl/baselines.py`, `factory_rl/cli.py` | Rules, solver and original commands |
| `scripts/review_submission.py` | Saved-result checks and model replay |
| `configs/`, `tests/` | Factory settings and implementation tests |

## Ideas we used

PPO, GAE, masking, shared dispatch, CEM and particle filtering come from existing
work. The [source map](SOURCES.md) connects those ideas to our implementation.
Our work brings them into this factory formulation and tests the resulting
controllers under declared protocols.

The [design](docs/research_upgrade_design.md) explains the assumptions. The
[implementation record](docs/research_upgrade_plan.md) and
[earlier development notes](docs/experiment_history.md) preserve the decisions
and interim results. Use the final report and
[locked selection](outputs/inspection_research/final_selection/selection.json)
for the current conclusions.
