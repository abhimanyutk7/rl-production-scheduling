# Reinforcement learning for production scheduling

22AIE401 term project: a simulated factory scheduler with setup, deadlines,
uncertain machine condition, wear and electricity demand charges. We implement
masked neural PPO, shared job scoring, readable CEM policy search and strong rules.
All factory outcomes are simulated.

## Team

| Member | University ID |
|---|---|
| Abhimanyu T K | AM.SC.U4AIE23102 |
| Adith Anish | AM.SC.U4AIE23103 |
| Vishnu J | AM.SC.U4AIE23160 |
| Swarag Padiyankunnath | AM.SC.U4AIE23058 |

## Final deliverables

- [18-page IEEE-template academic report](outputs/term_submission/RL_Project_Report_IEEE.pdf)
- [Complete LaTeX source ZIP](outputs/term_submission/RL_Project_Report_LaTeX.zip)
- [21-slide final presentation](outputs/term_submission/RL_Production_Scheduling_Final_Review.pptx)
- [Consolidated research report](outputs/inspection_research/report.md)
- [Locked selection](outputs/inspection_research/final_selection/selection.json)

The IEEE template is academic formatting; this is not an IEEE publication.

## Results and limitations

Four fixed validation-selected policies on 50 fresh three-day sequences:

| Controller | Mean cost (lower better) | Completed | On time |
|---|---:|---:|---:|
| Selected neural PPO | 39.733 | 98.42% | 49.20% |
| Selected CEM | 34.669 | 98.87% | 61.84% |
| Retuned setup-aware SPT | 36.308 | 98.87% | 60.13% |
| Original SPT | 37.772 | 98.51% | 60.38% |

CEM has the lowest observed test mean. The overall validation winner remains
retuned SPT; test results do not change selection. Fixed-policy scenario intervals
exclude training-seed and model-selection uncertainty.

A separate matched three-day study reduced mean cost from earlier neural 46.339
to revised neural 37.965 (about 18.1%); tuned SPT remained better at 33.802.
Architecture, inputs and service handling changed together. In the original
60-day study adaptive cost was 965.963 versus SPT 652.847; four adaptive seeds
retained pre-PPO imitation. Inspection benefit and neural superiority to strong
rules are unestablished. These are normalised simulation costs, not accuracy,
real currency savings, an optimality guarantee or industrial validation.

## Implementation and dataset

The simulator tracks ordered operations, setup, queues, hidden condition, service,
electricity and persistent billing. Particle filtering estimates condition;
action masks enforce legal choices. The latest neural wrapper stores 92 values.
The shared actor scores each job with six job features plus twelve context values:
`18 -> 64 -> 64 -> 1`; a separate pooled critic estimates return. A fixed public
rule handles service, so the newer neural study learns dispatch only.

MaskablePPO samples legal actions during training and updates actor/critic weights.
Deterministic evaluation chooses its preferred legal action. CEM separately
searches readable job-feature weights and service thresholds over whole episodes.
Base reward is negative incremental cost divided by `reward_scale`; the newer
wrapper also divides by episode days. Reported cost tables remain unscaled.

Durations derive from the Taillard generator and published seeds at
[OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html).
We sample/scale durations and generate routes, alternatives, arrivals, deadlines,
products, wear, failures and electricity assumptions. Real factory sensor data
and unchanged Taillard optimal-makespan comparisons are not used.
Duration families: ta01-ta06 training, ta07-ta08 validation, ta09-ta10 testing.
Final validation uses 10240-10249; final testing uses 24900-24949. The research shop
has five machines, three products and 30 jobs per day.
The [sample CSV](outputs/dataset_sample_scenario_20000.csv) is an earlier 50-job
configuration: its 200 rows are operation-machine alternatives, not 200 jobs.

## Install

Python 3.12 is recommended for a fresh installation. `pyproject.toml` declares
NumPy, Gymnasium, PyTorch, Stable-Baselines3, SB3-Contrib, OR-Tools, pandas,
Matplotlib and scikit-learn. The test extra adds pytest.

Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

CPU is sufficient for review. Recorded newer training used RTX 4050 Laptop CUDA
batch updates with CPU simulation/inference. GPU use is not a guaranteed speedup.
Manifests record original versions; exact training reproduction depends on them.

## Run without retraining

```powershell
# Verify the selected model hash, exact final-test grid and reported means.
& '.\.venv\Scripts\python.exe' scripts/review_submission.py
# Also execute eight real neural decisions on CPU.
& '.\.venv\Scripts\python.exe' scripts/review_submission.py --replay
# Run tests (a CUDA-only test can skip on CPU).
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider
# List commands.
& '.\.venv\Scripts\python.exe' -m factory_rl.cli --help
```

The default review uses the standard library and reads saved evidence. Optional
replay executes the saved neural model and simulator without modifying caches.

## Train new models

```powershell
# Original single-shift model, separate output.
& '.\.venv\Scripts\python.exe' -m factory_rl.cli train --config configs/shop.json --steps 200000 --output outputs/new_joint
& '.\.venv\Scripts\python.exe' -m factory_rl.cli evaluate --config configs/shop.json --models joint=outputs/new_joint --episodes 20 --output outputs/new_joint_evaluation
# Newer shared scorer; uses preserved pilot/priority controls in the snapshot.
& '.\.venv\Scripts\python.exe' -m factory_rl.residual_dispatch --device cpu --steps 65536 --seeds 3 --episodes 50 --output outputs/inspection_research/new_shared_study
```

Training can take substantial time. Existing frozen stages reject incompatible
identities; use a new directory for changed protocols. Large raw historical
evaluation tables are excluded from Git. Complete source, saved controllers,
compact historical evidence and all 200 final-test rows are retained.

## Code and evidence map

| Location | Purpose |
|---|---|
| `factory_rl/inspection.py` | Factory, condition dynamics, actions, reward and costs |
| `factory_rl/belief.py` | Particle condition estimation |
| `factory_rl/instances.py` | Duration and job generation |
| `factory_rl/residual_dispatch.py` | Shared scorer and PPO training/evaluation |
| `factory_rl/event_dispatch.py` | Grouped forced-transition experiment |
| `factory_rl/priority_search.py` | Readable CEM search |
| `factory_rl/refinement.py` | Actor/critic pretraining comparison |
| `factory_rl/inspection_study.py` | Original study and paired statistics |
| `factory_rl/agents.py` | Scenario pools and legacy PPO/DQN |
| `factory_rl/baselines.py`, `factory_rl/cli.py` | Rules, solver and original CLI |
| `scripts/review_submission.py` | Evidence verification and saved-model replay |
| `configs/`, `tests/` | Parameters and implementation checks |

Established PPO, GAE, masking, learned dispatch, CEM and particle methods are
credited in the report. Contributions concern integration, controller wrappers,
controlled protocols and audited evaluation. See the
[design](docs/research_upgrade_design.md), [ledger](docs/research_upgrade_plan.md),
[research report](outputs/inspection_research/report.md) and
[preserved history](docs/experiment_history.md). Historical notes contain dated
interim results; the final deliverables and locked selection are current.
