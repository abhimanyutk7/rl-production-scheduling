# Production scheduling with machine wear and demand peaks

## Completed neural improvement — 2 November 2026

Both shared-scorer experiments are complete in `residual_dispatch/` and
`event_dispatch/`, followed by a locked model-selection check in `final_selection/`.
Twelve new neural models and 1,600 new held-out rows were independently audited,
with zero invalid actions and reconciled costs. Original studies are preserved.
The latest suite passes 73 tests in the GPU runtime and 72 with one CUDA-only skip
in the original CPU runtime. Both dependency checks are clean.

The RTX4050 is enabled through the separate `.venv-gpu` Python. Training uses CPU
factory simulation and inference with CUDA gradient updates. A measured short
test found this faster than full-GPU execution for the small policy, but slightly
slower than pure CPU. GPU use is not presented as a guaranteed speedup.

Both new controllers learn job dispatch under an explicit public rule for
maintenance/WAIT. The initial residual policy reproduces a strong setup-aware
rule. Neither residual experiment established an improvement over that unchanged
prior; all six residual models retained step zero. The event
variant groups forced actions and records actual physical experience/overshoot.
See [current evidence and status](outputs/inspection_research/report.md).

The first new study has 700 independently verified rows. Standalone shared neural
PPO reduced mean cost from 46.339 for the old neural controller to 37.965 on the
same 50 fresh three-day sequences (about 18.1% lower). It remains worse than the
retuned rule at 33.802; its difference from original SPT at 36.816 is uncertain. All
three residual variants retained the strong starting rule, so their cost 33.802
does not demonstrate an added PPO gain. All selected steps, uncertainty and
limitations remain in the report.

The separate final selection compared 11 saved candidates on a common validation
pool, locked the choices, then tested four fixed controllers on 50 other fresh
sequences. The selected neural model is
[no_prior_s0/model.zip](outputs/inspection_research/residual_dispatch/models/no_prior_s0/model.zip).
It scored 39.733 versus 36.308 for the retuned rule, 34.669 for selected CEM and 37.772
for original SPT. No neural superiority is established. The overall validation
winner remains the retuned rule; test results do not change that selection.
[selection.json](outputs/inspection_research/final_selection/selection.json)
records the exact model/parameter identities. A saved
[actual neural decision](outputs/inspection_research/final_selection/decision_example.json)
shows masked probabilities and logits, without claiming causal explanation.

Replay the completed final check and selected-model example without retraining:

```powershell
& '.\.venv-gpu\Scripts\python.exe' outputs/inspection_research/final_selection/run_selection.py
```

## Verified completion — 2 November 2026

The planned inspection research evaluation is complete: **15 models, five training
seeds, 500 common 60-day scenario sequences and 12,500 verified rows**. All daily
and billing records reconcile, source/checkpoint identities match, and invalid
actions are zero. That stage used the earlier 63-test suite; the current 73-test
suite and dependency checks also pass.

The original adaptive neural controller has mean normalised cost **965.963**,
versus **652.847** for SPT plus belief maintenance and **846.824** for the
no-inspection neural controller. It does not establish an inspection advantage.
Four of five adaptive seeds retained imitation; all checkpoint choices and paired
uncertainty are disclosed in [the consolidated report](outputs/inspection_research/report.md).

Separate readable policy search improved on original SPT in exploratory tests
(6.6% lower mean cost on 50 three-day sequences; 11.0% on ten new 60-day
sequences). Its difference from a validation-retuned setup-aware rule remains
uncertain, and it used zero inspections. These are distinct protocols, synthetic
simulation findings and normalised objectives, not real-world money savings.
No PPT, presentation or demo UI was changed. Existing slides need a separately
requested update before they can represent this research round.

## Controller improvements — 1 November 2026

Substantive controller experiments now run separately from the frozen original
study. `factory_rl.refinement` compares identical imitation actors with and
without critic pretraining under controlled PPO updates. `factory_rl.priority_search`
learns a readable dispatch score and condition-service thresholds from episode
cost using cross-entropy policy search. The latter begins with the strong SPT
rule and can retain it if validation does not improve.

These methods use separate validation/test sequences, exact cache identities,
paired uncertainty and original-cost evaluation. New experiment directories are
`outputs/inspection_research/critic_refinement/` and
`outputs/inspection_research/priority_search/`. Their manifests determine actual
completion; engineering checks or validation gains do not prove test superiority.
See [the consolidated report](outputs/inspection_research/report.md) for measured
results and limits. PPT and presentation work is deferred until explicitly asked.

## Adaptive inspection upgrade — 30 October 2026

The new, separate experiment adds costly condition inspections, a stochastic
hidden wear model, particle-based condition estimates, explicit waiting, and
state that persists across operating shifts and billing periods. The old
environment, saved models and commands remain available.

Start with [the current research status and presentation guide](outputs/inspection_research/report.md).
The complete design is in [research_upgrade_design.md](docs/research_upgrade_design.md)
and the execution ledger is in [research_upgrade_plan.md](docs/research_upgrade_plan.md).

The original engineering checks passed 56 tests, dependency validation and the Gymnasium
environment check. A final two-scenario verification run has 18 evaluation rows,
zero invalid actions and reconciled daily/billing ledgers. Its
[report](outputs/inspection_research/verification/report.md) explicitly identifies
the imitation warm start and pre-PPO checkpoints. This smoke run is not research
evidence of superiority. The first controlled inspection diagnostic was negative.

The completed longer experiment is under `outputs/inspection_research/full_cycle/`:
60 simulated days, 30-day billing periods, five training seeds, 500 held-out
scenario sequences, and 65,536 PPO steps per neural model. Consult `run.json`,
`console.log` and `errors.log` for the recorded run. The independently checked
artifact evidence is in `verification.json` and `verification_review.json`.
The final shorter pilot is separately labelled under `final_pilot/`.
That pilot has now completed five training seeds, 50 test sequences and 1,250
verified rows. The adaptive controller did not beat the SPT/belief-maintenance
rule. Its [report](outputs/inspection_research/final_pilot/report.md) and the
[consolidated interpretation](outputs/inspection_research/report.md) retain this
negative result rather than presenting unproven superiority.

```powershell
# Test both old and new implementations
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider

# Completed study: exact identity can reuse its caches; no new training is needed
& '.\.venv\Scripts\python.exe' -m factory_rl.cli inspection --config configs/inspection.json --steps 65536 --seeds 5 --episodes 500 --workers 3 --stage research --output outputs/inspection_research/full_cycle

# Rebuild the automatic report/plot (preserve the separately maintained audit notes)
& '.\.venv\Scripts\python.exe' -m factory_rl.cli inspection --stage report --output outputs/inspection_research/full_cycle
```

An OS lock rejects a second coordinator for the same study. Source/config/step/
seed identities and checkpoint checksums prevent incompatible cache reuse; use
a new named stage when changing the protocol. Do not edit research source while
a frozen study is running. Partial model training restarts; completed matching
models and individual scenario evaluations can be reused.

Implementation of the 22AIE401 project described in **RL report MAchine.pdf** and **22AIE401_RL_Formulation_102.docx**. The Word formulation controls the action/state design. This is a controlled research simulation, not a calibrated factory controller. A joint-policy advantage is a hypothesis to test, not a built-in result.

## Verified run: 29 October 2026

The current run is `outputs/revision_2026-10-29/`. It corrects PPO shift-end bootstrapping, passes 25 tests and the Gymnasium/dependency checks, and preserves earlier outputs. Four models trained for about 200,000 steps each with seed 42. Rules and all four models were evaluated on the same 500 test scenarios. A separate short extended pilot completed 35 models across five training seeds.

| Method | Mean full cost | Completed within shift | Mean maintenance calls |
|---|---:|---:|---:|
| EDD | 38.312 | 75.32% | 0.000 |
| Health rule | 38.773 | 76.07% | 3.402 |
| Joint PPO | 38.312 | 75.32% | 0.000 |
| Wear-focused PPO | 38.312 | 75.32% | 0.000 |
| Peak-focused PPO | 38.312 | 75.32% | 0.000 |
| DQN | 46.534 | 70.08% | 0.004 |
| Extracted tree | 37.560 | 75.82% | 0.000 |

`results.csv` is the compact nine-method table. `evaluation/` contains the full 500-scenario measurements and plots, `validation/` contains 20 separate validation scenarios, `solver_comparison/` contains the 20-scenario solver comparison, and `rule/` contains the readable tree and its paired 500-scenario evaluation. `run.json` records versions, source hashes, checks, and paired scenario uncertainty.

The longer runs use one training seed. The five-seed pilot uses only 4,096 steps per model and five scenarios per setting. These are different evidence levels. Joint PPO and its tree still choose no preventive maintenance on the main test set; the project does not yet establish a joint scheduling/maintenance advantage. The full longer five-seed study and peak-tariff-shift experiments remain future work. The existing presentation still shows the earlier results, so use the new table when discussing this run.

## Run on this computer

The isolated Python environment is already installed in `.venv`. From PowerShell in `C:\RL project`:

```powershell
# Verify the simulator and learning checks
& '.\.venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider

# Train one joint policy
& '.\.venv\Scripts\python.exe' -m factory_rl.cli train --config configs/shop.json --steps 200000 --output outputs/joint

# Compare it with rules and time-limited CP-SAT on unseen scenarios
& '.\.venv\Scripts\python.exe' -m factory_rl.cli evaluate --config configs/shop.json --models joint=outputs/joint --episodes 20 --output outputs/evaluation

# Extract and test a readable depth-4 rule
& '.\.venv\Scripts\python.exe' -m factory_rl.cli distill --config configs/shop.json --model outputs/joint --output outputs/rule
```

On another computer, use Python 3.12 or newer, create a virtual environment, and install the package:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

All experiment results go under `outputs`. The project has no web server, database, deployment setup, or unrelated application files.

## Reproduce the experiment

```powershell
# A quick five-seed pilot, including attribution, sensitivity and transfer
& '.\.venv\Scripts\python.exe' -m factory_rl.cli study --config configs/shop.json --steps 4096 --seeds 5 --episodes 5 --extended --solver-seconds 0.2 --output outputs/pilot

# Longer research run: 5 seeds, 500 validation and 500 test scenarios
& '.\.venv\Scripts\python.exe' -m factory_rl.cli study --config configs/shop.json --steps 200000 --seeds 5 --episodes 500 --extended --solver-seconds 1 --output outputs/research
```

The extended study trains joint, wear-focused and peak-focused MaskablePPO agents and a masked DQN for each seed. It also trains the 2x2 attribution agents: direct health features visible/hidden, crossed with load-dependent/purely random failures. Every method is scored using the same full objective. Three stochastic mixtures of the single-goal policies are compared; this finite grid cannot establish superiority over **every possible** weighting or switching strategy.

Scenario seeds are disjoint: training 0–999, validation 10000–10499, and test 20000–20499. Training draws repeatedly from the 1,000-scenario pool. Validation/test are regenerated deterministically rather than saved as thousands of files. Increasing steps changes how often training scenarios are seen; it does not enlarge the training pool. Training seeds control policy initialisation and training scenario order. The scenario seed fixes jobs, inherited peak, and machine/time failure shocks.

The parameter sweep tests wear strengths 0, 0.5, 1 and 2, crossed with demand rates 0, 50, 100 and 200. Transfer tests light (20 jobs/5 machines), heavy (50/5), stressed (80/5), and a larger shop (60/10) without retraining. To keep an extended sweep practical it uses at most five test scenarios per setting; the main evaluation uses `--episodes`. Sweep policies are evaluated under changed parameters without retraining at each setting. These are sensitivity and transfer checks; robust conclusions need more scenarios and separate retraining at the relevant parameter settings.

## Simulator and RL formulation

- **Decision events:** a machine is idle and has an eligible job. Simultaneous free machines are scheduled in machine-index order at the same simulation time. One-minute physical integration runs between decisions.
- **Actions:** `Discrete(11)` by default: ten eligible jobs sorted by due date, plus preventive maintenance. Masks reject empty shortlist slots and recently serviced/full-health maintenance. There is no explicit waiting action, following Part II. Peak spreading is therefore achieved through job choice, product power and maintenance; adding deliberate idling would be a separate action-space experiment.
- **State:** a normalised 233-feature vector for the default configuration: clock, tariff, current demand-window energy, billing peak, shop pressure, ten job candidates, and ten padded machine slots. Candidate features include processing time, slack after remaining work, setup, remaining operations and power. Padded machine slots keep the input size fixed for 5-to-10-machine transfer. Increasing `machine_slots`, `shortlist` or `products` changes the input size and requires retraining.
- **Health:** all machines start at 1. Health drops by `wear_strength * (wear_per_minute * kW / 20 + continuous_wear * continuous_minutes / 120)` in processing/setup. An idle minute, maintenance or repair resets the continuous-running counter. Failure probability per minute is `failure_base + failure_health * (1-health)`. In the random-failure ablation it is a constant `failure_base + 0.25 * failure_health`. The ablation does not match the realised breakdown rate exactly.
- **Maintenance/repair:** preventive maintenance takes 30 minutes and restores health to 1. Repair takes 90 minutes and restores health to 0.8. Interrupted setup or processing resumes from its remaining work; completed work is not discarded. Preventive and corrective downtime enter the downtime cost. Breakdowns have no extra reward penalty.
- **Setups:** switching product type costs `5 + 3 * ((new-old) mod products)` minutes; the same product and the first product need no setup. This is a sequence-dependent matrix expressed directly, with no duplicate stored matrix.
- **Arrivals/routes:** a finite sequence of exponential interarrival times supplies Poisson arrivals, with the first job available at time zero. Each job has one to three sequential operations; an operation has up to two eligible machines with different speeds. Due date equals arrival plus nominal remaining work times a random tightness factor, plus 30 minutes.
- **Metering:** integrate kW into kWh each minute, apply the tariff of that minute, and meter fixed aligned 15-minute averages. Demand peak starts at a carried-in month-to-date value. A partial final window is zero-padded, corresponding to shutting down after the run. Windows start at the episode boundary. There is no carry-in window energy or tariff-specific billing ratchet/minimum-demand contract.
- **Episode end:** the shift ends at 480 minutes, or earlier if every order is complete. Work is not drained beyond the shift. Incomplete work remains explicitly reported and receives a normalised terminal penalty. Makespan is censored at the shift end when work remains; unfinished-job tardiness is a lower bound. Completion rate must be read together with cost, so abandoning orders cannot appear to be a good schedule.

The shortlist and queue aggregates deliberately compress the complete simulator state. They are not a proof of a fully observable Markov state: hidden backlog routes, future arrivals and interrupted machine details are not all exposed. The practical policy therefore operates on a compact, partially observed scheduling state. Keeping the billing peak in the observation fixes the demand-history issue but does not eliminate all partial observability. Hidden-health attribution removes the direct health numbers; legal action masks and operating history can still provide indirect information about health.

## Reward and units

The joint reward is the negative increase in normalised full cost since the previous decision:

```text
lateness = tardiness_weight * cumulative job tardiness / tardiness_scale
energy = energy_weight * accrued energy charge / energy_scale
demand = demand_weight * demand_rate * (billing_peak - inherited_peak) / demand_scale
downtime = maintenance_weight * (preventive + corrective minutes) / downtime_scale
terminal = unfinished_penalty * unfinished_jobs, if the shift ends before completion
full cost = lateness + energy + demand + downtime + terminal
```

Tardiness is integrated as the number of overdue unfinished jobs each minute. For completed jobs this is exactly the sum of positive completion lateness. We use `gamma=1.0` for this finite-horizon task: summed rewards equal negative episode cost, including the telescoping peak increments. A discount below one would change the timing preference even though the undiscounted charge still telescopes.

All agents keep lateness and the incomplete-order penalty. The **wear-focused** agent additionally includes downtime, while the **peak-focused** agent additionally includes energy and demand. The **joint** agent includes all terms. This common production objective prevents single-goal agents from winning by avoiding production. The wear-focused variant does not add an invented direct health or breakdown penalty.

Default tariffs are synthetic INR: off-peak 4/kWh (22:00–06:00), normal 7/kWh, peak 12/kWh (17:00–22:00), and demand 100/kW. They are chosen experiment parameters, not an actual utility tariff. An 08:00 eight-hour default shift stays in the normal slab, so tariff shifting is not tested in the pilot. Change `start_hour` to 14 to span normal/peak hours. The inherited monthly peak is sampled uniformly from 20–50 kW; disable `random_initial_peak` to use `initial_peak_kw` exactly.

`incremental_electricity_cost` is the energy charge plus the extra monthly demand liability caused by this episode. `total_electricity_bill` includes the inherited demand liability too; it is not the whole month's energy bill. Each shift reset samples a controlled inherited peak rather than maintaining a multi-shift monthly ledger.

## Benchmarks and provenance

Duration values come from Taillard instances ta01–ta10 generated with the published seeds: ta01–ta06 for training, ta07–ta08 for validation, and ta09–ta10 for test. Values are sampled from their duration matrices, scaled by `benchmark_time_scale` (default 0.5), and adjusted for synthetic machine speed. The routes and flexible-machine alternatives of generated factory scenarios are synthetic. These are **benchmark-derived extended scenarios**, not unmodified Taillard instances; do not compare their results directly with published Taillard optima.

The generator is checked against the published 4x4 verification matrix in [OR-Library's Taillard generator and seeds](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop2.txt). See also the [OR-Library job-shop description](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html).

Import an existing JSSP pair-list or Brandimarte FJSP file to preserve actual operation routes/durations:

```powershell
& '.\.venv\Scripts\python.exe' -m factory_rl.cli import 'C:\path\Mk01.fjs' --format brandimarte --output outputs/benchmark
& '.\.venv\Scripts\python.exe' -m factory_rl.cli evaluate --config outputs/benchmark/config.json --instance outputs/benchmark/instance.json --output outputs/benchmark/evaluation
```

JSSP input: header `jobs machines`, then one line per job with zero-based `machine duration` pairs. Brandimarte input uses one-based machine IDs and operation/alternative counts. The importer expects a header followed by one job per line; native Taillard separate time/machine matrices must first be converted to pair-list format. Imported static instances disable wear/failures/setups by default and use a long horizon to test the core scheduling problem. Synthetic due dates, products and electricity parameters remain identified as extensions. No proprietary dataset is required or bundled.

## Baselines and learning

FCFS, SPT, EDD and the early health-threshold baseline share the same shortlist as the agent. The threshold rule sends the current machine for maintenance below 0.55 health when allowed, otherwise dispatches by SPT including setup. Part II mentions separate refusal and maintenance thresholds, but its `K+1` action space has no waiting/refusal action; this implementation uses one maintenance threshold so every baseline action is executable in the same space.

The static CP-SAT reference has oracle access to the fixed instance, including future arrivals. The rolling version sees only arrived jobs and replans on arrivals, breakdowns and maintenance decisions. Its deterministic model includes routes, machine alternatives, resource availability, precedence and sequence-dependent setups. A greedy feasible schedule supplies a complete search hint. It minimises `10 * total tardiness + makespan`; it **does not optimise stochastic wear or the full electricity bill**. A time limit with no incumbent falls back to EDD and is counted. Solver model construction and search are both included in policy latency. A CP-SAT plan is converted into shortlist dispatch priorities, so the replay is not guaranteed to preserve a globally optimal static schedule, especially if it requires deliberate idling. The test suite separately verifies CP-SAT's optimal makespan of 11 on Google's small static example. Do not describe the replayed baseline as an exact optimum for the full project objective.

PPO uses [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html), two 64-unit layers, CPU execution, 256-step rollouts and four update epochs. DQN uses a small PyTorch network, replay buffer, target network and Huber loss. Its masks apply to random exploration, greedy selection and next-state Bellman maxima; standard unmodified DQN does not provide this full masking path. The simulator reports shift-end truncation, and the training wrapper converts it to a terminal transition so PPO cannot bootstrap value beyond the finite shift. A regression test exercises PPO's actual rollout buffer. Models trained with this correction record `shift_end_is_terminal: true` in their metadata. Earlier models under `outputs/main` and `outputs/pilot` predate this correction and require retraining before use as corrected research results.

The distilled rule scores each legal job/maintenance candidate using a depth-4 decision tree and selects its highest probability of matching the teacher. The tree sees named physical features instead of opaque vector indices. Distillation collects demonstrations from training scenarios, measures action agreement on validation scenarios, then evaluates both teacher and rule on paired test scenarios. `compression.csv` reports the cost lost or gained; agreement alone is not proof of policy quality.

## Reading results

- `episodes.csv`: every method/scenario with makespan, tardiness, completion, breakdowns, utilisation, health, downtime, kWh, shift/billing peak, energy/demand costs and decision latency.
- `summary.csv`: mean and sample standard deviation across test scenarios for each saved model or baseline.
- `paired_comparison.csv`: joint-minus-other cost differences averaged within each training seed, then a paired bootstrap across seeds. Negative differences favour the joint model. Five seeds give limited uncertainty estimates.
- `validation/`: separate validation measurements; no test-based checkpoint or hyperparameter selection is performed.
- `models/`: weights, training history and model metadata. Training history logs each cost component separately.
- `distillation/`: the saved tree, readable rule, agreement metadata and paired compression evaluation.
- `attribution.csv`, `sensitivity.csv`, `transfer.csv`, `mixtures.csv`: generated by `--extended`.
- `comparison.png` and `schedule_*.png`: charts of measured results and representative processing/load/health traces.
- `run.json`: software versions, experiment size, provenance and whether the study finished. `scenarios.json` stores split seeds rather than thousands of duplicated instances.

Decision latency includes policy feature/mask preparation and solver construction/search but excludes environment integration and model loading. It is measured on the current computer, not a claim about industrial hardware. The small pilot validates the pipeline; its 4,096 steps per model are not evidence of convergence. Check whether the learned policies beat the health rule before writing a positive contribution claim. Larger experiments, reward/threshold tuning using validation only, retraining across sweep conditions and real factory calibration remain necessary for stronger research or deployment claims.

## Files

`factory_rl/` contains the simulator, generators, baselines, training, evaluation, distillation, plots and CLI. `configs/shop.json` is the editable experiment configuration. `tests/test_project.py` checks the scheduling/metering/masking invariants. `pyproject.toml` declares the package and dependencies. This README is the only project documentation file.
