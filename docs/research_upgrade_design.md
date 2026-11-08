# Research upgrade: learning when to inspect a factory machine

Status: authorised for autonomous implementation on 30 October 2026. The design
is a specification, not a claim that every experiment is complete. See the plan
ledger and outputs/inspection_research/report.md for actual implementation and
evidence status.

## Purpose and contribution

The user wants a stronger RL project and places no deadline constraint on the
research scope. The project should remain understandable to a beginner, retain
the existing implementation and measured results, and avoid unnecessary files.

The proposed question is:

> When is it worth paying for a machine inspection before choosing production
> or maintenance, when machine condition is uncertain and wear, unfinished work,
> and electricity demand history persist across shifts?

The candidate contribution is a reproducible investigation of this interaction,
including the operating regimes in which adaptive inspection adds value. It is
not a claim to invent PPO, inspection planning, partially observed control,
counterfactual explanation, or demand-charge optimisation. No world-first or
superiority claim is justified before further literature comparison and results.

Builder choices below are recommendations from Codex, not preferences explicitly
selected by the user. They make the scope concrete enough to review and build.

## Existing foundation and compatibility

The existing Python project has a one-minute factory simulator, operation routes,
alternative machines, setups, usage-related wear, stochastic failures, fixed
15-minute demand windows, MaskablePPO, a masked DQN, scheduling rules, a CP-SAT
comparison, tree extraction, and paired scenario evaluation.

Current verified longer experiments use one training seed. PPO matches EDD on
aggregate measurements and selects no preventive maintenance. These are baseline
findings to preserve, not evidence that maintenance can never help.

Keep the existing `FactoryEnv`, old configurations, commands, observation schema,
saved models, and outputs usable. Build a separate inspection environment that
shares small physical/accounting helpers where their contracts are identical.
Do not silently turn old saved models into the new observation/action schema.
No new mandatory package is required: NumPy, Gymnasium, PyTorch, SB3-Contrib,
pandas and matplotlib are already available.

The new path needs its own schema version, config validation, training metadata,
CLI mode and model compatibility checks. Old commands remain the old experiment.

## Scope

Required:

1. Persistent factory state across operating shifts and billing boundaries.
2. Hidden stochastic machine deterioration and a documented condition belief.
3. Costly inspections that provide imperfect measurements but do not repair.
4. Dispatch, maintenance, inspection and bounded waiting decisions, including
   opportunities when a machine has no job immediately ready.
5. Observation-only baselines and a belief-augmented MaskablePPO controller.
6. Action comparisons estimated with simulator branches and explicit uncertainty.
7. Ablations, measurement-quality/cost experiments, repeated training seeds and
   a presentation-ready research report based on actual measurements.

Deferred: a new GNN or multi-agent architecture, industrial safety guarantees,
hard failure constraints, a physical digital twin, real-factory deployment,
spare-parts inventory, AGVs, renewables/storage, and LLM-generated explanations.
Existing CP-SAT and tree extraction may remain old-experiment tools; adapting them
is optional after the core new experiment passes its checks.

## Calendar, persistence and terminal semantics

Primary research setting: two consecutive 30-day synthetic billing periods, one
eight-hour operating shift per calendar day, initially 14:00–22:00. This spans
the existing normal/peak tariff boundary. Weekends and holidays are not modelled.
Development uses shorter billing periods explicitly labelled accelerated cycles.
Use 30 orders per operating day in the initial primary configuration and 240
minutes as the mean arrival-stream span, retaining one to three operations and
the legacy duration scale. This lower starting load is a builder choice intended
to leave capacity for service, not an empirical factory specification. Check
backlog stability on training/validation sequences before freezing the primary
load. Report deliberately overloaded cases separately; do not optimise the test
load to favour a controller.

Use absolute integer calendar minutes. Preserve the following at shift closure:
health, belief particles, jobs and due dates, operation progress, setup/product
state, unfinished service work, and the current billing peak. Do not call a fresh
environment reset at a shift boundary.

During closure the factory is powered down: no production, inspection,
maintenance or repair progress; no wear; zero power in the primary shutdown
model. Job due-date clocks and lateness continue in calendar time. All active
tasks pause and resume at reopening. Continuous-running counters reset at closure.
Inspection results arrive only when inspection service completes. This shutdown
model is an explicit assumption, not a calibrated industrial overnight model.

Fixed demand windows are aligned to absolute midnight in 15-minute intervals.
Account for any partial occupied window using the specified shutdown power for
the remaining minutes. At midnight on a billing boundary, close the previous
ledger and open a new peak ledger. Never erase accrued demand charges. Calendar
minute, tariff, billing-period index, day in period and time to closure are public.

Generate a finite, seeded daily arrival stream for the entire episode; expose
only arrived orders and the configured arrival law. Base job durations remain
Taillard-derived with separate source groups for train/validation/test. Shift the
generated arrivals into operating time. Keep the legacy due-date formula in
calendar minutes: arrival + nominal remaining work * Uniform(1.7,3.0) + 30, with
the same integer rounding as the generator. A due date falling in closure is
allowed and is treated equally for all policies. Resample a day's finite arrival
stream if its final arrival would fall outside that day's operating shift,
using a daily job-only random channel; describe this as a conditioned synthetic
arrival stream, not an unconditioned Poisson process. Record backlog growth to
detect an unstable workload.

End the RL episode only at the configured full horizon, not at each shift and
not when the currently released queue becomes empty. A full research horizon is
a finite decision problem. Realise all accrued costs and charge the existing
unfinished-order penalty once at its end. Do not add a reward for maintenance or
an invented terminal health bonus. Preserve terminal/no-bootstrap treatment for
the genuine horizon; ordinary shift and billing transitions are nonterminal.

Finite-horizon bias still exists near the final day. Report it, compare 30-, 60-
and 90-day horizons in a sensitivity study, and distinguish first-period from
last-period behaviour. Do not claim that the finite model solves an infinite
continuing factory. Training discount remains 1 for the specified total-cost
finite problem; changing it is a separately named experiment.

## Hidden condition and belief

Retain continuous true health h in [0,1] as a simulator-only physical variable.
At episode start sample each machine's true health from a documented bounded
prior and its latent wear multiplier from a documented positive prior. The
initial particle belief uses the same training model, not the sampled truth.
Primary illustrative starting choices are Uniform(0.6,1.0) health and a lognormal
wear multiplier with mean 1 and log-standard-deviation 0.2. These values are
simulation assumptions and will be fixed in configuration before test evaluation.

Retain the existing load/continuous-runtime wear mean but multiply each active
minute's decrement by an unobserved positive mean-one Gamma shock (CV 0.35 as an
initial assumption). This prevents runtime history from revealing exact health.
Use the existing health-related failure hazard, clipped to [0,1]. PM restores
true health to 1; corrective repair restores it to 0.8 and resumes remaining work.
These known restoration outcomes update the belief too. No true health value,
latent wear multiplier or future shock is sent to ordinary policies.

Implement a bootstrap particle filter, initially 256 particles per machine,
jointly representing health and latent wear multiplier. It predicts using public
workload/power and the specified stochastic deterioration model; conditions on
observed survival or failure; and updates using inspection measurement likelihood.
Failure/survival are legitimate condition evidence. After known restoration,
reset the health coordinate while retaining/updating wear-rate uncertainty.
Resample when effective sample size falls below half the particle count; define
a stable fallback for numerically tiny likelihoods and record its use.

Use separate physical, observation and particle-filter random streams. A branch
or different inspection count must not consume the physical random stream in a
different order. Physical shocks are keyed by scenario, absolute minute,
machine and shock channel. This enables paired comparisons with action-dependent
failure hazards while preserving correct marginal distributions.

An inspection returns a health measurement with Gaussian noise clipped to [0,1]
at service completion. The filter uses censored-normal likelihood masses at the
two boundaries rather than treating clipped values as ordinary interior density.
Primary standard deviation 0.03, inspection duration five operating minutes, and
direct cost 100 synthetic INR are starting assumptions. All will be identified
as configured parameters and swept; inspection does not improve true health.

Validate belief quality with health-estimation error, interval coverage and
interval width on held-out trajectories. Introduce deterioration-model mismatch
as a separate robustness evaluation, not the default source of unlabelled errors.

## Decisions, observations and information boundary

The environment selects an idle machine in deterministic machine-index order;
the agent chooses the action. Keep the ten EDD-shortlisted eligible job slots.
Add preventive maintenance, inspection, and five-operating-minute wait/defer:
13 actions by default. WAIT applies to the selected machine only; other machines
can be assigned at the same minute. The defer clock prevents an infinite
same-minute loop and is interrupted by shift closure as documented.

Present decisions for idle machines with work ready and, even without work,
at public periodic control opportunities (initially every ten operating minutes).
Arrivals and task completions are decision triggers. Paused/busy machines are
not eligible for conflicting actions. A no-job idle machine can be inspected or
maintained. Inspection and PM consume their configured service time and energy.
Use existing maintenance/repair service power initially; log inspection power.

Legal masks may depend on public availability, machine mode, task compatibility,
service capacity if later introduced, and elapsed public cooldowns. They must not
depend on true health, the latent wear multiplier or unobserved noise. Replace
the old true-health maintenance threshold in the new environment. A just-serviced
machine may be masked using a fixed public cooldown, initially 30 operating
minutes, rather than a true-health test. Inspection repeat cooldown is five
operating minutes from completion. Do not mask an action merely because it is
estimated to be economically unattractive.

Use a schema-described flat observation with global calendar/billing/queue
features, job candidates and padded machine slots. Replace exact health with
posterior mean, standard deviation, selected quantiles, time since measurement,
and a measurement-present flag. Include public selected-machine identifiers,
availability/defer/service status, observed operating history, accrued demand
peak and window energy, tariff, and time remaining. Compute the exact new input
size from one ordered schema at implementation time; do not call it 233 features.

The belief is an approximate filter of condition. The complete factory
observation still compresses backlog/routes and is not proven to be a sufficient
Markov state. Describe the controller as belief-augmented under partial
observability, not an exact POMDP solver. Any sensor history received by a baseline
must equal the history available to the corresponding RL condition.

Provide a policy-facing observation/mask interface. Simulator truth and realised
future shocks may appear in a clearly separate evaluator/debug record but must
not enter ordinary baseline rules, neural inputs, feature scaling, adaptive
action masks, or counterfactual recommendations.

## Objective and accounting

Use negative incremental normalised realised cost. Preserve named old cost
scales where comparable, and add a separately named inspection term:

    J = lateness/480 + energy_charge/1000 + accrued_added_demand_charge/3000
        + 0.2*(PM_minutes + repair_minutes + inspection_minutes)/60
        + inspection_direct_cost/1000 + 2*terminal_unfinished_jobs

The direct inspection cost and its time penalty are distinct configured
consequences, both reported. There is no arbitrary health or inspection-use
reward. Terms use synthetic assumptions; J is not INR. Report physical lateness,
energy, bills and service time separately so trade-offs remain visible.

For billing period b with carried-in peak P0_b, accrue demand liability only by
positive increments of P_b. Sum increments into a lifetime ledger and retain it
when the period peak resets. A complete new period starts at zero unless an
explicit partial-period initial condition is requested. Never charge the full
same peak again at each shift. Energy is integrated per minute with that minute's
tariff. Sum of rewards must equal minus final cumulative J from the episode's
initial accounting state, including boundary and terminal charges.

Downtime, backlog age, energy and demand must keep accruing appropriately during
WAIT; policies cannot improve scores by silently discarding work. Report costs
per released job and completion/on-time rates as well as totals. Do not select a
winner solely from a lower combined cost if completion materially deteriorates.

## Controllers and experimental comparisons

All main comparisons use identical jobs, physical-shock distributions, horizon,
legal actions, metering and cost scales. Policy observation access is explicit:

* EDD + belief-mean maintenance threshold, with no inspection.
* SPT including setup + the same maintenance mechanism, with no inspection.
* Fixed-period inspection + EDD and belief-based maintenance.
* Uncertainty-threshold inspection + EDD and belief-based maintenance.
* MaskablePPO with belief features but inspection disabled.
* MaskablePPO with adaptive inspection: proposed controller.
* Exact-health MaskablePPO: privileged-information reference, not an optimum.

Tune rule intervals/thresholds, PPO checkpoints and hyperparameters on training
and validation scenarios only. Baseline threshold candidates and tuning budget
must be declared before test. Include a breakdown-only reactive baseline to
establish whether preventive maintenance has economic value under each setting.
If a stronger non-RL one-step or short-rollout belief controller is practical,
include it with its simulation/time budget. Do not present the existing CP-SAT
surrogate as an optimal solver for the expanded stochastic objective.

Start with belief-augmented MaskablePPO using an MLP. Compare larger networks or
masked DQN only after correctness and the inspection mechanism are validated.
No separate recurrent/GNN algorithm is required for the core research question.

Study gates:

1. Deterministic toy cases prove that sensing can change the best feasible action
   and has both positive- and negative-value regimes. With no uncertainty or
   useless measurements, inspection should not be assumed beneficial.
2. Compare sensible PM and reactive rules to establish where PM actually pays.
   Record parameter regimes with no benefit instead of adjusting test rewards
   until the proposed method wins.
3. One-seed smoke training verifies valid actions and stable finite accounting.
4. A multi-seed pilot estimates training/evaluation cost and convergence before
   the larger experiment. The no-deadline preference does not imply unlimited
   machine resources; measure them and report run duration/actual step counts.
5. Primary study: at least five independent training seeds, 500 held-out scenario
   sequences if feasible after the measured pilot, and validation-selected
   checkpoints. Use learning curves and a declared step-budget schedule rather
   than assuming the old 200k steps is sufficient for the longer horizon.

Use distinct master sequence seeds for train, validation and test. No daily
subseed or Taillard source group may cross splits. Derive daily jobs, physical
noise, initial conditions, inspection noise and filter streams from stable
named seed channels. Record configs, schema, package versions, source hashes,
training steps, checkpoint selection and scenario manifests.

## Research outputs and ablations

Primary outputs: cost components, completion/on-time rate, total/mean/p95
lateness, backlog, breakdowns, PM/inspection counts, downtime, energy, each period's
peak and demand charge, inference time, and worst-10%-scenario cost (an evaluation
tail metric, not evidence of CVaR-trained control).

Quantify uncertainty over both training seeds and paired scenarios. Use a
hierarchical bootstrap that resamples training seeds and shared scenario indices;
show the seed-level spread. A scenario-only interval must be labelled as such.
When comparing many regimes, report all tested conditions and distinguish
exploratory from predeclared primary comparisons.

Core ablations: inspection disabled; fixed inspection; uncertainty-triggered
inspection; adaptive inspection; exact health. A separately trained one-shift
condition versus persistent shifts tests the horizon contribution. Suppressing
peak information tests billing-history access, not removal of the physical bill.
Disabling demand charges is a different dynamics/objective experiment.

Create an inspection-value map over inspection cost x measurement noise, and
selected slices for wear variability, workload pressure, and existing peak
headroom. Keep a small predeclared grid practical; validate/tune on separate
scenarios and use held-out scenarios for reported performance. The report should
show where inspection is worthwhile and where it is not, not assume all settings
will benefit. Report uncertainty and training-seed variation on main comparisons.

Robustness: shifted wear-rate distribution, noisier-than-trained inspections,
different job load and initial machine condition, and different horizon lengths.
Do not conflate evaluation under shift with successful adaptation or retraining.

## Branch-based action comparisons

Two explicit modes prevent hindsight from becoming a recommendation:

1. Deployable model estimate: sample latent health/wear and future disturbances
   from the current permitted belief and known stochastic model. Force each legal
   first action, then use the same declared continuation policy over a fixed
   calendar horizon. Use matched sampled worlds across actions. No hidden true
   health or realised future shocks may enter. Include accrued costs, estimated
   tail cost/unfinished-work accounting, Monte Carlo uncertainty and simulation
   latency. Short-horizon estimates are not guarantees about the whole episode.
2. Retrospective evaluator audit: compare actions from the actual simulator
   snapshot using its truth and held-out future shock table. Label this privileged
   simulator hindsight; never feed it into the acting policy or tuning on test.

Use structured wording such as 'under the declared model and rollout horizon,
inspection had lower estimated cost, with interval ...'. State when estimates
overlap. The continuation policy must not get future knowledge, and every branch
must have its own valid belief update. Paired random numbers reduce variance;
they do not prove a real-world causal effect or explain neural internals.

## Tests and acceptance

Meaningful tests must cover:

* Shift close/reopen preserves state, pauses tasks and advances due-date clocks.
* Billing rollover resets only the period peak; lifetime demand liability never
  decreases; fixed-window metering is correct across calendar boundaries.
* Inspection consumes time/cost and updates belief without restoring health.
* Zero-information measurements do not create artificial information gain;
  exact known restoration produces the specified belief update.
* Two hidden worlds with the same public belief/history produce identical policy
  observations, masks and ordinary baseline choices until public evidence differs.
* No same-minute WAIT loop, no conflicting tasks, and service opportunities exist
  without a ready job. Late/unfinished work is never silently dropped.
* Rewards telescope to realised objective; only the genuine horizon is terminal
  for learning. No shift-boundary bootstrap/terminal mistake.
* Scenario and branch randomness are deterministic and independent of the order
  in which alternatives/controllers are evaluated.
* Belief estimation/coverage is assessed empirically, not asserted from the
  particle count. Particle degeneracy is detected and visible.
* Existing old-environment tests and saved-model loading still pass unchanged.
* New observation/action schema mismatch is rejected before model evaluation.

Engineering completion requires passing old and new meaningful tests, Gymnasium
checks, a reproducible smoke run and verified saved outputs. Research completion
additionally requires the declared multi-seed comparisons, uncertainty analysis
and a report that includes negative findings. No minimum improvement percentage
is an acceptance condition; a null result is valid if methods and evidence hold.

## File and presentation discipline

Keep this as the single design document. Add only necessary implementation,
configuration and test modules during construction. Use one named output root for
the upgraded study with a bounded manifest, not a directory per hypothetical
idea. Preserve all previous results. Do not generate a new PPT in the design stage.
Once the experiment is verified, update presentation evidence and speaker material
to distinguish current results, new measured results, and remaining limitations.

## Literature boundaries and sources checked

These are foundations/comparisons, not assertions of exact feature copying:

* Saifullah, Andriotis and Papakonstantinou (2023), value of information in DRL:
  https://research.tudelft.nl/en/publications/the-role-of-value-of-information-in-multi-agent-deep-reinforcemen/
* Koutas, Bismut and Straub (2024), inspection and maintenance under erroneous
  observations: https://link.springer.com/article/10.1186/s43065-024-00098-9
* Mehdiyev, Majlatow and Fettke (2024), job-shop counterfactual explanations:
  https://www.dfki.de/en/web/research/projects-and-publications/publication/14967
* Jiang and Shang (2026), energy-aware FJSP with PM and finite AGVs:
  https://www.sciencedirect.com/science/article/pii/S0957417425046391
* Tariff-aware manufacturing load shifting with PPO (2026):
  https://www.mdpi.com/1424-8220/26/6/1858
* MIRA (2026), safety-constrained joint maintenance and production rescheduling:
  https://www.mdpi.com/1999-5903/18/8/430

The targeted search found overlapping components. It does not establish that
their exact combination here is novel or rare. A focused full-text comparison
matrix is a research deliverable before strong contribution wording is used.
