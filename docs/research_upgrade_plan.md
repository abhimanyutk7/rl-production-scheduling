# Inspection study: implementation record

This file keeps the original checklist and dated progress notes. The planned
training and evaluation round is complete. The entries describe what was known
at each stage, including failed checks and earlier restrictions on presentation
work. Read the [final research report](../outputs/inspection_research/report.md)
for the current conclusions.

**Goal:** Build and evaluate adaptive inspection under uncertain machine health,
persistent production state and electricity demand history.

**Implementation approach:** Keep the earlier FactoryEnv and its saved models
usable. Add a separate environment for persistent factory state, hidden condition
and inspection. Connect it to the particle estimate, public-information rules,
PPO training, matched evaluation and branch comparisons.

**Tech stack:** Existing Python, NumPy, Gymnasium, SB3-Contrib, pandas, matplotlib.

**Spec:** [research_upgrade_design.md](research_upgrade_design.md).

## Requirements used during implementation

- Preserve old models, commands and outputs; no mandatory dependencies added.
- 13 masked actions: 10 job candidates, PM, inspection, bounded wait.
- Gamma 1, negative incremental realised cost; genuine horizon only is terminal.
- Hidden health never controls ordinary observation/action masks or baselines.
- Shift/billing transitions preserve state and accrued liabilities.
- No reward bonus for inspection/maintenance and no invented performance claims.
- Scenarios, observation noise and filter streams are independently reproducible.
- Only necessary source, tests, configs, report and one experiment output tree.

## Review focus

1. Zero-noise and clipped sensor measurements need correct likelihood handling.
2. A closure during service must pause progress, preserve belief and age jobs.
3. Empty queues must permit service and must not end a multi-day episode.
4. Different inspection/action orders must not change physical shock sequences.
5. Billing resets must not refund previously accrued peak charges.

## Tasks

### 1. Configuration and condition inference

Files: create `factory_rl/belief.py`, `factory_rl/inspection.py`; test in
`tests/test_inspection.py`. InspectionConfig extends validated ShopConfig;
HealthBelief provides predict, update, restore, summary and independently seeded
resampling. Exact signatures are documented in the implemented class API.

- [x] Add tests for invalid configs, informative/no-information/censored sensors,
  survival evidence, restoration and deterministic streams; observe failure.
- [x] Implement stable particle likelihoods, ESS resampling and summaries.
- [x] Run focused tests, record the result below.

### 2. Persistent physical environment

Files: `factory_rl/inspection.py`, `tests/test_inspection.py`.
InspectionEnv(config, record=False, exact_health=False, inspections_enabled=True)
has standard Gymnasium reset/step, action_masks(), observation_schema, metrics(),
snapshot() and seeded belief-compatible branches.

- [x] Add failing tests for inspection time/cost, shift/billing carry, waiting,
  no-ready-job service, privacy, reward telescoping and paired randomness.
- [x] Implement calendar jobs, physical states, condition filters, accounting,
  public masks, finite-horizon terminal semantics and explicit observation schema.
- [x] Pass focused tests and Gymnasium checks; run legacy tests.

### 3. Controllers, training, evaluation and comparisons

Files: create `factory_rl/inspection_study.py`; modify `factory_rl/cli.py` only
to register an inspection command; extend `tests/test_inspection.py`.
Public controllers accept policy observations/belief summaries only. Model
metadata validates schema/config identity. Experiment outputs are resumable.

- [x] Write failing tests for no hidden access, stable evaluation, mismatched
  checkpoints, paired statistics and branch isolation.
- [x] Add EDD/reactive/belief-threshold/periodic/uncertainty policies, masked PPO,
  manifests and paired metrics, inspection-value grids and belief calibration.
- [x] Add model-based branch comparisons with a declared continuation policy,
  uncertainty, and separate retrospective privileged audits.
- [x] Pass tests; verify CLI help, model save/reload and a small train/eval run.

### 4. Mechanism and research experiments

Files: `configs/inspection.json`, one root `outputs/inspection_research/`.

- [x] Test informative and uninformative inspection regimes, PM economics,
  backlog stability, billing-history sensitivity and belief calibration.
- [x] Run a measured multi-seed pilot before allocating larger budgets.
- [x] Tune baselines/checkpoints using validation only; freeze primary test setup.
- [x] Run feasible multi-seed held-out comparisons and selected cost/noise grids.
- [x] Label accelerated-cycle, short-training and full-cycle evidence separately.

### 5. Review and presentation evidence

Files: create `outputs/inspection_research/report.md` and verified figures;
update README and this ledger. No automatic PPT rewrite before results exist.

- [x] Obtain fresh independent whole-change review and fix material findings.
- [x] Run all meaningful tests and dependency/environment checks.
- [x] Report actual results, uncertainty, negative findings, provenance and
  candidate contribution versus closest prior work. Provide simple viva wording.

## Dated decisions and progress

ACTIVE 2026-11-02 renewed improvement request: user asks for research and
maximum improvement before retraining, GPU if possible, with little time.
The completed original studies remain frozen. No PPT or demo UI is authorised.

Research-led next protocol: a separately named hybrid residual-dispatch study.
Use an explicit public belief-maintenance/WAIT gate, shared candidate scoring,
fixed bounded inputs, and a frozen validation-retuned setup-aware SPT prior.
Initial deterministic decisions must exactly reproduce that prior; stochastic
training explores other jobs. Neural PPO learns dispatch only, not service.
Compare residual PPO with the frozen prior and a matched shared scorer without
the prior. Keep original SPT and original neural controllers as separate controls.
No claim to invent residual RL, shared scoring, PPO or service thresholds.

Primary references reviewed before retraining: Zhang et al.2020 Learning to
Dispatch, https://arxiv.org/abs/2010.12367 (shared learned dispatch representation);
Johannink et al.2018, https://arxiv.org/abs/1812.03201 (residual RL idea, robotics,
not scheduling evidence); SB3 custom policies and MaskablePPO documentation.
The 2023 paper titled Residual Scheduling uses residual problem graphs; it must
not be misrepresented as the same residual-to-heuristic policy design.

Hardware verified: RTX4050Laptop6GB, driver576.57 supportsCUDA12.9. Original
torch2.14.1+cpu cannot useCUDA. Install officialtorch2.11.0+cu128 in a separate
.venv-gpu, reuse existing other packages through a local .pth, preserve .venv.
Verify GPU arithmetic, model reload, dependency checks and actual PPO throughput.
Small-network PPO can be CPU-bound; report measured throughput and chosen device.

Pre-training gates: tests for full-episode exact prior matching, service legality,
public-only inputs, bounded60day features, permutation-consistent shared scoring,
forward/evaluate/predict distribution agreement, effective residual gradients and
serialization. Engineering sequences 24400-24401; new validation10200-10209;
held-out primary tests24500-24549. Train0-999 only. Three training seeds,
65536 PPO steps per condition, same budgets. Validation selects checkpoints,
including stepzero; never use test outcomes to tune. Samefreshgrid for allcontrols.
No superiority claim until completed paired seed/scenario evaluation and review.

2026-11-02 preflight passed: 71tests in32.22s under .venv-gpu, dependencycheck
clean, GPU forward/backwardfinite, existingoriginalcheckpointreloadsuccessful,
originalfrozenhashunchanged. Newmodule+tests independently reviewed, noimportant
correctnessfinding. Initial engineeringrun failed on deepcopy of cached non-leaf
distributiontensors; preserved and labelled failed_engineering_check. Fixed using
explicitstate-dict inferencecopy with RNGpreservationregression. Successor
study_v2 completed12rows/2models, allledgersreconciled,zeroinvalid,stepzeroequals
strongprior. No efficacyclaim from that tinycheck.

2048step actualPPObenchmark: CPU 631.2steps/s, fullCUDA409.9, CUDAupdates579.0.
Selected CUDAupdates for explicit GPUgradient use with limited(~9%)overhead vs
CPU; CPUphysicalsimulationandper-decisioninference, CUDAminibatchupdates. All
device/Adamstate transitions passed two-cycle tests and independentcheck.
Maincommand: .venv-gpu/Scripts/python.exe -u -m factory_rl.residual_dispatch
--device cuda_updates --steps65536 --seeds3 --episodes50 --validation10 --envs4
--output outputs/inspection_research/residual_dispatch (actualflagswithspaces).
Freeze residual_dispatch.py and its priority/core dependencies while active.

Optional transfer protocol declared before main test inspection: use allthree
saved residual policies, the frozen strongprior, originalSPT and three saved
CEMpriority policies on ten new sequences 24600-24609, exact60day/30daybilling
config. No retraining or model/seed selection from these sequences. 80rows in a
separate residual_transfer stage; do not mix with main700rows or oldtransfer.
Do this only after main completes and its identities reconcile. Samefeature
schema, explicithorizonchange, all60daily/twobillingentries mustreconcile.

2026-11-02 additional TRAINING-only diagnosis: ten paired trajectories, seeds
0,17,83,317,777 withpriorandlastlegaljob controls.3573physicalactions,1200 genuine
multi-jobchoices,66.41%forced; eventaggregation yields1210outertransitions.
Independentdiagnostic verified everyphysicalaction/state/eventtrace/finalmetric
and rewardexactlyequal. Sourceevidence residual_verification/forced_decisions.json.
Firstmainresidualseedretainedstep0through65536steps, zeroargmaxchanges, tinyKL;
theforcedtransitionfindingdoesnotproveitistheonlycause. Mainremainsfrozen.

New separate event_dispatch protocol justified by these training-only findings:
aggregateconsecutivesingle-actiongatedtransitions AFTER eachlearneraction;
neveradvanceatreset/dropinitialcosts. Sumrewards,stopatnextchoice/truehorizon.
Same92featurepolicy andfixedservice; gamma1preservesrealisedobjective but GAE
lambda.98 now decaysperaggregateddecision. Usephysicalbudget65536 percondition
andseed,not65536outerchoices; n_steps128, batch256,4envs,otherPPOsettingssame.
Stop atfirstcompletedrollout reachingphysicalbudget, record boundedovershoot,
outerdecisions,physicalactions,realchoices,episodes,walltime andGPUupdates.
Freshvalidation10220-10229,test24700-24749,three seeds,two conditions. Controls
are frozenprior,originalSPT and both saved non-aggregated mainconditions onthis
samefreshgrid:700rows. This is a whole representation/training-protocol change,
not an isolated lambda/architectureclaim. Freeze before testresults. Train only
after maincompletes and its policy/control hashesarebound. No PPT/UI work.

Eventpreflight independentreviewclean: twoexactequivalencetests pass, tiny2048
physicalbudgetcheckcompleted2520trainedphysicalsteps,2524totalincludingprobe,
fourCUDAupdates,recorded472overshoot,correctstepzero/minvalidationselection.
WholeGPUruntime suite73passed14.66s. Source event_dispatch.py is now frozen.
Queuedhiddenlauncher outputs/inspection_research/event_dispatch/launch.py waits
forparentresidual_dispatch manifestcomplete,thenruns physical65536,seeds3,
episodes50,validation10. Inspect launcher.json/logs/actualprocess beforeaction.
Do not duplicate either run; monitorrecords preserve truecompleted/pending state.

2026-11-02 residual_dispatch COMPLETE:6models,700exactrows,50freshsequences,
3trainingseeds. All2100daily/1400billingrecords/formula reconcile,0invalid,
source/control/modelhashes,caches,checkpointselection andbootstrapreproduced.
Allsixmodels performed32CUDAupdates. Residual selected[0,0,0], cost 33.8021
exactlythefrozenretunedprior; doNOTcallthatPPOgain. Standalone no_prior selected
[65536,57344,16384], cost 37.9650 vslegacy46.3386 (~18.1%lower). Additional
exploratorydifference-8.3736,95%[-12.9778,-3.3631]. VersusSPT36.8158 difference
+1.1492[-2.1639,5.0655],uncertain; versusstrongprior difference+4.1629
[1.0322,7.9730],worse. CEM33.8501, no_prior-minusCEM+4.1149[.2935,8.2949].
Completionstandalone98.511% vslegacy97.104%; ontime55.92% vs50.34%. Noinspection
actions innewpolicies. This is the24500-24549 THREE-DAYstudy, nottheoriginal60day
dataset. Reproduciblechecksandvalidation-selectedactualneuralexample saved.

Ruling: Skip the optionalresidual_transfer plannedabove: allthree residual models
selectedstepzero andexactlyequalthefrozenrule. Repeating their60daytransfer
would add no neurallearning evidence. No transferstagehasbeenrunfor thisround.
Eventstudy nowstarted viaqueuedlauncher47212;parentcomplete andpreserved.

Methodbackground clarified: grouping forcedactions is establishedtemporal
abstraction, notanewalgorithm (Sutton/Precup/Singh1999,
https://www.sciencedirect.com/science/article/pii/S0004370299000521).
GAEtime-indexchangeisexplicitlyconsistentwithitsdefinition (Schulmanetal2015,
https://arxiv.org/abs/1506.02438). Neitherpaperprovesthisfactoryresult; actual
physical/objectiveequivalencewasverifiedinourpairedtrajectories.
OriginalCPUenvironmentcompatibilityalsoverified:72tests pass,1CUDA-only test
skipped16.89s. GPUruntime73pass; nooldsourcecorechanged.

Final model-selection protocol declared BEFORE finaltest24900+ is evaluated:
afterbothnewstudiescomplete, compare sixsavedstandaloneNNpolicies (plain/event,
3seedseach),3savedCEMpolicies and2rules on commonvalidation10240-10249. Select
bestNN andbestCEM by meanrawvalidationcost only; alsorecordbestoverallcandidate.
Lock model/weights hashes inselection.json BEFORE test. Four fixedcontrollers
(selectedNN,selectedCEM,strongretunedSPT,originalSPT) on50unusedsequences
24900-24949,200rows. Pairedscenario uncertaintyonly: thesepoliciesarefixed;
do notmislabelas3seedalgorithmcomparison orselect/testtuneafavourableseed.
This suppliesaready-to-useNNpath andindependentselectioncheck withoutmore
training orPPT/UI. Allpriorstages/negatives remainreported. Output final_selection.

2026-11-02 event_dispatch COMPLETE and independentlyaudited:6models700exactrows,
all2100daily/1400billingrecords/formula/source/control/modelhashes/cache/checkpoint
minima andindependentbootstrapmatch;0invalid. Event no_prior40.4457 vsplain
40.0628, difference+.3829 CI[-2.5096,3.6290]:noestablishedcostgain. All3event
residualsstep0=prior37.1056. Eventstandalone selectedoutersteps[15360,18432,12800].
Physicaltrainedcounts65638-66989,overshoot102-1453,discardedprobe4-65;GPUcalls
39-50each. Tests24700-24749areseparatefromparent24500+. Thisnegativefinding
mustremainvisible. No moretrainingplannedforthisround.
Finalselectionqueuedlauncher45224 nowactiveoncommonvalidation;codeindependently
reviewedclean.Selectionmuststaylockedbefore24900+tests. Awaititsactualartifacts.

FINAL2026-11-02 renewedimprovementroundCOMPLETE:12newCUDA-trainedmodels,
two700rowstudiesplus200finalfixed-policyrows=1600newheldoutrecords,110common
validationrowsseparately.73GPUtests pass;72CPUtests passand1CUDA-onlyskip.
All3resultstagesindependentlyaudited,zero invalid,everydaily/billing/objective
reconciled,source/control/modelidentities andselectedcheckpointsverified.

Finalvalidation10240-10249 selectedplainno_prior_s0(model65536step) asNN,
CEM_s1asreadablelearner,retunedsetupSPTasoverallwinner. ChoiceslockedBEFORE
24900-24949test. FinalcostNN39.7330,CEM34.6688,retunedSPT36.3081,originalSPT
37.7723. NN-minusprior+3.4249 CI[1.8746,5.3565]; NN-minusCEM+5.0642
[2.7689,7.4338]; NN-minusSPT+1.9607[-.4971,4.5842]. Fixed-policySCENARIO-only
uncertainty,nottrainingseed/selectionuncertainty. NoNNsuperiorityestablished;
doNOTreplaceglobalvalidationwinnerwithCEMbasedonitstestmean.
SelectedNNactualpredictionreplayedundereachCPUtorch2.14.1andGPUruntime2.11
withmatchingchosenaction/probabilities. Finalselection/selection.json bindsmodel
path+hash;decision_example.json usesactualselectedNN.Allprocessesexited,
errorlogsempty. Preserve18.1%oldNNgainfrom24500studyandallnegatives;nevermix
thosedataset/seedaverageswiththefinalfixed-policytest. Report/README/checks
updated. NoPPT/UIcreated. Deleteovernightautomationafterfinalfilechecks.

Closure2026-11-02: finalfile/processchecks passed; all3newstagescomplete with
independentreviewandcleanlogs; noPythontraining/evaluationprocessremains.
Automation complete-rl-research-upgrade-overnight DELETED throughapp tool after
verifiedcompletion. Reports/modelchoicesarefinalforthisround; awaituserrequest
beforeanyPPT/demoUIworkoradditionalresearchprotocol.

FINAL 2026-11-02: The original full_cycle completed all 15 models and exactly
12,500 rows (five seeds, 500 common sequences, 60 days, 30-day billing). No active
launcher/coordinator/workers remain; errors.log is empty. Reproducible independent
artifact checks verified zero invalid actions, exact unique grids, all 750,000
daily records and 25,000 billing entries, raw objective formula, cache/aggregate
agreement, unchanged frozen source, model/config hashes and every selected
validation checkpoint. Evidence: full_cycle/verification.json and verify_results.py.

Adaptive965.963 vsSPT652.847, no-inspection846.824, imitation991.631. Supplemental
adaptive-minus-SPT313.117, paired95%[277.388,355.844]; adaptive-minus-no-inspection
119.139[86.163,155.472]; adaptive-minus-imitation-25.667[-78.741,0.000]. No adaptive
superiority to strong rules or established added PPO benefit. Adaptive selected
steps[0,0,0,8192,0]; no-inspection[0,0,0,0,32768]; exact-health[0,8192,0,8192,0].
11/15 selected policies retain imitation. Coverage90 adaptive87.19%, SPT89.88%.
Completion99.807% vsSPT99.929%; on-time44.37% vs60.67%, not AI accuracy.

Ruling: Feasible planned research implementation, evaluation and reporting are
complete. Preserve negative inspection findings and separate positive priority
search evidence versus original but not tuned rules. Do not start deadline-driven
test-set tuning or manufacture a novelty claim. No presentation/UI work is
authorised. Delete the overnight monitor after final evidence review/reporting.
Older ACTIVE/RUNNING entries below are dated history, superseded by this entry.

2026-11-01 completed: critic_refinement 850 exact rows, zero invalid, ledgers
reconciled, source frozen. Revised actor-only44.0415, actor-critic44.2626,
imitation45.6760, legacyadaptive46.9490, SPT36.9042. All six PPO learners chose
nonzero updates. Actor-critic vs actor-only CI crosses zero; critic contribution
unestablished. Whole revised protocol improves legacy in exploratory comparison,
but remains weaker than SPT. Do not attribute whole-protocol changes to critic.

2026-11-01 completed: priority_search 550 exact rows, zero invalid, ledgers
reconciled, source frozen. Learned35.6562 vs originalSPT38.1834 (6.6% lower mean
normalised cost), paired difference -2.5272, 95%[-4.2650,-.7494]. Three optimiser
seeds/fifty new sequences, three-day accelerated billing. All learned policies
chose zero inspections; no inspection-value gain. New full suite63passed.

Supplementary priority_audit completed500rows: setup-weightedSPT comparator
selected on validation from15 settings gives35.9946, 99.22% completion. Learned
vs this comparator CI[-1.8871,1.3888] crosseszero. Energy/urgency/service fitted
parameter interventions do not isolate supported component gains. Preserve
these controls; do not claim learning beats all tuned rules. Audit was specified
before reading the primary test summary, but remains exploratory reuse.

Zero-shot priority_transfer completed via python -u -m factory_rl.transfer_study
(session24480). Same actual60day/30daybilling config as originalfullcycle,
three learned seeds plus originalSPT and validation-retunedsetupSPT, ten new
sequences 23300-23309,50verifiedrows,fourworkers. No transfer retraining or
selection. Learned592.431 vs originalSPT665.974 (about11.0% lower mean cost),
paired difference -73.543,95%[-100.594,-46.554]. RetunedsetupSPT597.897;
learned-minus-retunedCI[-25.904,20.637] crosseszero. All choosezeroinspections.
Independent review verified hashes, exactgrid,60dailyrecordsand2billingperiods
perrow, allcostsreconciled,zeroinvalid. This is separate from the original
five-seed 500-sequence neural study. Keep horizon/algorithm/evidence boundaries.
Audit identity implicitly imports currentpriorityimplementation; review verified
it matches the primary hash now. If policy code changes later, do not resume
this audit output; create a new named stage with explicit dependency hashing.

LATEST 2026-11-01: User has about 1.5 days and explicitly prioritised research
and substantive project improvement. Do not build or edit a PPT until asked.
Presentation/demo work is deferred. The original full_cycle remains frozen and
active; inspect actual processes before any resume.

Controller diagnosis: current actor-only imitation leaves all critic weights
random. Adaptive full-cycle seed zero validation rose from 907.62 before PPO
to about 159,727 afterward. Its maintenance imitation recall was only 7.6%,
despite high overall agreement. The EDD-based teacher cannot teach SPT dispatch.
These observations motivate controlled experiments; they do not prove which
mechanism caused collapse.

Separate refinement: factory_rl/refinement.py tests matched actor-only versus
actor-plus-critic pretraining from identical saved imitation actors. It uses
full on-policy Monte Carlo returns, positive reward scaling 1/days, gamma=1,
longer rollouts, smaller/clipped/KL-limited PPO updates, and persisted diagnostics.
Fresh validation starts at 10020; new held-out tests start at 21100, outside all
earlier test sequences (21000-21001 were engineering checks). Fixed three-day/accelerated-billing configuration comes
from final_pilot. A learned benefit and 60-day transfer remain unestablished.

Method sources: Kernbach et al. (2026), Actor-Critic Pretraining for PPO,
https://arxiv.org/abs/2602.23804, motivates fitting the critic from the pretrained
policy's own returns. Official SB3 guidance motivates scaling and controlled
updates: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
and https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html.
This is an adaptation of existing methods, not a new PPO algorithm.

Initial refinement checks: three targeted tests pass after observed initial
missing-module failures. They cover terminal-inclusive returns, objective-
preserving reward scaling, and critic fitting that leaves actor logits exactly
unchanged. Initial end-to-end verification exposed an optional report dependency;
remove that dependency rather than install an unnecessary package. Preserve
superseded verification outputs. Independently review before a larger run.

2026-11-01: Critic refinement passed independent review after fixing exploration
RNG matching, complete comparator hashes and cache row identity. Final small
verification completed 18 reconciled rows, zero invalid actions, both actors
retained step zero. Critic holdout MSE 113.27 -> 20.16 is calibration evidence
only. The main refinement launch PID is 49768, output critic_refinement, exact
command: python -u -m factory_rl.refinement --steps 65536 --seeds 3 --episodes 50
--validation 5 --workers 3 --output outputs/inspection_research/critic_refinement.
Freeze refinement.py during this run. Its checksums include frozen original core.

2026-11-01: Separate interpretable direct policy search addresses weak EDD
dispatch. priority_search.py learns eight coefficients/thresholds from full
episode objective using Gaussian cross-entropy search; the starting policy is
exactly SPT plus belief maintenance, verified through a full episode. It consumes
only public state. Training populations share scenario seeds <1000, validation
10100-10104 selects nominees, untouched main tests are 22300-22349 (earlier
engineering checks used 22100-22101 and 22200-22201). Three independent
optimiser seeds, eight generations, population 20, four training scenarios per
generation. Final small verification completed 14 reconciled rows, zero invalid.
Review fixed policy-artifact hashes in evaluation caches and ensured exported
job-score explanations correspond to actual job actions. No causal/optimality
or algorithm-novelty claim. Main launcher PID 46180. Exact command: python -u
-m factory_rl.priority_search --seeds 3 --iterations 8 --population 20
--train-scenarios 4 --episodes 50 --validation 5 --workers 4
--output outputs/inspection_research/priority_search. Freeze priority_search.py.
Report no-inspection fixed-parameter intervention separately from retraining.
Keep all previous results; no PPT work until the user asks.

LATEST 2026-10-30: 56 tests pass. The final_pilot completed 15 models at 65,536
steps each and 1,250 verified rows (50 scenarios, five neural seeds, nine methods).
All grids/ledgers reconcile and invalid actions are zero. Adaptive cost 47.6829,
SPT+belief-maintenance 38.1250. The additional adaptive-minus-SPT interval is
[7.15999,12.02364] around +9.55791; no claimed superiority. Four adaptive seeds
selected step zero. Main full_cycle remains active; do not duplicate it.

Ruling: A deterministic analytical information-value test supplies positive and
negative toy regimes, explicitly not factory results. Actual model diagnostics
and fresh-scenario confirmation did not confirm the hoped-for advantage. This
is a negative research finding, not an excuse to manufacture a winner.

Next after full_cycle: verify its complete data, update root report and plots,
assess whether further controller refinement is justified (e.g. critic warm
starting or a stronger teacher) under a new named protocol. Do not silently alter
the frozen study. Novelty remains a candidate formulation/regime investigation;
world-first status and a learned-policy advantage have not been established.

2026-10-30 final verification now 55 tests (new nested-interval test), pip check,
Gymnasium and source-freeze check pass; evidence saved to verification/checks.json.
The five-fresh-scenario confirmation completed 20 cases / 60 action rows, 128
matched worlds, contrasting fast-low-cost and default inspection under two
headroom values. It did not confirm an inspection advantage versus processing;
do not convert the earlier favourable exploratory means into a claimed result.
See confirmation/report.md. All tests/regime results are model-based.

2026-10-30 FINAL PROTOCOL: Core implementation is reviewed and frozen. Verification
completed 18 rows across two scenarios and nine policies, zero invalid actions;
all daily/billing ledgers reconcile. Full suite: 54 passed. Independent follow-up
review verified the final two fixes with no additional material finding.

ACTIVE PROCESSES (do not duplicate or edit hashed research source while running):
- full_cycle launched hidden with initial launcher PID 21496. Command:
  `python -u -m factory_rl.cli inspection --config configs/inspection.json --steps 65536 --seeds 5 --episodes 500 --workers 3 --stage research --output outputs/inspection_research/full_cycle`
  Check console.log/errors.log/run.json and the current Python process tree.
- final_pilot: same command with days=3, billing_days=2, particles=64, steps=65536,
  five seeds, 50 scenarios, three workers; output final_pilot. Session 63958.
- exploratory grid: `python -m factory_rl.inspection_analysis --explore --worlds 16 --output outputs/inspection_research/exploration`, session 42030.

Next continuation: inspect all actual process/manifests; finish analysis/reporting
from completed data; verify no invalid actions and reconcile ledgers; compare
adaptive versus imitation, simple rules and no-inspection with paired intervals.
Keep step-zero selection and negative findings visible. Do not call the project
research-complete while the full run is active. On completion update the root
report/README status, show verified output plots, and pause the overnight heartbeat.

Known scope still pending: independent confirmatory inspection-regime evaluation
if exploration suggests a useful regime, fuller parameter/horizon robustness,
and any final presentation update. Do not claim these ran merely because code
and primary experiments exist.

2026-10-30 exploration completed: 96 fully exported cells on diagnostic validation
scenario 10011, using a wider initial-health prior [0.2,1]. Inspection had lower
mean truncated cost than processing and maintenance in 33 cells, but neither
multiple-testing-adjusted evidence nor any cell with both paired upper bounds
below zero. Do not call this a confirmed advantage. Exploration is separate from
the main prior [0.6,1] and does not establish a learned policy improvement.

2026-10-30 review/fixes: 54 tests pass (29 new + 25 legacy); pip check passes.
Independent review found and regression-tested fixes for future-order leakage in
model-based branches, stale controller cache identities, missing learned seed
grids, and final daily ledger terminal reconciliation. All billing/daily ledgers
now survive evaluation. Full declared grids and checkpoint identities are checked.

Training ruling: Initial scratch PPO pilot exhibited WAIT-collapse. Preserve
its results as a negative exploratory finding. Neural models now use disclosed,
class-weighted imitation of public training-only rules, followed by PPO. The
pre-PPO checkpoint is included in validation selection and separately evaluated
as 'imitation'; selected_steps=0 is explicitly disclosed, never called RL gain.

2026-10-30 mechanism: The predeclared single-snapshot diagnostic completed
12 cost/noise/headroom cells, 24 matched worlds and 180-minute lookahead.
Inspection did not beat both alternatives in any cell. This is a negative
diagnostic, not proof that inspection never helps or a policy comparison. Do not
claim a demonstrated billing-history/inspection advantage from this output.

Ruling: The partially run validated_pilot was stopped after review so the final
protocol can be frozen before longer experiments. Its files remain, labelled
superseded_after_review, and must not be used as primary evidence. 'verification'
is the final small end-to-end check; 'full_cycle' will hold the frozen longer run.

2026-10-30 implementation: Tasks 1 and 2 core complete. Observed missing-module
tests fail, then pass; 20 focused tests now pass, 25 legacy tests passed before
the CLI integration, and Gymnasium check passed. New code is belief.py,
inspection.py and inspection_study.py; old FactoryEnv remains unchanged.
Task 3 core controllers/training/statistics/branch interfaces pass tests; smoke
training, final validation and independent review are pending.

Ruling: Unmasked external actions fall back to WAIT and increment invalid_actions
by default, allowing the Gymnasium checker; all research runs set strict_actions
and use masks, so any invalid research action fails visibly.

Ruling: Short branch estimates stop at an explicit calendar limit and exclude
unmodelled residual value. They must be labelled truncated cost estimates, not
claims about total downstream benefit. Actual study comparisons use full episodes.

2026-10-30: User authorised autonomous implementation and maximum completion.
Overnight heartbeat `complete-rl-research-upgrade-overnight` is active with hourly
checks until evaluation and reporting finish; inspect actual processes/manifests
before resuming work. Its original ten-check limit was removed on 30 October
while the frozen study was still progressing, so monitoring does not expire
before completion. Delete the automation after the final verified report.

Ruling: This directory is not a Git repository. Work directly in the authorised
workspace; do not initialise Git or fabricate commits/worktrees. Keep one ledger
here instead of Git-dependent execution scratch directories.

Ruling: User explicitly asked to proceed autonomously. No extra plan approval is
needed; implement natively, with a fresh final reviewer as the execution skill
requires. Feature superiority is an experimental question, not a completion gate.

Pre-flight: config/belief feed the environment; its public observation/mask API
feeds controllers, schema-checked models and branches. Experiment metadata must
record all three. Old ScenarioPool's genuine-horizon handling can be reused.
