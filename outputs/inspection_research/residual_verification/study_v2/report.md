# Hybrid neural dispatch: separate research protocol

Three-day synthetic episodes; accelerated two-day billing. Raw normalised objective is evaluated.
The neural policy learns job dispatch only. A fixed public belief rule controls maintenance/WAIT.
Shared candidate scorer, bounded residual to a previously validation-selected setup-aware SPT prior.
These established ideas are adapted here; no new-algorithm or inspection-value claim.

1 training seeds, 2048 requested PPO steps/condition, 2 fresh test sequences, 12 rows.
Training runtime: cuda_updates. CUDA-update counts and selected steps are persisted per model.

|policy|full_objective|completion_rate|on_time_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|changed_dispatches|dispatch_decisions|
|---|---|---|---|---|---|---|---|---|---|
|learned_priority|39.1941|1|0.605556|0|19.5|6.5|0.842708|nan|nan|
|legacy_adaptive|59.3866|0.966667|0.427778|11.5|25|6.5|0.835417|nan|nan|
|no_prior|54.5091|0.983333|0.522222|0|12.5|9.5|0.807292|83.5|188.5|
|residual|39.5713|0.994444|0.55|0|12|9|0.805208|0|189|
|retuned_setup_spt|39.5713|0.994444|0.55|0|12|9|0.805208|0|189|
|spt|48.4432|1|0.544444|0|24|6.5|0.84375|nan|nan|

Negative paired differences favour residual PPO; seed/scenario uncertainty:

|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|residual|no_prior|-14.9378|-16.1349|-13.7407|1|2|hierarchical training-seed and paired-scenario bootstrap|
|residual|retuned_setup_spt|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|
|residual|spt|-8.87192|-12.3029|-5.44095|1|2|hierarchical training-seed and paired-scenario bootstrap|
|residual|legacy_adaptive|-19.8154|-35.2383|-4.39242|1|2|hierarchical training-seed and paired-scenario bootstrap|
|residual|learned_priority|0.377152|-1.33388|2.08818|1|2|hierarchical training-seed and paired-scenario bootstrap|

## Selected checkpoints

|condition|seed|selected_steps|validation_cost|gpu_update_calls|
|---|---|---|---|---|
|residual|0|0|36.6368|1|
|no_prior|0|0|57.2697|1|

Step zero means the initial rule/scorer was retained; never call that an added PPO gain.
The initial deterministic residual policy equals its prior, but stochastic training explores alternative jobs.
The no-prior arm uses the same service gate, bounded scorer and training budget, without job-priority logits.
Legacy neural controls have different architectures and training protocols; that comparison cannot isolate a single cause.
Low cost versus older policies does not establish improvement over the strong prior.
Synthetic simulation only; normalized cost is neither accuracy nor money savings. Test outcomes must not tune this protocol.

Sources: https://arxiv.org/abs/2010.12367 ; https://arxiv.org/abs/1812.03201 ; https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
