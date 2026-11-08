# Hybrid neural dispatch: separate research protocol

Three-day synthetic episodes; accelerated two-day billing. Raw normalised objective is evaluated.
The neural policy learns job dispatch only. A fixed public belief rule controls maintenance/WAIT.
Shared candidate scorer, bounded residual to a previously validation-selected setup-aware SPT prior.
These established ideas are adapted here; no new-algorithm or inspection-value claim.

3 training seeds, 65536 requested PPO steps/condition, 50 fresh test sequences, 700 rows.
Training runtime: cuda_updates. CUDA-update counts and selected steps are persisted per model.

|policy|full_objective|completion_rate|on_time_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|changed_dispatches|dispatch_decisions|
|---|---|---|---|---|---|---|---|---|---|
|learned_priority|33.8501|0.990593|0.620222|0|20.54|6.3|0.896833|nan|nan|
|legacy_adaptive|46.3386|0.971037|0.503407|8.88|10.0733|11.92|0.857361|nan|nan|
|no_prior|37.965|0.985111|0.559185|0|11.2067|9.57333|0.894542|47.0533|179.82|
|residual|33.8021|0.990667|0.605333|0|11.3|8.84|0.8965|0|180.18|
|retuned_setup_spt|33.8021|0.990667|0.605333|0|11.3|8.84|0.8965|0|180.18|
|spt|36.8158|0.987556|0.628889|0|20.96|6.68|0.896833|nan|nan|

Negative paired differences favour residual PPO; seed/scenario uncertainty:

|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|residual|no_prior|-4.16291|-7.97301|-1.03218|3|50|hierarchical training-seed and paired-scenario bootstrap|
|residual|retuned_setup_spt|0|0|0|3|50|hierarchical training-seed and paired-scenario bootstrap|
|residual|spt|-3.01374|-4.91739|-1.13873|3|50|hierarchical training-seed and paired-scenario bootstrap|
|residual|legacy_adaptive|-12.5365|-15.0261|-10.0973|3|50|hierarchical training-seed and paired-scenario bootstrap|
|residual|learned_priority|-0.0480135|-1.78623|1.4719|3|50|hierarchical training-seed and paired-scenario bootstrap|

## Selected checkpoints

|condition|seed|selected_steps|validation_cost|gpu_update_calls|
|---|---|---|---|---|
|residual|0|0|34.8363|32|
|no_prior|0|65536|37.5261|32|
|residual|1|0|34.8363|32|
|no_prior|1|57344|37.7169|32|
|residual|2|0|34.8363|32|
|no_prior|2|16384|35.0387|32|

Step zero means the initial rule/scorer was retained; never call that an added PPO gain.
The initial deterministic residual policy equals its prior, but stochastic training explores alternative jobs.
The no-prior arm uses the same service gate, bounded scorer and training budget, without job-priority logits.
Legacy neural controls have different architectures and training protocols; that comparison cannot isolate a single cause.
Low cost versus older policies does not establish improvement over the strong prior.
Synthetic simulation only; normalized cost is neither accuracy nor money savings. Test outcomes must not tune this protocol.

Sources: https://arxiv.org/abs/2010.12367 ; https://arxiv.org/abs/1812.03201 ; https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
