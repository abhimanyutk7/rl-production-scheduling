# Physical-budget event-dispatch study

Same physical factory and public service gate; only consecutive forced actions are grouped for learning.
All costs remain included; gamma1 preserves the undiscounted objective. GAE lambda now decays per aggregated choice.
Physical-budget targets are matched to the preceding study; recorded rollout overshoots and update counts differ.
This is a complete representation/training-protocol comparison, not an isolated causal claim about one parameter.
700 exact held-out rows. Fresh validation10220+, test24700+.

|policy|full_objective|completion_rate|on_time_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|changed_dispatches|
|---|---|---|---|---|---|---|---|---|
|event_no_prior|40.4457|0.981185|0.597185|0|11.5867|10.4133|0.88575|45.0067|
|event_residual|37.1056|0.988444|0.589556|0|11.08|10.28|0.887|0|
|plain_no_prior|40.0628|0.984889|0.551852|0|11.4333|10.4533|0.887361|46.72|
|plain_residual|37.1056|0.988444|0.589556|0|11.08|10.28|0.887|0|
|retuned_setup_spt|37.1056|0.988444|0.589556|0|11.08|10.28|0.887|0|
|spt|38.8588|0.984|0.604889|0|21.06|7.3|0.88925|nan|

|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|event_residual|event_no_prior|-3.34011|-6.48902|0.247806|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_residual|plain_residual|0|0|0|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_residual|plain_no_prior|-2.95726|-6.72476|-0.0213457|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_residual|retuned_setup_spt|0|0|0|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_residual|spt|-1.75327|-3.67562|0.385392|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_no_prior|plain_no_prior|0.382857|-2.50956|3.62895|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_no_prior|retuned_setup_spt|3.34011|-0.247806|6.48902|3|50|hierarchical training-seed and paired-scenario bootstrap|
|event_no_prior|spt|1.58684|-2.15195|5.08411|3|50|hierarchical training-seed and paired-scenario bootstrap|

## Training and selected checkpoints

|condition|seed|selected_steps|physical_steps|trained_rollout_physical_steps|physical_overshoot|choice_steps|outer_steps|episodes|gpu_update_calls|
|---|---|---|---|---|---|---|---|---|---|
|event_residual|0|0|66711|66707|1171|20303|20484|177|40|
|event_no_prior|0|15360|65642|65638|102|24374|24580|202|48|
|event_residual|1|0|65972|65968|432|19793|19972|175|39|
|event_no_prior|1|18432|66065|66000|464|24373|24580|203|48|
|event_residual|2|0|66995|66989|1453|20813|20996|179|41|
|event_no_prior|2|12800|66340|66335|799|25393|25604|207|50|

Stepzero means the starting policy was retained; report it explicitly. Negative cost differences favour the candidate.
Learned service/inspection value, real-factory calibration, monetary savings and new-algorithm novelty are unestablished.
