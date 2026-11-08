# Supplementary priority-controller audit



Exploratory reuse of the completed primary test grid. Baseline parameters selected using validation only. The setup-weighted SPT comparator searches 15 setup/maintenance combinations; this is a stronger comparator, not a new learning algorithm.

Other comparisons change fixed parameters of each learned policy without retraining. They measure interventions on those fitted policies, not the best achievable separately trained ablations. Comparisons are exploratory and not adjusted across all examined contrasts.

Selected baseline: setup weight 8, maintenance threshold 0.55.



|policy|full_objective|completion_rate|inspection_count|maintenance_count|breakdowns|
|---|---|---|---|---|---|
|belief|41.2315|0.981556|0|20.98|6.72|
|fixed_service|34.7941|0.992148|0|20.54|6.49333|
|learned_no_inspection|35.6562|0.990667|0|20.38|7.02667|
|learned_priority|35.6562|0.990667|0|20.38|7.02667|
|periodic|47.2322|0.978444|29.96|11.52|9.86|
|reactive|50.0269|0.968|0|0|16.36|
|retuned_setup_spt|35.9946|0.992222|0|11.02|9.52|
|spt|38.1834|0.988889|0|21|6.66|
|uncertainty|43.8491|0.978222|11.46|11.58|9.72|
|without_energy|35.0869|0.99163|0|20.3933|6.79333|
|without_urgency_age|35.592|0.991481|0|20.4533|6.88667|



## Learned minus reference cost with paired 95% intervals



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|learned_priority|retuned_setup_spt|-0.3384|-1.88711|1.38884|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|fixed_service|0.862069|-0.659172|2.46766|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|without_energy|0.569294|-0.108305|1.67224|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|without_urgency_age|0.064164|-0.976929|1.06201|3|50|hierarchical training-seed and paired-scenario bootstrap|