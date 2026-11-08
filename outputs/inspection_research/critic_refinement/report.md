# Controller calibration experiment



Separate exploratory protocol. Three-day accelerated billing; fresh test scenarios starting 21100. No real factory data.

Same imitation actor, fixed positive reward scaling 1/days, gamma=1, controlled PPO updates. Actor-critic adds full-episode on-policy value pretraining; actor-only omits it.

Pretraining adds simulation/computation and is not a matched total-compute comparison. Calibration and PPO hyperparameters are adaptations, not a new PPO algorithm.

A zero selected PPO step means the imitation actor was retained. Interpret cost with completion. Neither a lower point estimate nor a crossing interval proves superiority.



|policy|full_objective|completion_rate|maintenance_count|inspection_count|breakdowns|belief_coverage90|
|---|---|---|---|---|---|---|
|actor_critic|44.2626|0.973926|17.9867|6.14667|9.36667|0.836361|
|actor_only|44.0415|0.972963|13.7067|6.99333|10.5933|0.832153|
|belief|39.1686|0.980889|20.7|0|6.96|0.898125|
|imitation|45.676|0.972815|14.3933|11.56|10.4533|0.832264|
|legacy_adaptive|46.949|0.970222|10.0067|8.86667|12.0933|0.844806|
|periodic|45.9378|0.975556|11.78|29.98|9.62|0.80925|
|reactive|47.2747|0.964667|0|0|16.34|0.901292|
|spt|36.9042|0.986889|20.54|0|6.98|0.896458|
|uncertainty|43.4675|0.976889|11.98|11.5|9.94|0.787625|



## Paired cost differences (candidate minus reference; lower is better)



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|actor_critic|actor_only|0.221092|-1.39036|1.88583|3|50|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|imitation|-1.41338|-2.98946|0.218882|3|50|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|legacy_adaptive|-2.68643|-5.44263|-0.190423|3|50|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|spt|7.35843|5.43189|9.32026|3|50|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|belief|5.09403|3.1703|7.15153|3|50|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|uncertainty|0.795127|-1.28981|2.66596|3|50|hierarchical training-seed and paired-scenario bootstrap|



## Checkpoints



- actor_critic_s0: selected PPO step 32768; critic {'examples': 1310, 'updates': 800, 'mse_before': 87.94700622558594, 'mse_after': 0.16993582248687744, 'training_pool_holdout_mse_before': 113.2698745727539, 'training_pool_holdout_mse': 20.15892219543457, 'additional_simulation_decisions': 1639}

- actor_critic_s1: selected PPO step 16384; critic {'examples': 1277, 'updates': 800, 'mse_before': 113.9261474609375, 'mse_after': 0.17606014013290405, 'training_pool_holdout_mse_before': 33.97361373901367, 'training_pool_holdout_mse': 24.62380599975586, 'additional_simulation_decisions': 1660}

- actor_critic_s2: selected PPO step 49152; critic {'examples': 1397, 'updates': 800, 'mse_before': 88.32965087890625, 'mse_after': 0.04172665253281593, 'training_pool_holdout_mse_before': 121.18778228759766, 'training_pool_holdout_mse': 6.945842742919922, 'additional_simulation_decisions': 1698}

- actor_only_s0: selected PPO step 40960; critic not pretrained

- actor_only_s1: selected PPO step 57344; critic not pretrained

- actor_only_s2: selected PPO step 57344; critic not pretrained