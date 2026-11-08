# Controller calibration experiment



Separate exploratory protocol. Three-day accelerated billing; fresh test scenarios starting 21000. No real factory data.

Same imitation actor, fixed positive reward scaling 1/days, gamma=1, controlled PPO updates. Actor-critic adds full-episode on-policy value pretraining; actor-only omits it.

Pretraining adds simulation/computation and is not a matched total-compute comparison. Calibration and PPO hyperparameters are adaptations, not a new PPO algorithm.

A zero selected PPO step means the imitation actor was retained. Interpret cost with completion. Neither a lower point estimate nor a crossing interval proves superiority.



|policy|full_objective|completion_rate|maintenance_count|inspection_count|breakdowns|belief_coverage90|
|---|---|---|---|---|---|---|
|actor_critic|57.1898|0.972222|19|7|10|0.90625|
|actor_only|57.1898|0.972222|19|7|10|0.90625|
|belief|38.891|0.994444|20.5|0|6.5|0.916667|
|imitation|57.1898|0.972222|19|7|10|0.90625|
|legacy_adaptive|57.1898|0.972222|19|7|10|0.90625|
|periodic|55.8513|0.983333|10|30|12|0.861458|
|reactive|53.8946|0.972222|0|0|18|0.873958|
|spt|34.8633|0.988889|19.5|0|7|0.895833|
|uncertainty|56.492|0.977778|9|10.5|14|0.821875|



## Paired cost differences (candidate minus reference; lower is better)



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|actor_critic|actor_only|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|imitation|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|legacy_adaptive|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|spt|22.3265|10.3281|34.325|1|2|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|belief|18.2988|11.0498|25.5478|1|2|hierarchical training-seed and paired-scenario bootstrap|
|actor_critic|uncertainty|0.697772|-3.57879|4.97433|1|2|hierarchical training-seed and paired-scenario bootstrap|



## Checkpoints



- actor_critic_s0: selected PPO step 0; critic {'examples': 1310, 'updates': 800, 'mse_before': 87.94700622558594, 'mse_after': 0.16993582248687744, 'training_pool_holdout_mse_before': 113.2698745727539, 'training_pool_holdout_mse': 20.15892219543457, 'additional_simulation_decisions': 1639}

- actor_only_s0: selected PPO step 0; critic not pretrained