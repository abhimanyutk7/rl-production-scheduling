# Zero-shot longer-horizon transfer



Controllers trained/selected on three-day accelerated-billing episodes are evaluated without retraining on 60 simulated days and two genuine 30-day billing periods. New scenario sequences 23300-23309; three fitted search seeds. The two simple rules are deterministic comparators, not three independently trained models.

This is a small exploratory transfer check, not the separate original500-sequence neural study and not real-factory validation. Costs span60days and must not be compared directly with three-day costs. Intervals resample training seeds and paired scenario sequences; ten sequences give limited power.



|policy|full_objective|completion_rate|on_time_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|belief_health_mae|
|---|---|---|---|---|---|---|---|---|
|learned_priority|592.431|0.999463|0.605537|0|403.233|144.667|0.887937|0.0235091|
|retuned_setup_spt|597.897|0.999556|0.587556|0|238.1|194.1|0.88024|0.0297059|
|spt|665.974|0.999333|0.596333|0|414|150.8|0.878708|0.0231497|



## Learned minus reference cost, paired95% intervals



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|learned_priority|spt|-73.543|-100.594|-46.5538|3|10|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|retuned_setup_spt|-5.4661|-25.9038|20.6368|3|10|hierarchical training-seed and paired-scenario bootstrap|