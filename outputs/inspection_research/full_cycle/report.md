# Adaptive inspection research results

Evidence stage: **research**.

60 simulated days; billing period 30 days; 5 training seeds; 500 held-out scenario sequences.

All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. The combined objective is normalised and is not an electricity bill or accuracy percentage.

|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|
|---|---:|---:|---:|---:|---:|---:|
|adaptive|965.963|99.8%|182.13|315.95|235.83|87.2%|
|belief|753.584|99.9%|0.00|416.12|146.60|90.0%|
|exact_health|849.419|99.9%|0.20|255.47|216.49|89.8%|
|imitation|991.631|99.8%|252.51|211.99|257.25|86.6%|
|no_inspection|846.824|99.9%|0.00|253.63|217.45|90.0%|
|periodic|922.094|99.9%|599.66|247.65|205.09|87.2%|
|reactive|1118.387|99.8%|0.00|0.00|365.39|90.2%|
|spt|652.847|99.9%|0.00|408.12|141.06|89.9%|
|uncertainty|793.175|99.9%|13.70|245.24|205.39|87.6%|

## Interpretation and limitations

Adaptive inspection is a hypothesis, not an assumed improvement. Read cost together with completion, breakdowns and inspection spending. The exact-health policy has privileged information and is not an optimal bound.
The paired-comparison file includes uncertainty over both training seeds and matched scenarios. Small seed/scenario counts and short training are preliminary evidence.
Condition intervals are approximate particle-filter intervals; empirical coverage must be assessed. Poor coverage is an inference limitation, not proof that the policy knows the true health.
This is a finite-horizon study. Long-run factory performance and real-world savings are not established.
Candidate contribution: the interaction of paid condition information, job dispatch and persistent billing history. Individual inspection, maintenance, PPO and electricity-scheduling methods already exist.

## Neural training and checkpoint selection

The current neural protocol uses a public-rule imitation warm start from training scenarios only, with inverse-square-root action-frequency weighting. PPO then trains on the same realised-cost objective. The imitation-only policy is evaluated separately. Validation may select the checkpoint before any PPO updates; a selected step of zero means imitation was retained, not that PPO improved it.

|Condition / seed|Selected PPO step|Teacher|Imitation training agreement|
|---|---:|---|---:|
|adaptive_s0|0|uncertainty_0.05_threshold_0.55|95.6%|
|adaptive_s1|0|uncertainty_0.05_threshold_0.55|95.3%|
|adaptive_s2|0|uncertainty_0.05_threshold_0.55|96.1%|
|adaptive_s3|8192|uncertainty_0.05_threshold_0.55|95.4%|
|adaptive_s4|0|uncertainty_0.05_threshold_0.55|96.0%|
|exact_health_s0|0|belief_threshold_0.55|95.9%|
|exact_health_s1|8192|belief_threshold_0.55|95.9%|
|exact_health_s2|0|belief_threshold_0.55|96.7%|
|exact_health_s3|8192|belief_threshold_0.55|96.1%|
|exact_health_s4|0|belief_threshold_0.55|96.2%|
|no_inspection_s0|0|belief_threshold_0.55|96.3%|
|no_inspection_s1|0|belief_threshold_0.55|96.9%|
|no_inspection_s2|0|belief_threshold_0.55|96.6%|
|no_inspection_s3|0|belief_threshold_0.55|96.4%|
|no_inspection_s4|32768|belief_threshold_0.55|96.7%|

## Paired cost differences

Negative values favour the candidate. These are uncertainty intervals, not a guarantee of future performance.

|Candidate minus reference|Mean difference|95% interval|
|---|---:|---|
|adaptive minus reactive|-152.424|[-188.549, -109.396]|
|adaptive minus belief|212.379|[177.012, 254.820]|
|adaptive minus periodic|43.869|[8.203, 85.684]|
|adaptive minus uncertainty|172.788|[137.684, 215.536]|
|adaptive minus no_inspection|119.139|[86.163, 155.472]|
|adaptive minus imitation|-25.667|[-78.741, 0.000]|

## Research foundations

- [Inspection and maintenance under erroneous observations, 2024](https://link.springer.com/article/10.1186/s43065-024-00098-9)
- [Maintenance and value of information, 2025](https://doi.org/10.1016/j.array.2025.100454)
- [Energy-aware scheduling with preventive maintenance, 2026](https://www.sciencedirect.com/science/article/pii/S0957417425046391)
- [Tariff-aware manufacturing load shifting, 2026](https://www.mdpi.com/1424-8220/26/6/1858)

## Files

See summary.csv, episodes.csv, paired_comparisons.csv, selected_rules.json and run.json.
