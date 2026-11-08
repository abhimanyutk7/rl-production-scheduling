# Adaptive inspection research results

Evidence stage: **smoke**.

2 simulated days; billing period 1 days; 1 training seeds; 2 held-out scenario sequences.

All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. The combined objective is normalised and is not an electricity bill or accuracy percentage.

|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|
|---|---:|---:|---:|---:|---:|---:|
|adaptive|31.921|94.2%|7.50|9.00|7.00|76.4%|
|belief|30.301|96.7%|0.00|4.50|8.50|87.5%|
|exact_health|29.253|96.7%|0.00|8.00|7.00|88.8%|
|imitation|31.921|94.2%|7.50|9.00|7.00|76.4%|
|no_inspection|31.551|95.8%|0.00|11.50|7.00|89.4%|
|periodic|27.080|95.8%|20.00|8.50|5.00|73.4%|
|reactive|32.541|95.0%|0.00|0.00|9.50|85.6%|
|spt|26.537|98.3%|0.00|4.00|8.00|87.0%|
|uncertainty|26.387|96.7%|6.50|7.50|5.50|85.5%|

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
|adaptive_s0|0|uncertainty_0.05_threshold_0.55|97.7%|
|exact_health_s0|0|belief_threshold_0.55|99.9%|
|no_inspection_s0|0|belief_threshold_0.55|100.0%|

## Paired cost differences

Negative values favour the candidate. These are uncertainty intervals, not a guarantee of future performance.

|Candidate minus reference|Mean difference|95% interval|
|---|---:|---|
|adaptive minus reactive|-0.620|[-3.016, 1.776]|
|adaptive minus belief|1.620|[-0.041, 3.281]|
|adaptive minus periodic|4.841|[1.045, 8.637]|
|adaptive minus uncertainty|5.533|[1.606, 9.461]|
|adaptive minus no_inspection|0.369|[-0.396, 1.135]|
|adaptive minus imitation|0.000|[0.000, 0.000]|

## Research foundations

- [Inspection and maintenance under erroneous observations, 2024](https://link.springer.com/article/10.1186/s43065-024-00098-9)
- [Maintenance and value of information, 2025](https://doi.org/10.1016/j.array.2025.100454)
- [Energy-aware scheduling with preventive maintenance, 2026](https://www.sciencedirect.com/science/article/pii/S0957417425046391)
- [Tariff-aware manufacturing load shifting, 2026](https://www.mdpi.com/1424-8220/26/6/1858)

## Files

See summary.csv, episodes.csv, paired_comparisons.csv, selected_rules.json and run.json.
