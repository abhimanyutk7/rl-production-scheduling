# Adaptive inspection research results

Evidence stage: **pilot**.

3 simulated days; billing period 2 days; 5 training seeds; 50 held-out scenario sequences.

All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. The combined objective is normalised and is not an electricity bill or accuracy percentage.

|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|
|---|---:|---:|---:|---:|---:|---:|
|adaptive|47.683|96.7%|9.17|12.32|11.51|84.2%|
|belief|40.878|97.8%|0.00|21.04|6.90|87.1%|
|exact_health|45.024|96.8%|0.09|11.07|11.84|87.4%|
|imitation|46.730|96.9%|10.99|15.11|10.28|84.1%|
|no_inspection|46.278|96.8%|0.00|13.54|10.88|87.1%|
|periodic|47.445|97.0%|30.00|11.58|9.96|82.0%|
|reactive|49.826|95.7%|0.00|0.00|16.52|87.4%|
|spt|38.125|98.5%|0.00|21.02|6.90|86.8%|
|uncertainty|43.847|97.6%|11.48|11.44|9.92|80.1%|

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
|adaptive_s0|0|uncertainty_0.05_threshold_0.55|97.0%|
|adaptive_s1|0|uncertainty_0.05_threshold_0.55|96.8%|
|adaptive_s2|49152|uncertainty_0.05_threshold_0.55|97.3%|
|adaptive_s3|0|uncertainty_0.05_threshold_0.55|97.4%|
|adaptive_s4|0|uncertainty_0.05_threshold_0.55|95.8%|
|exact_health_s0|0|belief_threshold_0.55|98.7%|
|exact_health_s1|0|belief_threshold_0.55|98.5%|
|exact_health_s2|40960|belief_threshold_0.55|97.7%|
|exact_health_s3|57344|belief_threshold_0.55|98.9%|
|exact_health_s4|0|belief_threshold_0.55|98.8%|
|no_inspection_s0|8192|belief_threshold_0.55|98.5%|
|no_inspection_s1|0|belief_threshold_0.55|97.8%|
|no_inspection_s2|65536|belief_threshold_0.55|98.2%|
|no_inspection_s3|16384|belief_threshold_0.55|99.1%|
|no_inspection_s4|8192|belief_threshold_0.55|98.6%|

## Paired cost differences

Negative values favour the candidate. These are uncertainty intervals, not a guarantee of future performance.

|Candidate minus reference|Mean difference|95% interval|
|---|---:|---|
|adaptive minus reactive|-2.143|[-4.605, 0.119]|
|adaptive minus belief|6.805|[4.492, 9.220]|
|adaptive minus periodic|0.238|[-2.109, 2.661]|
|adaptive minus uncertainty|3.836|[1.676, 6.166]|
|adaptive minus no_inspection|1.405|[-0.625, 3.433]|
|adaptive minus imitation|0.953|[0.000, 3.068]|

## Research foundations

- [Inspection and maintenance under erroneous observations, 2024](https://link.springer.com/article/10.1186/s43065-024-00098-9)
- [Maintenance and value of information, 2025](https://doi.org/10.1016/j.array.2025.100454)
- [Energy-aware scheduling with preventive maintenance, 2026](https://www.sciencedirect.com/science/article/pii/S0957417425046391)
- [Tariff-aware manufacturing load shifting, 2026](https://www.mdpi.com/1424-8220/26/6/1858)

## Files

See summary.csv, episodes.csv, paired_comparisons.csv, selected_rules.json and run.json.
