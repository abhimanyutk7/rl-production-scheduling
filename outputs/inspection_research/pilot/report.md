# Adaptive inspection research results

Evidence stage: **pilot**.

3 simulated days; billing period 2 days; 5 training seeds; 50 held-out scenario sequences.

All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. The combined objective is normalised and is not an electricity bill or accuracy percentage.

|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|
|---|---:|---:|---:|---:|---:|---:|
|adaptive|102.869|53.4%|0.00|0.00|3.62|85.6%|
|belief|11.282|99.1%|0.00|3.52|4.18|88.6%|
|exact_health|51.664|78.4%|0.00|8.81|4.04|87.7%|
|no_inspection|26.001|92.8%|0.00|0.12|5.87|87.4%|
|periodic|13.151|99.1%|18.00|3.70|4.10|83.0%|
|reactive|12.356|98.6%|0.00|0.00|6.24|87.6%|
|spt|11.080|99.4%|0.00|3.46|4.12|88.2%|
|uncertainty|11.601|98.9%|3.86|3.62|4.12|85.0%|

## Interpretation and limitations

Adaptive inspection is a hypothesis, not an assumed improvement. Read cost together with completion, breakdowns and inspection spending. The exact-health policy has privileged information and is not an optimal bound.
The paired-comparison file includes uncertainty over both training seeds and matched scenarios. Small seed/scenario counts and short training are preliminary evidence.
Condition intervals are approximate particle-filter intervals; empirical coverage must be assessed. Poor coverage is an inference limitation, not proof that the policy knows the true health.
This is a finite-horizon study. Long-run factory performance and real-world savings are not established.
Candidate contribution: the interaction of paid condition information, job dispatch and persistent billing history. Individual inspection, maintenance, PPO and electricity-scheduling methods already exist.

## Research foundations

- [Inspection and maintenance under erroneous observations, 2024](https://link.springer.com/article/10.1186/s43065-024-00098-9)
- [Maintenance and value of information, 2025](https://doi.org/10.1016/j.array.2025.100454)
- [Energy-aware scheduling with preventive maintenance, 2026](https://www.sciencedirect.com/science/article/pii/S0957417425046391)
- [Tariff-aware manufacturing load shifting, 2026](https://www.mdpi.com/1424-8220/26/6/1858)

## Files

See summary.csv, episodes.csv, paired_comparisons.csv, selected_rules.json and run.json.
