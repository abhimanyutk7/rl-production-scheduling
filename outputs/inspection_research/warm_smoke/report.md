# Adaptive inspection research results

Evidence stage: **smoke**.

1 simulated days; billing period 1 days; 1 training seeds; 1 held-out scenario sequences.

All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. The combined objective is normalised and is not an electricity bill or accuracy percentage.

|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|
|---|---:|---:|---:|---:|---:|---:|
|adaptive|19.302|86.7%|0.00|0.00|7.00|80.6%|
|belief|16.900|90.0%|0.00|3.00|5.00|80.6%|
|exact_health|19.302|86.7%|0.00|0.00|7.00|80.6%|
|no_inspection|19.302|86.7%|0.00|0.00|7.00|80.6%|
|periodic|11.355|100.0%|10.00|6.00|2.00|86.9%|
|reactive|19.302|86.7%|0.00|0.00|7.00|80.6%|
|spt|17.226|90.0%|0.00|0.00|7.00|81.2%|
|uncertainty|13.365|96.7%|7.00|5.00|3.00|80.6%|

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
