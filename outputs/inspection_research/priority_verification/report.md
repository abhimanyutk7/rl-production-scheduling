# Interpretable scheduling policy search



Exploratory three-day accelerated-billing simulation; synthetic data. Coefficients learned from whole-episode cost feedback using Gaussian cross-entropy search. This is direct black-box policy optimisation, not neural PPO or a newly invented RL algorithm.

Training scenario IDs are below 1000. Validation starts 10100; untouched test starts 22100. Each training population shares the same sampled scenario IDs. Validation can retain the original SPT + belief-maintenance rule.

The no-inspection comparison disables inspections in the learned policy without retraining. It is a fixed-parameter intervention, not a separately optimised no-inspection method. Scores and inspection thresholds are heuristics, not calculated value-of-information guarantees.

This adds no real-factory validation and no claim of world-first novelty. Read costs with completion, paired uncertainty, seed counts and actual inspection use.



|policy|full_objective|completion_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|
|---|---|---|---|---|---|---|
|belief|33.0065|0.972222|0|21|4|0.970833|
|learned_no_inspection|29.5616|0.994444|0|20.5|3.5|0.963542|
|learned_priority|29.5616|0.994444|0|20.5|3.5|0.963542|
|periodic|34.8358|0.966667|30|12|4.5|0.891667|
|reactive|41.5286|0.955556|0|0|13.5|0.951042|
|spt|29.5616|0.994444|0|20.5|3.5|0.963542|
|uncertainty|31.6164|0.988889|12|12|6|0.894792|



## Paired cost differences; negative favours learned priority



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|learned_priority|spt|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|belief|-3.44489|-6.86045|-0.0293367|1|2|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|uncertainty|-2.05476|-4.46767|0.358146|1|2|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|reactive|-11.967|-17.0726|-6.86136|1|2|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|learned_no_inspection|0|0|0|1|2|hierarchical training-seed and paired-scenario bootstrap|



## Learned weights



{
  "identity": {
    "config": {
      "machines": 5,
      "jobs": 30,
      "max_operations": 3,
      "products": 3,
      "shortlist": 10,
      "machine_slots": 10,
      "benchmark_time_scale": 0.5,
      "shift_minutes": 480,
      "arrival_minutes": 240,
      "max_minutes": 1440,
      "start_hour": 14,
      "demand_window": 15,
      "initial_peak_kw": 0.0,
      "random_initial_peak": false,
      "demand_rate": 100.0,
      "offpeak_rate": 4.0,
      "normal_rate": 7.0,
      "peak_rate": 12.0,
      "wear_strength": 1.0,
      "wear_per_minute": 0.0015,
      "continuous_wear": 0.0002,
      "health_visible": true,
      "load_dependent_failure": true,
      "failure_base": 0.0003,
      "failure_health": 0.008,
      "maintenance_minutes": 30,
      "repair_minutes": 90,
      "maintenance_enabled": true,
      "setup_enabled": true,
      "demand_enabled": true,
      "tardiness_weight": 1.0,
      "energy_weight": 1.0,
      "demand_weight": 1.0,
      "maintenance_weight": 0.2,
      "reward_scale": 1.0,
      "tardiness_scale": 480.0,
      "energy_scale": 1000.0,
      "demand_scale": 3000.0,
      "downtime_scale": 60.0,
      "unfinished_penalty": 2.0,
      "days": 3,
      "billing_days": 2,
      "particles": 64,
      "inspection_minutes": 5,
      "inspection_noise": 0.03,
      "inspection_cost": 100.0,
      "control_minutes": 10,
      "wait_minutes": 5,
      "maintenance_cooldown": 30,
      "initial_health_low": 0.6,
      "wear_rate_sigma": 0.2,
      "wear_noise_cv": 0.35,
      "inspections_enabled": true
    },
    "source_hash": "b6c0b4c791a824130caa72ebc596a571377bd2db46b11df485071d98afd3ecf7",
    "rules_sha256": "a3f51abf3a817458af73ec677ecf3a42c3ab4f931acefecaf511311f5088bf4c",
    "seeds": 1,
    "iterations": 1,
    "population": 4,
    "training_scenarios_per_iteration": 1,
    "episodes": 2,
    "validation": 2,
    "test_start": 22100,
    "validation_start": 10100,
    "features": [
      "duration",
      "setup",
      "slack",
      "age",
      "operations_left",
      "nominal_energy"
    ],
    "initial_weights": [
      1.0,
      0.5,
      0.0,
      0.0,
      0.0,
      0.0,
      0.7,
      1.0
    ],
    "lower": [
      0.1,
      0.0,
      0.0,
      -2.0,
      -2.0,
      0.0,
      0.35,
      0.03
    ],
    "upper": [
      3.0,
      3.0,
      3.0,
      0.0,
      2.0,
      3.0,
      0.9,
      1.0
    ],
    "method": "Gaussian cross-entropy direct policy search; 25 percent elites, smoothed updates and nonzero variance floor"
  },
  "seed": 0,
  "weights": [
    1.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.0,
    0.7,
    1.0
  ],
  "initial_validation_cost": 32.892103888888855,
  "selected_validation_cost": 32.892103888888855,
  "initial_retained": true,
  "training_seconds": 4.068090899963863,
  "selection": "Baseline and one training-selected nominee per iteration; validation only"
}