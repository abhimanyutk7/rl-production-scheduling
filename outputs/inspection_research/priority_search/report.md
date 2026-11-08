# Interpretable scheduling policy search



Exploratory three-day accelerated-billing simulation; synthetic data. Coefficients learned from whole-episode cost feedback using Gaussian cross-entropy search. This is direct black-box policy optimisation, not neural PPO or a newly invented RL algorithm.

Training scenario IDs are below 1000. Validation starts 10100; untouched test starts 22300. Each training population shares the same sampled scenario IDs. Validation can retain the original SPT + belief-maintenance rule.

The no-inspection comparison disables inspections in the learned policy without retraining. It is a fixed-parameter intervention, not a separately optimised no-inspection method. Scores and inspection thresholds are heuristics, not calculated value-of-information guarantees.

This adds no real-factory validation and no claim of world-first novelty. Read costs with completion, paired uncertainty, seed counts and actual inspection use.



|policy|full_objective|completion_rate|inspection_count|maintenance_count|breakdowns|belief_coverage90|
|---|---|---|---|---|---|---|
|belief|41.2315|0.981556|0|20.98|6.72|0.896167|
|learned_no_inspection|35.6562|0.990667|0|20.38|7.02667|0.900417|
|learned_priority|35.6562|0.990667|0|20.38|7.02667|0.900417|
|periodic|47.2322|0.978444|29.96|11.52|9.86|0.822167|
|reactive|50.0269|0.968|0|0|16.36|0.901667|
|spt|38.1834|0.988889|0|21|6.66|0.90225|
|uncertainty|43.8491|0.978222|11.46|11.58|9.72|0.802042|



## Paired cost differences; negative favours learned priority



|candidate|reference|mean_difference|ci_low|ci_high|training_seeds|scenarios|uncertainty|
|---|---|---|---|---|---|---|---|
|learned_priority|spt|-2.52724|-4.26505|-0.749415|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|belief|-5.57529|-7.98298|-3.30697|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|uncertainty|-8.19295|-10.1073|-6.19122|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|reactive|-14.3707|-16.9617|-11.8042|3|50|hierarchical training-seed and paired-scenario bootstrap|
|learned_priority|learned_no_inspection|0|0|0|3|50|hierarchical training-seed and paired-scenario bootstrap|



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
    "source_hash": "37710e21ca848499aeda0853378f6257a3e0c3ca293ac82266b20aac10427507",
    "rules_sha256": "a3f51abf3a817458af73ec677ecf3a42c3ab4f931acefecaf511311f5088bf4c",
    "seeds": 3,
    "iterations": 8,
    "population": 20,
    "training_scenarios_per_iteration": 4,
    "episodes": 50,
    "validation": 5,
    "test_start": 22300,
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
    0.34059910889729894,
    1.3536549273195544,
    0.1231717946506255,
    -0.11671176354702535,
    0.27461715626061534,
    0.262409821736285,
    0.675096464356626,
    0.9658991024702387
  ],
  "initial_validation_cost": 37.73590533333329,
  "selected_validation_cost": 33.28168377777778,
  "initial_retained": false,
  "training_seconds": 95.95583029999398,
  "selection": "Baseline and one training-selected nominee per iteration; validation only"
}

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
    "source_hash": "37710e21ca848499aeda0853378f6257a3e0c3ca293ac82266b20aac10427507",
    "rules_sha256": "a3f51abf3a817458af73ec677ecf3a42c3ab4f931acefecaf511311f5088bf4c",
    "seeds": 3,
    "iterations": 8,
    "population": 20,
    "training_scenarios_per_iteration": 4,
    "episodes": 50,
    "validation": 5,
    "test_start": 22300,
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
  "seed": 1,
  "weights": [
    0.3726491288929786,
    0.9872141684071128,
    0.22276822745446676,
    -0.21784913503887732,
    0.10775561096750011,
    0.0,
    0.6474658516105329,
    0.9775176729708205
  ],
  "initial_validation_cost": 37.73590533333329,
  "selected_validation_cost": 33.13289999999995,
  "initial_retained": false,
  "training_seconds": 96.63375409995206,
  "selection": "Baseline and one training-selected nominee per iteration; validation only"
}

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
    "source_hash": "37710e21ca848499aeda0853378f6257a3e0c3ca293ac82266b20aac10427507",
    "rules_sha256": "a3f51abf3a817458af73ec677ecf3a42c3ab4f931acefecaf511311f5088bf4c",
    "seeds": 3,
    "iterations": 8,
    "population": 20,
    "training_scenarios_per_iteration": 4,
    "episodes": 50,
    "validation": 5,
    "test_start": 22300,
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
  "seed": 2,
  "weights": [
    0.6846293479926122,
    0.7444299644407886,
    0.0,
    -0.14565401156964927,
    -0.008074847006058095,
    0.43466423140350546,
    0.75686270278125,
    0.8805026668713878
  ],
  "initial_validation_cost": 37.73590533333329,
  "selected_validation_cost": 31.97313822222219,
  "initial_retained": false,
  "training_seconds": 96.01525299996138,
  "selection": "Baseline and one training-selected nominee per iteration; validation only"
}