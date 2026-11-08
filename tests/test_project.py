from dataclasses import replace
import json
import numpy as np
import pytest
from factory_rl import FactoryEnv, ShopConfig
from factory_rl.baselines import dispatch, CPSATPolicy
from factory_rl.instances import taillard_matrix, import_benchmark
from factory_rl.agents import TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS, MaskedDQN, ScenarioPool, EpisodeLog
from factory_rl.evaluate import rollout


def small(**kwargs):
    settings = dict(machines=2, jobs=4, max_operations=2, shift_minutes=120, arrival_minutes=20,
                    initial_peak_kw=0.0, random_initial_peak=False, failure_base=0.0, failure_health=0.0)
    settings.update(kwargs)
    return ShopConfig(**settings)


def fixed_instance(tmp_path, jobs):
    path = tmp_path / "instance.json"
    path.write_text(json.dumps({"jobs": jobs}))
    return str(path)


def test_taillard_matches_published_verification():
    times, routes = taillard_matrix(4, 4, 1166510396, 164000672)
    assert times == [[54, 34, 61, 2], [9, 15, 89, 70], [38, 19, 28, 87], [95, 34, 7, 29]]
    assert routes == [[2, 0, 3, 1], [3, 0, 1, 2], [0, 1, 2, 3], [0, 2, 1, 3]]


def test_seed_splits_are_disjoint():
    assert len(TRAIN_SEEDS) == 1000 and len(VALIDATION_SEEDS) == len(TEST_SEEDS) == 500
    assert not set(TRAIN_SEEDS) & set(VALIDATION_SEEDS) and not set(VALIDATION_SEEDS) & set(TEST_SEEDS)


def test_reproducibility_and_valid_actions():
    a, b = FactoryEnv(small(), strict_actions=True), FactoryEnv(small(), strict_actions=True)
    oa, _ = a.reset(seed=8)
    ob, _ = b.reset(seed=8)
    assert np.array_equal(oa, ob)
    while not (a.terminated or a.truncated):
        action = dispatch(a, "health")
        assert a.action_masks()[action]
        xa, ra, ta, ua, _ = a.step(action)
        xb, rb, tb, ub, _ = b.step(action)
        assert np.array_equal(xa, xb) and ra == rb and (ta, ua) == (tb, ub)
    assert a.metrics() == b.metrics()


def test_shortlist_shape_transfer_and_health_hidden():
    a, b = FactoryEnv(small()), FactoryEnv(replace(small(), machines=10, jobs=60))
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    assert oa.shape == ob.shape == a.observation_space.shape
    assert a.action_space.n == b.action_space.n == 11
    a.machines[0].health = 0.3
    assert a.action_masks()[a.maintenance_action]
    hidden = FactoryEnv(replace(small(), health_visible=False))
    hidden.reset(seed=0)
    before = hidden._observation()
    for machine in hidden.machines:
        machine.health = 0.2
    assert np.array_equal(before, hidden._observation())


def test_invalid_action_is_rejected_and_state_not_mutated():
    env = FactoryEnv(small(), strict_actions=True)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(env.maintenance_action)
    assert env.time == 0 and not env.active.any()
    with pytest.raises(ValueError):
        env.step(0.5)


def test_demand_window_and_peak_reward_telescope(tmp_path):
    instance = fixed_instance(tmp_path, [{"arrival": 0, "due": 60, "product": 1, "operations": [{"0": 15}]}])
    config = small(machines=1, jobs=1, max_operations=1, shift_minutes=60, setup_enabled=False,
                   energy_weight=0, tardiness_weight=0, maintenance_weight=0, demand_rate=100, demand_scale=100)
    env = FactoryEnv(config, instance=instance, strict_actions=True)
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    assert env.energy_kwh == pytest.approx(2.5)
    assert env.peak_kw == pytest.approx(10)
    assert reward == pytest.approx(-10)
    assert env.demand_windows[0]["average_kw"] == pytest.approx(10)
    inherited = FactoryEnv(replace(config, initial_peak_kw=30), instance=instance)
    inherited.reset(seed=0)
    inherited.step(0)
    assert inherited.cost_components()["demand"] == 0
    assert inherited.metrics()["total_electricity_bill"] > inherited.metrics()["incremental_electricity_cost"]


def test_partial_demand_window_is_zero_padded(tmp_path):
    instance = fixed_instance(tmp_path, [{"arrival": 0, "due": 60, "product": 1, "operations": [{"0": 5}]}])
    env = FactoryEnv(small(machines=1, jobs=1, max_operations=1), instance=instance)
    env.reset(seed=0)
    env.step(0)
    assert env.shift_peak_kw == pytest.approx(10 * 5 / 15)


def test_precedence_no_overlap_and_exact_processing(tmp_path):
    instance = fixed_instance(tmp_path, [
        {"arrival": 0, "due": 100, "product": 1, "operations": [{"0": 5}, {"1": 7}]},
        {"arrival": 0, "due": 100, "product": 2, "operations": [{"0": 6}, {"1": 3}]}])
    metrics, env = rollout(small(jobs=2, setup_enabled=True), lambda e: dispatch(e, "edd"), 0, instance, True)
    assert metrics["completed_jobs"] == 2
    process = [r for r in env.trace if r["mode"] == "process"]
    assert len(process) == 21
    for job in range(2):
        first = [r["time"] for r in process if r["job"] == job and r["operation"] == 0]
        second = [r["time"] for r in process if r["job"] == job and r["operation"] == 1]
        assert max(first) < min(second)
    assert len({(r["time"], r["machine"]) for r in process}) == len(process)


def test_breakdown_resumes_work_without_duplicate_processing(tmp_path):
    instance = fixed_instance(tmp_path, [{"arrival": 0, "due": 100, "product": 1, "operations": [{"0": 6}]}])
    env = FactoryEnv(small(machines=1, jobs=1, max_operations=1, failure_base=0.1, repair_minutes=3), instance=instance, record=True)
    env.reset(seed=0)
    env.shocks[:] = 1
    env.shocks[1, 0] = 0
    env.step(0)
    assert env.breakdowns == 1 and env.completion[0] == 9
    assert env.productive_minutes[0] == 6 and env.repair_downtime == 3


def test_maintenance_and_health_rule(tmp_path):
    instance = fixed_instance(tmp_path, [{"arrival": 0, "due": 100, "product": 1, "operations": [{"0": 6}]}])
    env = FactoryEnv(small(machines=1, jobs=1, max_operations=1, maintenance_minutes=3), instance=instance)
    env.reset(seed=0)
    env.machines[0].health = 0.3
    action = dispatch(env, "health")
    assert action == env.maintenance_action
    env.step(action)
    assert env.machines[0].health == 1 and env.time == 3 and env.maintenance_downtime == 3


def test_shift_censoring_and_telescoping():
    env = FactoryEnv(small(shift_minutes=30, arrival_minutes=0), strict_actions=True)
    env.reset(seed=0)
    total_reward = 0
    while not (env.terminated or env.truncated):
        _, reward, _, _, _ = env.step(dispatch(env, "spt"))
        total_reward += reward
    assert env.time <= 30
    assert total_reward == pytest.approx(-env.cost_components()["full"])
    if env.truncated:
        assert env.metrics()["unfinished_jobs"] > 0


def test_ppo_shift_end_has_no_continuation_value(tmp_path, monkeypatch):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    instance = fixed_instance(tmp_path, [{"arrival": 0, "due": 100, "product": 1, "operations": [{"0": 6}]}])
    env = ScenarioPool(FactoryEnv(small(machines=1, jobs=1, max_operations=1, shift_minutes=1),
                                  instance=instance), [0], 0)
    model = MaskablePPO("MlpPolicy", env, n_steps=2, batch_size=2, n_epochs=1, gamma=1.0,
                        seed=0, device="cpu", policy_kwargs={"net_arch": [8, 8]})
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.full((obs.shape[0], 1), 100.0))
    log = EpisodeLog()
    try:
        model.learn(total_timesteps=2, callback=log)
        assert len(log.rows) == 2 and all(row["unfinished_jobs"] == 1 for row in log.rows)
        assert model.rollout_buffer.rewards[:, 0] == pytest.approx([-row["cost_full"] for row in log.rows])
    finally:
        env.close()


def test_cp_sat_known_static_optimum(tmp_path):
    jobs = [
        {"arrival": 0, "due": 100, "product": 0, "operations": [{"0": 3}, {"1": 2}, {"2": 2}]},
        {"arrival": 0, "due": 100, "product": 0, "operations": [{"0": 2}, {"2": 1}, {"1": 4}]},
        {"arrival": 0, "due": 100, "product": 0, "operations": [{"1": 4}, {"2": 3}]}]
    env = FactoryEnv(small(machines=3, jobs=3, max_operations=3, setup_enabled=False), instance=fixed_instance(tmp_path, jobs))
    env.reset(seed=0)
    policy = CPSATPolicy(False, 2)
    policy._solve(env, False)
    assert policy.statuses[-1] == "OPTIMAL"
    makespan = max(start + jobs[j]["operations"][k][str(m)] for (j, k, m), start in policy.plan.items())
    assert makespan == 11


def test_masked_dqn_never_chooses_masked_action(tmp_path):
    import torch
    model = MaskedDQN(np.int64(3), np.int64(4))
    with torch.no_grad():
        for param in model.net.parameters():
            param.zero_()
        model.net[-1].bias[0] = 1000
    mask = np.array([False, True, False, False])
    assert model.predict(np.zeros(3), mask) == 1
    model.save(tmp_path / "model.pt")
    assert MaskedDQN.load(tmp_path / "model.pt").predict(np.zeros(3), mask) == 1


def test_solver_hint_is_feasible_with_setups_and_alternatives():
    env = FactoryEnv(small(jobs=8))
    env.reset(seed=0)
    solver = CPSATPolicy(False, 2)
    solver._solve(env, False)
    assert solver.statuses[-1] in ("FEASIBLE", "OPTIMAL") and solver.plan


def test_import_formats_and_reject_malformed(tmp_path):
    path = tmp_path / "small.txt"
    path.write_text("1 2\n0 3 1 4\n")
    data, config = import_benchmark(path, "jssp")
    assert config.machines == 2 and data["jobs"][0]["operations"] == [{0: 3}, {1: 4}]
    path.write_text("1 2\n2 2 1 3 2 4 1 2 5\n")
    data, _ = import_benchmark(path, "brandimarte")
    assert data["jobs"][0]["operations"] == [{0: 3, 1: 4}, {1: 5}]
    path.write_text("1 2\n2 2 1\n")
    with pytest.raises(ValueError):
        import_benchmark(path, "brandimarte")


@pytest.mark.parametrize("change", [{"machines": 11}, {"failure_base": 2.0}, {"energy_scale": 0}, {"jobs": 0}, {"start_hour": 25},
                                   {"wear_strength": -1}, {"random_initial_peak": "false"}, {"arrival_minutes": True}, {"energy_weight": "1"}])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        ShopConfig(**change)
