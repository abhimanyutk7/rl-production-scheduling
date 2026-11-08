from dataclasses import replace
import importlib.util
import numpy as np
import pytest


def test_inspection_components_exist():
    assert importlib.util.find_spec('factory_rl.belief') is not None, 'condition belief is missing'
    assert importlib.util.find_spec('factory_rl.inspection') is not None, 'inspection environment is missing'


def tiny(**kwargs):
    from factory_rl.inspection import InspectionConfig
    settings = dict(machines=2, machine_slots=2, jobs=3, max_operations=1,
                    shift_minutes=60, arrival_minutes=10, start_hour=14,
                    days=2, billing_days=1, particles=64,
                    failure_base=0.0, failure_health=0.0)
    settings.update(kwargs)
    return InspectionConfig(**settings)


def test_config_rejects_invalid_information_and_calendar():
    for override in ({'days': 0}, {'inspection_noise': -1.0}, {'particles': 1},
                     {'start_hour': 23, 'shift_minutes': 120}, {'inspection_minutes': 0}):
        with pytest.raises(ValueError):
            tiny(**override)


def test_belief_inspection_reduces_uncertainty_without_restoring_health():
    from factory_rl.belief import HealthBelief
    belief = HealthBelief(tiny(particles=512), np.random.default_rng(2))
    before = belief.summary()
    belief.observe(0.7, 0.01)
    after = belief.summary()
    assert abs(after['mean'] - 0.7) < 0.025
    assert after['std'] < before['std'] / 2
    belief.restore(0.8)
    assert belief.summary()['mean'] == pytest.approx(0.8)
    assert belief.summary()['std'] < 1e-12


def test_belief_no_information_and_censored_observations_are_stable():
    from factory_rl.belief import HealthBelief
    belief = HealthBelief(tiny(), np.random.default_rng(3))
    before = belief.summary()
    belief.observe(0.0, float('inf'))
    assert belief.summary() == before
    for value in (0.0, 1.0, 0.5):
        belief.observe(value, 0.03)
        assert np.isfinite(list(belief.summary().values())).all()
        assert belief.weights.sum() == pytest.approx(1)
    belief.observe(0.61, 0.0)
    assert belief.summary()['mean'] == pytest.approx(0.61)


def test_survival_evidence_and_separate_filter_randomness():
    from factory_rl.belief import HealthBelief
    c = tiny(particles=512)
    first = HealthBelief(c, np.random.default_rng(9))
    second = HealthBelief(c, np.random.default_rng(9))
    before = first.summary()['mean']
    first.predict(0.0, failure=False, base=0.0, slope=0.8)
    assert first.summary()['mean'] > before
    second.predict(0.0, failure=False, base=0.0, slope=0.8)
    assert first.summary() == second.summary()


def environment(**kwargs):
    from factory_rl.inspection import InspectionEnv
    env = InspectionEnv(tiny(**kwargs))
    env.reset(seed=42)
    return env


def test_environment_exists():
    from factory_rl import inspection
    assert hasattr(inspection, 'InspectionEnv'), 'persistent environment is missing'


def test_unmasked_external_actions_are_counted_and_safely_deferred():
    env = environment()
    env.step(env.config.shortlist - 1)  # Empty shortlist slot.
    assert env.invalid_actions == 1


def test_inspection_takes_time_charges_cost_and_does_not_repair():
    env = environment(machines=1, machine_slots=1, jobs=1, days=1,
                      shift_minutes=20, inspection_minutes=5, wear_strength=0.0)
    before = env.machines[0].health
    obs, reward, done, truncated, info = env.step(env.inspection_action)
    assert env.time >= 5
    assert env.machines[0].health == before
    assert env.inspection_count == 1
    assert env.inspection_downtime == 5
    assert env.inspection_direct_cost == 100
    assert env.beliefs[0].summary()['std'] < 0.06
    assert reward < 0


def test_hidden_truth_cannot_change_observation_or_mask():
    import copy
    env = environment()
    other = copy.deepcopy(env)
    other.machines[0].health = 0.01
    other.true_rates[0] = 5.0
    np.testing.assert_array_equal(env._observation(), other._observation())
    np.testing.assert_array_equal(env.action_masks(), other.action_masks())
    assert 'true_health' not in env._info()


def test_wait_progress_and_services_without_ready_jobs():
    env = environment(jobs=1)
    for job in env.jobs:
        job.arrival = 10000
    env.arrivals.fill(10000)
    assert not env.candidates()
    assert env.action_masks()[env.inspection_action]
    assert env.action_masks()[env.maintenance_action]
    for _ in range(5):
        env.step(env.wait_action)
    assert env.time > 0


def test_shift_closure_preserves_service_and_health():
    env = environment(machines=1, machine_slots=1, jobs=1, shift_minutes=10,
                      inspection_minutes=15, days=2, billing_days=30)
    before = env.machines[0].health
    env.step(env.inspection_action)
    assert env.time >= 1445
    assert env.inspection_downtime == 15
    assert env.machines[0].health == before
    assert env.inspection_count == 1
    assert not env.terminated


def test_billing_reset_cannot_refund_liability_and_uses_calendar_midnight():
    env = environment(billing_days=1)
    env._account_power(60.0)
    assert env.demand_liability > 0
    charged = env.demand_liability
    env._advance_closed(1440)
    assert env.period == 1
    assert env.peak_kw == 0
    assert env.demand_liability == charged
    env._account_power(30.0)
    assert env.demand_liability > charged


def test_reward_telescope_terminal_and_schema():
    env = environment(days=2, shift_minutes=30)
    total = 0.0
    for _ in range(10000):
        obs, reward, done, truncated, info = env.step(env.wait_action)
        total += reward
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        if done or truncated:
            break
    else:
        pytest.fail('waiting did not reach finite horizon')
    assert done and not truncated
    assert total == pytest.approx(-env.cost_components()['full'])
    assert len(env.observation_schema) == len(obs)
    assert env.metrics()['unfinished_jobs'] == len(env.jobs)


def test_paired_physical_shocks_do_not_depend_on_inspection_count():
    first, second = environment(), environment()
    first.step(first.inspection_action)
    second.step(second.wait_action)
    np.testing.assert_array_equal(first.failure_shocks, second.failure_shocks)
    np.testing.assert_array_equal(first.wear_shocks, second.wear_shocks)


def test_particle_coverage_measurements_are_available_to_evaluator_only():
    env = environment()
    assert 'belief_health_mae' in env.metrics()
    assert 'belief_health_mae' not in env.policy_state()


def test_study_module_exists():
    assert importlib.util.find_spec('factory_rl.inspection_study') is not None, 'experiment runner missing'


def test_public_controllers_respect_masks_and_never_need_truth():
    from factory_rl.inspection_study import public_action
    env = environment()
    for kind in ('reactive', 'belief', 'periodic', 'uncertainty', 'spt'):
        state = env.policy_state()
        action = public_action(state, env.config, kind)
        assert state['action_mask'][action]
    state = env.policy_state()
    state['action_mask'][env.inspection_action] = False
    assert public_action(state, env.config, 'periodic') != env.inspection_action


def test_branch_comparison_is_reproducible_and_does_not_mutate_parent():
    from factory_rl.inspection_study import compare_actions
    env = environment(machines=1, machine_slots=1, days=1, shift_minutes=30, jobs=1)
    before = env._observation().copy()
    first = compare_actions(env, actions=[env.wait_action, env.inspection_action], worlds=3, horizon=15, seed=8)
    second = compare_actions(env, actions=[env.inspection_action, env.wait_action], worlds=3, horizon=15, seed=8)
    np.testing.assert_array_equal(before, env._observation())
    assert first.set_index('action')['mean_incremental_cost'].to_dict() == second.set_index('action')['mean_incremental_cost'].to_dict()
    assert set(first['information_mode']) == {'belief_model'}


def test_paired_statistics_include_seed_variability():
    import pandas as pd
    from factory_rl.inspection_study import paired_statistics
    rows = [dict(policy=policy, training_seed=s, scenario_seed=j, full_objective=float(j+s+delta))
            for s in (0,1) for j in (20000,20001) for policy,delta in (('reference',0),('candidate',-2))]
    result = paired_statistics(pd.DataFrame(rows), 'candidate', 'reference', draws=100)
    assert result['mean_difference'] == -2
    assert result['ci_low'] == -2 and result['ci_high'] == -2
    assert result['training_seeds'] == 2


def test_schema_mismatch_rejected_before_loading(tmp_path):
    import json
    from factory_rl.inspection_study import load_model
    (tmp_path/'metadata.json').write_text(json.dumps({'schema_version':'wrong'}))
    with pytest.raises(ValueError, match='schema'):
        load_model(tmp_path, environment())


def test_resumable_evaluation_job_reuses_matching_rows(tmp_path):
    from dataclasses import asdict
    from factory_rl.inspection_study import evaluation_job
    config = tiny(days=1, shift_minutes=20, jobs=1)
    job = dict(config=asdict(config),directory=str(tmp_path),label='reactive',
               parameters={},condition=None,seed=-1,steps=1,episodes=2)
    first = evaluation_job(job)
    second = evaluation_job(job)
    assert first == second
    assert len(first) == 2
    assert all(row['invalid_actions']==0 for row in first)


def test_branch_limit_stops_closed_time_without_false_terminal():
    env = environment(machines=1, machine_slots=1, days=2, shift_minutes=10, jobs=1)
    env.rollout_limit = 30
    env.step(env.maintenance_action)
    assert env.time == 30
    assert not env.terminated


def test_deployable_comparison_does_not_read_unreleased_orders():
    from factory_rl.inspection_study import compare_actions
    env = environment(machines=1,machine_slots=1,jobs=2,days=1,shift_minutes=60)
    other = env.snapshot()
    for j in np.flatnonzero(other.arrivals > other.time):
        other.jobs[j].operations = [{0:999}]
        other.jobs[j].due += 1000
        other.dues[j] += 1000
    np.testing.assert_array_equal(env._observation(),other._observation())
    a = compare_actions(env,[0],worlds=2,horizon=50,seed=12)
    b = compare_actions(other,[0],worlds=2,horizon=50,seed=12)
    assert a.mean_incremental_cost.iloc[0] == b.mean_incremental_cost.iloc[0]


def test_missing_learned_reference_seeds_are_not_averaged_away():
    import pandas as pd
    from factory_rl.inspection_study import paired_statistics
    frame=pd.DataFrame([dict(policy=p,training_seed=s,scenario_seed=20000,full_objective=1.)
                        for p,s in [('candidate',0),('candidate',1),('reference',0)]])
    with pytest.raises(ValueError,match='seed'):
        paired_statistics(frame,'candidate','reference')


def test_daily_and_billing_ledgers_survive_evaluation():
    env=environment(days=2,billing_days=1,jobs=1,shift_minutes=20)
    while not env.terminated:
        env.step(env.wait_action)
    metrics=env.metrics()
    assert len(metrics['ledger_daily'])==2
    assert len(metrics['ledger_billing'])==2
    assert sum(p['demand_cost'] for p in metrics['ledger_billing']) == pytest.approx(env.demand_liability)
    assert metrics['ledger_daily'][-1]['full'] == pytest.approx(env.cost_components()['full'])
    assert metrics['ledger_daily'][-1]['terminal'] == env.cost_components()['terminal']


def test_partial_window_calendar_rollover_has_expected_numerical_bill():
    env=environment(machines=1,machine_slots=1,jobs=1,days=2,billing_days=1,
                    start_hour=23,shift_minutes=10,demand_window=15)
    for _ in range(10):
        env._account_power(60.)
        env.time+=1
    assert env.energy_kwh == pytest.approx(10.)
    assert env.energy_cost == pytest.approx(40.)
    assert env.peak_kw == pytest.approx(40.)
    assert env.demand_liability == pytest.approx(4000.)
    env._advance_closed(1440)
    env._account_power(30.)
    assert env.energy_cost == pytest.approx(42.)
    assert env.demand_liability == pytest.approx(4200.)


def test_rule_imitation_can_initialise_legal_production_decisions():
    from factory_rl.inspection_study import warm_start_model
    from sb3_contrib import MaskablePPO
    from factory_rl.inspection import InspectionEnv
    from factory_rl.agents import ScenarioPool
    config=tiny(days=1,jobs=2,shift_minutes=30)
    wrapper=ScenarioPool(InspectionEnv(config),[0,1],7)
    model=MaskablePPO('MlpPolicy',wrapper,n_steps=8,batch_size=8,seed=7,device='cpu')
    result=warm_start_model(model,config,seed=7,episodes=3,epochs=25)
    assert result['examples']>0
    assert result['training_agreement']>0.75


def test_run_identity_rejects_changed_training_budget_before_mutation(tmp_path):
    import json
    from factory_rl.inspection_study import run_study,source_fingerprint,_config_fingerprint
    config=tiny()
    old=dict(config_hash=_config_fingerprint(config),requested_steps=1,
             training_seeds=[0],source_hash=source_fingerprint())
    (tmp_path/'run.json').write_text(json.dumps(old))
    with pytest.raises(ValueError,match='identity'):
        run_study(config,tmp_path,steps=2,seeds=1,episodes=1)
    assert json.loads((tmp_path/'run.json').read_text())['requested_steps']==1


def test_declared_evaluation_grid_rejects_missing_rows():
    import pandas as pd
    from factory_rl.inspection_study import validate_grid
    with pytest.raises(ValueError,match='grid'):
        validate_grid(pd.DataFrame([dict(policy='adaptive',training_seed=0,scenario_seed=20000)]),1,1)


def test_nested_model_interval_preserves_constant_paired_effect():
    from factory_rl.inspection_analysis import nested_interval
    result=nested_interval(np.full((5,16),-.2),draws=100)
    assert result['mean']==pytest.approx(-.2)
    assert result['low']==pytest.approx(-.2) and result['high']==pytest.approx(-.2)


def test_information_value_has_positive_negative_and_billing_dependent_regimes():
    from factory_rl.inspection_analysis import information_value
    costs=np.array([[8.,1.],[0.,1.]])  # Worn/healthy: process versus maintain.
    assert information_value([.5,.5],costs,np.eye(2),.2)==pytest.approx(.3)
    assert information_value([.5,.5],costs,np.full((2,2),.5),.2)==pytest.approx(-.2)
    charged=costs.copy(); charged[:,0]+=1.2
    assert information_value([.5,.5],charged,np.eye(2),.2)==pytest.approx(-.2)
