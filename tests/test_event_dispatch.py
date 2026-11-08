import numpy as np
import pytest
from factory_rl.inspection import InspectionConfig
from factory_rl.residual_dispatch import DispatchEnv, PRIOR
from factory_rl.priority_search import priority_action
from factory_rl.event_dispatch import DecisionEnv


@pytest.mark.parametrize('last_job', [False, True])
def test_aggregation_preserves_actions_physics_and_all_costs(last_job):
    cfg = InspectionConfig(days=2, billing_days=1, machines=2, jobs=12, arrival_minutes=30, particles=16)
    plain, grouped = DispatchEnv(cfg), DecisionEnv(cfg)
    a, _ = plain.reset(seed=17)
    b, _ = grouped.reset(seed=17)
    assert plain.unwrapped.time == grouped.unwrapped.time == 0
    np.testing.assert_array_equal(a, b)
    total_plain = total_grouped = 0.
    outer_steps = physical_steps = 0
    while not plain.unwrapped.terminated:
        mask = plain.action_masks()
        action = int(np.flatnonzero(mask)[-1]) if last_job else priority_action(plain.unwrapped.policy_state(), cfg, PRIOR)
        b, r, terminal, _, info = grouped.step(action)
        total_grouped += r
        physical_actions = [action]
        a, r, _, _, _ = plain.step(action)
        total_plain += r
        while not plain.unwrapped.terminated and plain.action_masks().sum() == 1:
            forced = int(np.flatnonzero(plain.action_masks())[0])
            physical_actions.append(forced)
            a, r, _, _, _ = plain.step(forced)
            total_plain += r
        assert info['aggregated_physical_actions'] == physical_actions
        physical_steps += len(physical_actions)
        outer_steps += 1
        assert terminal == plain.unwrapped.terminated
        np.testing.assert_array_equal(a, b)
        assert plain.unwrapped.metrics() == grouped.unwrapped.metrics()
    assert np.isclose(total_plain, total_grouped)
    assert grouped.physical_steps == physical_steps > outer_steps
    assert grouped.choice_steps > 0
    assert grouped.episodes == 1
