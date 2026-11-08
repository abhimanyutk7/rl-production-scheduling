"""Checks for a separate controller-refinement protocol; frozen studies stay intact."""
import numpy as np
import pytest
from factory_rl.refinement import monte_carlo_returns, ScaledReward
from factory_rl.inspection import InspectionConfig, InspectionEnv


def test_returns_include_final_penalty_and_reset_at_episode_boundary():
    assert np.allclose(monte_carlo_returns([-1., -2., -7.]), [-10., -9., -7.])
    assert np.allclose(monte_carlo_returns([-3.]), [-3.])
    assert len(monte_carlo_returns([])) == 0
    with pytest.raises(ValueError):
        monte_carlo_returns([float('nan')])


def test_fixed_positive_reward_scaling_preserves_raw_cost_and_terminal():
    from factory_rl.inspection_study import public_action
    config = InspectionConfig(days=1, billing_days=1, particles=16, jobs=3)
    raw, scaled = InspectionEnv(config, strict_actions=True), ScaledReward(InspectionEnv(config, strict_actions=True), .25)
    obs, _ = raw.reset(seed=17)
    other, _ = scaled.reset(seed=17)
    assert np.array_equal(obs, other)
    total_raw = total_scaled = 0.
    while not raw.terminated:
        action = public_action(raw.policy_state(), config, 'spt', threshold=.7)
        a, r, done, truncated, info = raw.step(action)
        b, sr, sdone, struncated, sinfo = scaled.step(action)
        assert np.array_equal(a, b)
        assert np.array_equal(raw.action_masks(), scaled.action_masks())
        assert (done, truncated) == (sdone, struncated)
        assert sinfo['cost_components'] == info['cost_components']
        total_raw += r
        total_scaled += sr
    assert total_scaled == pytest.approx(total_raw * .25)
    assert -total_raw == pytest.approx(raw.metrics()['full_objective'])
    for bad in [0, -1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            ScaledReward(raw, bad)


def test_critic_fitting_preserves_actor_and_reduces_value_error():
    import torch
    from sb3_contrib import MaskablePPO
    from factory_rl.refinement import fit_critic
    torch.set_num_threads(1)
    env = InspectionEnv(InspectionConfig(days=1, particles=16, jobs=3))
    model = MaskablePPO('MlpPolicy', env, n_steps=16, batch_size=16, seed=91, device='cpu', policy_kwargs={'net_arch':[16,16]})
    obs, _ = env.reset(seed=18)
    x = np.stack([obs]*32)
    masks = np.stack([env.action_masks()]*32)
    with torch.no_grad():
        before = model.policy.get_distribution(torch.as_tensor(x), action_masks=masks).distribution.logits.clone()
    result = fit_critic(model, x, np.full(32, -4.,dtype=np.float32), updates=100, seed=7)
    with torch.no_grad():
        after = model.policy.get_distribution(torch.as_tensor(x), action_masks=masks).distribution.logits
    assert torch.equal(before, after)
    assert result['mse_after'] < result['mse_before'] * .1


def test_cache_rejects_swapped_controller_seed_duplicate_and_unplanned_scenario():
    from factory_rl.refinement import validate_cached_rows
    good = dict(policy='actor_critic', training_seed=0, scenario_seed=21000)
    validate_cached_rows([good], 'actor_critic', 0, {21000,21001})
    for bad in [dict(good,policy='actor_only'), dict(good,training_seed=1), dict(good,scenario_seed=20000)]:
        with pytest.raises(ValueError):
            validate_cached_rows([bad], 'actor_critic', 0, {21000,21001})
    with pytest.raises(ValueError):
        validate_cached_rows([good,good], 'actor_critic', 0, {21000,21001})
