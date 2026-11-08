import numpy as np
import pytest
import torch

from factory_rl.inspection import InspectionConfig, InspectionEnv
from factory_rl.priority_search import priority_action
from factory_rl.residual_dispatch import DispatchEnv, ResidualPolicy, make_model, PRIOR


def small_env():
    return DispatchEnv(InspectionConfig(machines=2, days=2, billing_days=1, particles=16, jobs=12, arrival_minutes=20))


def test_initial_neural_dispatch_matches_the_frozen_rule_through_episodes():
    torch.set_num_threads(1)
    env = small_env()
    model = make_model(env, seed=0, device='cpu')
    seen = set()
    for seed in [12, 21]:
        obs, _ = env.reset(seed=seed)
        while not env.unwrapped.terminated:
            expected = priority_action(env.unwrapped.policy_state(), env.unwrapped.config, PRIOR)
            mask = env.action_masks()
            action = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
            assert action == expected and env.unwrapped.action_masks()[action]
            seen.add(action)
            assert np.isfinite(obs).all() and (np.abs(obs) <= 20).all()
            obs, _, _, _, _ = env.step(action)
    assert env.unwrapped.invalid_actions == 0
    assert env.unwrapped.maintenance_action in seen and env.unwrapped.wait_action in seen


def test_policy_forward_log_prob_gradient_and_reload_agree(tmp_path):
    torch.set_num_threads(1)
    env = small_env()
    model = make_model(env, seed=2, device='cpu')
    obs, _ = env.reset(seed=23)
    x = torch.as_tensor(np.repeat(obs[None], 3, axis=0))
    masks = np.repeat(env.action_masks()[None], 3, axis=0)
    policy = model.policy
    actions, values, log_prob = policy(x, deterministic=True, action_masks=masks)
    distribution = policy.get_distribution(x, action_masks=masks)
    evaluated, evaluated_prob, entropy = policy.evaluate_actions(x, actions, action_masks=masks)
    torch.testing.assert_close(log_prob, distribution.log_prob(actions))
    torch.testing.assert_close(log_prob, evaluated_prob)
    torch.testing.assert_close(values, evaluated)
    assert torch.isfinite(entropy).all()
    scorer = policy.mlp_extractor.job_net
    ids = {id(p) for group in policy.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in scorer.parameters())
    # A nontrivial legal dispatch state gives gradients to the residual head.
    k = env.unwrapped.config.shortlist
    while env.action_masks()[:k].sum() < 2:
        obs, _, _, _, _ = env.step(int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]))
        assert not env.unwrapped.terminated
    x = torch.as_tensor(obs[None])
    distribution = policy.get_distribution(x, action_masks=env.action_masks()[None])
    target = torch.tensor([np.flatnonzero(env.action_masks()[:k])[-1]])
    before = scorer[-1].weight.detach().clone()
    policy.optimizer.zero_grad()
    (-distribution.log_prob(target).mean()).backward()
    assert scorer[-1].weight.grad.abs().sum() > 0
    policy.optimizer.step()
    assert not torch.equal(before, scorer[-1].weight)
    assert torch.isfinite(policy.get_distribution(x, action_masks=env.action_masks()[None]).distribution.logits).all()
    model.save(tmp_path/'model.zip')
    from sb3_contrib import MaskablePPO
    loaded = MaskablePPO.load(tmp_path/'model.zip', device='cpu')
    assert isinstance(loaded.policy, ResidualPolicy)
    a = model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]
    b = loaded.predict(obs, action_masks=env.action_masks(), deterministic=True)[0]
    assert a == b


def test_shared_scoring_is_permutation_equivariant_and_can_change_the_prior():
    torch.set_num_threads(1)
    env = small_env()
    model = make_model(env, seed=1, device='cpu')
    obs, _ = env.reset(seed=8)
    k = env.unwrapped.config.shortlist
    x = torch.as_tensor(obs[None]).clone()
    # Two equal candidate features, only the prior preference differs.
    x[:, :k*8] = 0
    x[:, 7] = x[:, 15] = 1
    x[:, 6] = 1
    x[:, 0] = 1
    x[:, 8] = -1
    extractor = model.policy.mlp_extractor
    for p in extractor.job_net.parameters():
        p.data.zero_()
    extractor.job_net[0].weight.data[0, 0] = 1
    extractor.job_net[2].weight.data[0, 0] = 1
    extractor.job_net[4].weight.data[0, 0] = -10
    logits = extractor.forward_actor(x)
    assert logits[0, 1] > logits[0, 0], 'Residual must be able to overturn a prior decision'
    permuted = x.clone()
    permuted[:, :8], permuted[:, 8:16] = x[:, 8:16].clone(), x[:, :8].clone()
    changed = extractor.forward_actor(permuted)
    torch.testing.assert_close(logits[:, 0], changed[:, 1])
    torch.testing.assert_close(logits[:, 1], changed[:, 0])


def test_wrapper_uses_public_state_and_preserves_reward_and_mask():
    env = small_env()
    obs, _ = env.reset(seed=8)
    original_mask = env.unwrapped.action_masks().copy()
    gated_mask = env.action_masks().copy()
    env.unwrapped.machines[0].health = .001
    np.testing.assert_array_equal(obs, env.observation())
    np.testing.assert_array_equal(gated_mask, env.action_masks())
    assert np.all(~gated_mask | original_mask)
    with pytest.raises(ValueError):
        DispatchEnv(InspectionConfig(), prior=np.ones(7))


def test_research_cache_rejects_incompatible_or_duplicate_rows(tmp_path):
    from factory_rl.residual_dispatch import validate_rows, bind_identity
    rows = [dict(policy='residual', training_seed=0, scenario_seed=24500)]
    validate_rows(rows, 'residual', 0, {24500, 24501})
    with pytest.raises(ValueError):
        validate_rows(rows+rows, 'residual', 0, {24500, 24501})
    with pytest.raises(ValueError):
        validate_rows(rows, 'no_prior', 0, {24500, 24501})
    bind_identity(tmp_path, {'steps': 1024, 'weights': [1., 4.]})
    bind_identity(tmp_path, {'steps': 1024, 'weights': [1., 4.]})
    with pytest.raises(ValueError):
        bind_identity(tmp_path, {'steps': 2048, 'weights': [1., 4.]})


def test_evaluation_copy_after_training_preserves_rng_and_policy():
    from factory_rl.residual_dispatch import evaluation_policy
    torch.set_num_threads(1)
    env = small_env()
    model = make_model(env, seed=4, device='cpu')
    model.learn(512)
    state = torch.get_rng_state().clone()
    actor = evaluation_policy(model.policy)
    assert torch.equal(state, torch.get_rng_state())
    assert next(actor.parameters()).device.type == 'cpu'
    obs, _ = env.reset(seed=12)
    mask = env.action_masks()
    np.testing.assert_array_equal(model.predict(obs, action_masks=mask, deterministic=True)[0],
                                  actor.predict(obs, action_masks=mask, deterministic=True)[0])


def test_sixty_day_features_stay_bounded_and_scaled_rewards_reconcile():
    config = InspectionConfig(days=60, billing_days=30, particles=16, jobs=2, machines=2)
    env = DispatchEnv(config)
    obs, _ = env.reset(seed=13)
    total_reward = 0.
    while not env.unwrapped.terminated:
        assert env.observation_space.contains(obs)
        action = priority_action(env.unwrapped.policy_state(), config, PRIOR)
        obs, reward, _, _, _ = env.step(action)
        total_reward += reward
    assert np.isclose(-total_reward*config.days, env.unwrapped.metrics()['full_objective'])
    assert len(env.unwrapped.metrics()['ledger_daily']) == 60
    assert env.unwrapped.invalid_actions == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime required')
def test_cpu_rollouts_with_cuda_updates_work_across_optimizer_steps(tmp_path):
    torch.set_num_threads(1)
    env = small_env()
    model = make_model(env, seed=7, device='cuda_updates')
    model.learn(1024)
    assert model.gpu_update_calls == 2
    assert next(model.policy.parameters()).device.type == 'cuda'
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    assert all(s['exp_avg'].device.type == 'cuda' for s in model.policy.optimizer.state.values() if 'exp_avg' in s)
    model.save(tmp_path/'hybrid.zip')
    from sb3_contrib import MaskablePPO
    loaded = MaskablePPO.load(tmp_path/'hybrid.zip', device='cpu')
    obs, _ = env.reset(seed=9)
    mask = env.action_masks()
    assert int(model.predict(obs, action_masks=mask, deterministic=True)[0]) == int(loaded.predict(obs, action_masks=mask, deterministic=True)[0])

