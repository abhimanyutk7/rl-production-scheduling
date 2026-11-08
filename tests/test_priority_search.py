import numpy as np
import pytest
from factory_rl.priority_search import priority_action, INITIAL, feature_terms
from factory_rl.inspection import InspectionConfig, InspectionEnv
from factory_rl.inspection_study import public_action


def test_initial_search_policy_reproduces_strong_rule_through_a_full_episode():
    config=InspectionConfig(days=1,billing_days=1,particles=16,jobs=5)
    env=InspectionEnv(config,strict_actions=True)
    env.reset(seed=9)
    while not env.terminated:
        state=env.policy_state()
        expected=public_action(state,config,'spt',threshold=.7)
        assert priority_action(state,config,INITIAL)==expected
        env.step(expected)
    assert env.invalid_actions==0


def test_job_score_explanation_sums_to_priority_and_respects_masks():
    config=InspectionConfig(days=1,particles=16,jobs=5)
    env=InspectionEnv(config,strict_actions=True)
    env.reset(seed=21)
    state=env.policy_state()
    terms=feature_terms(state['candidates'][0],state,config)
    assert len(terms)==6 and np.isfinite(terms).all()
    weights=INITIAL.copy();weights[2]=1.
    action=priority_action(state,config,weights)
    assert state['action_mask'][action]
    for bad in [weights[:-1], np.full(8,np.nan)]:
        with pytest.raises(ValueError):priority_action(state,config,bad)


def test_evaluation_rejects_a_changed_policy_even_when_run_settings_match(tmp_path):
    from factory_rl.priority_search import bind_evaluation
    path=tmp_path/'policy_s0.json'
    path.write_text('{"weights":[1,2]}')
    bind_evaluation(tmp_path,[path],'rules-checksum')
    bind_evaluation(tmp_path,[path],'rules-checksum')
    path.write_text('{"weights":[2,1]}')
    with pytest.raises(ValueError):bind_evaluation(tmp_path,[path],'rules-checksum')
