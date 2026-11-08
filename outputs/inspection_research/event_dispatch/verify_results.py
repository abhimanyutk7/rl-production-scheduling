"""Audit the completed physical-budget event study without retraining anything."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))
from factory_rl.inspection_study import source_fingerprint, paired_statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads((ROOT/'run.json').read_text())
    assert manifest['status'] == 'complete', 'Wait for the frozen study to complete'
    identity = manifest['identity']
    args, config = identity['settings'], identity['config']
    fingerprint = hashlib.sha256((source_fingerprint()+(PROJECT/'factory_rl/event_dispatch.py').read_text()
        +(PROJECT/'factory_rl/residual_dispatch.py').read_text()+(PROJECT/'factory_rl/priority_search.py').read_text()).encode()).hexdigest()
    assert fingerprint == identity['source_hash']
    for path, expected in identity['controls_sha256'].items():
        assert sha(PROJECT/path) == expected
    frame = pd.read_csv(ROOT/'episodes.csv')
    jobs = [(p,s) for s in range(args['seeds']) for p in ['event_residual','event_no_prior','plain_residual','plain_no_prior']]
    jobs += [('retuned_setup_spt',-1),('spt',-1)]
    scenarios = range(args['test_start'],args['test_start']+args['episodes'])
    key = ['policy','training_seed','scenario_seed']
    actual = list(frame[key].itertuples(index=False,name=None))
    expected = {(p,s,j) for p,s in jobs for j in scenarios}
    assert len(actual) == len(set(actual)) and set(actual) == expected
    accounting = runpy.run_path(str(PROJECT/'outputs/inspection_research/residual_dispatch/verify_results.py'))['accounting']
    checks = accounting(frame,config)
    metadata = []
    for policy,seed in jobs:
        cached = pd.read_csv(ROOT/f'{policy}_s{seed}.csv')
        subset = frame[frame.policy.eq(policy)&frame.training_seed.eq(seed)]
        pd.testing.assert_frame_equal(cached.sort_values(key).reset_index(drop=True),
            subset[cached.columns].sort_values(key).reset_index(drop=True),check_like=True,check_dtype=False)
        if not policy.startswith('event_'):
            continue
        directory = ROOT/'models'/f'{policy}_s{seed}'
        meta = json.loads((directory/'metadata.json').read_text())
        best = min(json.loads((directory/'validation.json').read_text()),key=lambda r:r['mean_cost'])
        assert meta['status'] == 'complete' and meta['condition'] == policy and meta['seed'] == seed
        assert meta['config'] == config
        assert meta['selected_steps'] == best['outer_steps'] and meta['validation_cost'] == best['mean_cost']
        assert sha(directory/'model.zip') == meta['model_sha256']
        saved = json.loads(zipfile.ZipFile(directory/'model.zip').read('data'))
        assert saved['num_timesteps'] == meta['selected_steps'] and saved['gamma'] == 1
        assert meta['physical_steps'] >= meta['trained_rollout_physical_steps'] >= args['physical_steps']
        assert meta['physical_overshoot'] == meta['trained_rollout_physical_steps']-args['physical_steps']
        assert meta['choice_steps'] <= meta['outer_steps'] and meta['gpu_update_calls'] > 0 and meta['episodes'] > 0
        if policy == 'event_residual' and meta['selected_steps'] == 0:
            assert subset.changed_dispatches.eq(0).all()
            prior = frame[frame.policy.eq('retuned_setup_spt')].set_index('scenario_seed').full_objective
            np.testing.assert_allclose(subset.set_index('scenario_seed').full_objective.sort_index(),prior.sort_index())
        metadata.append({k:meta[k] for k in ['condition','seed','selected_steps','physical_steps',
            'trained_rollout_physical_steps','physical_overshoot','outer_steps','choice_steps','episodes','gpu_update_calls','model_sha256']})
    for row in pd.read_csv(ROOT/'paired_comparisons.csv').to_dict('records'):
        recomputed = paired_statistics(frame,row['candidate'],row['reference'])
        for field in ['mean_difference','ci_low','ci_high']:
            assert np.isclose(row[field],recomputed[field])
    checks.update(status='passed',exact_grid=True,source_and_controls_unchanged=True,
        all_cache_rows_match=True,paired_uncertainty_reproduced=True,episodes_sha256=sha(ROOT/'episodes.csv'),
        models=metadata,optional_missing_fields='original SPT has no residual-change diagnostics')
    (ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    verify()
