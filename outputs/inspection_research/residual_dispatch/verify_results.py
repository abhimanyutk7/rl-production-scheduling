"""Reproduce saved-result checks and the declared zero-shot transfer study."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accounting(frame, config):
    core = ['full_objective', 'completion_rate', 'on_time_rate', 'maintenance_count',
            'inspection_count', 'breakdowns', 'belief_coverage90', 'energy_cost', 'demand_cost']
    assert np.isfinite(frame[core].to_numpy()).all()
    assert frame.invalid_actions.eq(0).all()
    daily_count, billing_count = 0, 0
    for r in frame.to_dict('records'):
        daily, bills = json.loads(r['ledger_daily']), json.loads(r['ledger_billing'])
        assert [d['day'] for d in daily] == list(range(config['days']))
        assert [b['period'] for b in bills] == list(range(int(np.ceil(config['days']/config['billing_days']))))
        daily_count += len(daily)
        billing_count += len(bills)
        assert np.isclose(daily[-1]['full'], r['full_objective'], rtol=1e-10, atol=1e-7)
        assert np.isclose(sum(b['demand_cost'] for b in bills), r['demand_cost'])
        for d in daily:
            assert np.isfinite(list(d.values())).all()
            assert np.isclose(sum(d[k] for k in ['lateness','terminal','energy','demand','downtime','inspection']), d['full'])
        for b in bills:
            assert np.isclose(b['demand_cost'], b['peak_kw']*config['demand_rate'])
        objective = (config['tardiness_weight']*r['total_tardiness_min']/config['tardiness_scale']
            + config['energy_weight']*r['energy_cost']/config['energy_scale']
            + config['demand_weight']*r['demand_cost']/config['demand_scale']
            + config['maintenance_weight']*(r['maintenance_downtime_min']+r['repair_downtime_min']+r['inspection_downtime_min'])/config['downtime_scale']
            + r['inspection_direct_cost']/config['energy_scale']+config['unfinished_penalty']*r['unfinished_jobs'])
        assert np.isclose(objective, r['full_objective'])
    return dict(rows=len(frame), invalid_actions=0, daily_records=daily_count, billing_records=billing_count,
                accounting='all daily/billing/objective formulas reconciled')


def verify():
    from factory_rl.inspection_study import source_fingerprint, paired_statistics
    manifest = json.loads((ROOT/'run.json').read_text())
    assert manifest['status'] == 'complete', 'Do not analyse an unfinished primary study'
    identity = manifest['identity']
    fingerprint = hashlib.sha256((source_fingerprint()+(PROJECT/'factory_rl/residual_dispatch.py').read_text()
        +(PROJECT/'factory_rl/priority_search.py').read_text()).encode()).hexdigest()
    assert fingerprint == identity['source_hash']
    for name, digest in identity['controls_sha256'].items():
        assert sha(PROJECT/name) == digest
    cfg, args = identity['config'], identity['settings']
    frame = pd.read_csv(ROOT/'episodes.csv')
    jobs = [(p, s) for s in range(args['seeds']) for p in ['residual','no_prior','legacy_adaptive','learned_priority']]
    jobs += [('retuned_setup_spt', -1), ('spt', -1)]
    scenarios = range(args['test_start'], args['test_start']+args['episodes'])
    expected = {(p,s,j) for p,s in jobs for j in scenarios}
    key = ['policy','training_seed','scenario_seed']
    actual = list(frame[key].itertuples(index=False, name=None))
    assert len(actual) == len(set(actual)) and set(actual) == expected
    checks = accounting(frame, cfg)
    models = []
    for policy, seed in jobs:
        cached = pd.read_csv(ROOT/f'{policy}_s{seed}.csv')
        subset = frame[frame.policy.eq(policy) & frame.training_seed.eq(seed)]
        # Different controls have no residual-change diagnostic, hence declared
        # missing values in the aggregate; their physical metrics are complete.
        pd.testing.assert_frame_equal(cached.sort_values(key).reset_index(drop=True),
            subset[cached.columns].sort_values(key).reset_index(drop=True), check_like=True, check_dtype=False)
        if policy not in ['residual','no_prior']:
            continue
        directory = ROOT/'models'/f'{policy}_s{seed}'
        meta = json.loads((directory/'metadata.json').read_text())
        validation = json.loads((directory/'validation.json').read_text())
        best = min(validation, key=lambda r:r['mean_cost'])
        assert meta['condition'] == policy and meta['seed'] == seed and meta['status'] == 'complete'
        assert meta['config'] == cfg and meta['schema'] == identity['schema']
        assert meta['selected_steps'] == best['step'] and meta['validation_cost'] == best['mean_cost']
        assert sha(directory/'model.zip') == meta['model_sha256']
        saved = json.loads(zipfile.ZipFile(directory/'model.zip').read('data'))
        assert saved['num_timesteps'] == meta['selected_steps'] and saved['gamma'] == 1
        assert meta['actual_steps'] >= args['steps']
        if args['device'] == 'cuda_updates':
            assert meta['gpu_update_calls'] > 0 and meta['final_parameter_device'].startswith('cuda')
        if policy == 'residual' and meta['selected_steps'] == 0:
            assert subset.changed_dispatches.eq(0).all()
            prior = frame[frame.policy.eq('retuned_setup_spt')].set_index('scenario_seed').full_objective
            np.testing.assert_allclose(subset.set_index('scenario_seed').full_objective.sort_index(), prior.sort_index())
        models.append({k:meta[k] for k in ['condition','seed','selected_steps','gpu_update_calls','model_sha256']})
    paired = pd.read_csv(ROOT/'paired_comparisons.csv')
    for r in paired.to_dict('records'):
        recomputed = paired_statistics(frame, r['candidate'], r['reference'])
        for key in ['mean_difference','ci_low','ci_high']:
            assert np.isclose(r[key], recomputed[key])
    supplemental = pd.DataFrame([paired_statistics(frame,'no_prior',p) for p in
        ['legacy_adaptive','retuned_setup_spt','spt','learned_priority']])
    supplemental['analysis_status'] = 'additional standalone-neural comparisons after primary run; exploratory unadjusted intervals'
    supplemental.to_csv(ROOT/'paired_standalone_supplement.csv',index=False)
    components = pd.DataFrame(dict(policy=frame.policy,
        lateness=cfg['tardiness_weight']*frame.total_tardiness_min/cfg['tardiness_scale'],
        energy=cfg['energy_weight']*frame.energy_cost/cfg['energy_scale'],
        demand=cfg['demand_weight']*frame.demand_cost/cfg['demand_scale'],
        downtime=cfg['maintenance_weight']*(frame.maintenance_downtime_min+frame.repair_downtime_min+frame.inspection_downtime_min)/cfg['downtime_scale'],
        inspection=frame.inspection_direct_cost/cfg['energy_scale'],unfinished=cfg['unfinished_penalty']*frame.unfinished_jobs))
    np.testing.assert_allclose(components.drop(columns='policy').sum(axis=1),frame.full_objective)
    components['full_objective'] = frame.full_objective
    components.groupby('policy').mean().reset_index().to_csv(ROOT/'cost_components.csv',index=False)
    checks.update(status='passed', exact_grid=True, source_and_controls_unchanged=True, all_cache_rows_match=True,
        episodes_sha256=sha(ROOT/'episodes.csv'), selected_checkpoints=models, paired_uncertainty_reproduced=True,
        optional_missing_fields='changed_dispatches and dispatch_decisions are not measured for external legacy/CEM/SPT controls')
    (ROOT/'verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return manifest


def decision_example(root, condition, model_path=None):
    import torch
    from sb3_contrib import MaskablePPO
    from factory_rl.inspection import InspectionConfig
    from factory_rl.residual_dispatch import DispatchEnv, PRIOR
    from factory_rl.priority_search import priority_action
    root = Path(root)
    configs = json.loads((root/'run.json').read_text())['identity']['config']
    choices = list((root/'models').glob(f'{condition}_s*/metadata.json'))
    path = Path(model_path) if model_path else min(choices,key=lambda p:json.loads(p.read_text())['validation_cost']).parent
    model = MaskablePPO.load(path/'model.zip',device='cpu')
    torch.set_num_threads(1)
    env = DispatchEnv(InspectionConfig(**configs))
    obs,_ = env.reset(seed=997)  # Training split; never pick a test example for selection.
    while not env.unwrapped.terminated:
        action = int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
        if action < env.unwrapped.config.shortlist and env.action_masks().sum() >= 2:
            tensor = torch.as_tensor(obs[None])
            extractor = model.policy.mlp_extractor
            with torch.no_grad():
                jobs,context = extractor.split(tensor)
                residual = 2*torch.tanh(extractor.job_net(torch.cat([jobs[:,:,:6],
                    context[:,None,:].expand(-1,extractor.shortlist,-1)],dim=-1)).squeeze(-1))
                prior = jobs[:,:,6] if extractor.use_prior else torch.zeros_like(residual)
                probs = model.policy.get_distribution(tensor,action_masks=env.action_masks()[None]).distribution.probs[0].numpy()
                logits = extractor.forward_actor(tensor)[0].numpy()
            np.testing.assert_allclose((prior+residual)[0].numpy(),logits[:extractor.shortlist],atol=1e-6)
            assert int(probs.argmax()) == action
            state = env.unwrapped.policy_state()
            result = dict(scenario=997,split='training; illustrative actual prediction',condition=condition,
                selected_by='common validation selection; not test selection' if model_path else 'minimum validation cost among this condition; not best test seed',
                model=str(path.relative_to(PROJECT)),model_sha256=sha(path/'model.zip'),
                chosen_action=action,prior_action=priority_action(state,env.unwrapped.config,PRIOR),
                time=state['time'],machine=state['machine'],public_context=dict(zip(env.observation_schema[-12:],context[0].numpy().astype(float).tolist())),
                explanation='Exact actor logit decomposition and masked probabilities; not a causal feature attribution or optimality guarantee.',
                service='fixed public rule; no learned inspection/maintenance',
                candidates=[dict(job=int(j['job']),duration=j['duration'],setup=j['setup'],slack=j['slack'],
                    prior_logit=float(prior[0,i]),neural_residual_logit=float(residual[0,i]),
                    final_logit=float(logits[i]),masked_probability=float(probs[i]))
                    for i,j in enumerate(state['candidates'])])
            (root/'decision_example.json').write_text(json.dumps(result,indent=2))
            return
        obs,_,_,_,_ = env.step(action)
    raise RuntimeError('No nontrivial job choice found in the declared illustrative scenario')


def plot_comparisons():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(ROOT/'paired_standalone_supplement.csv')
    labels = {'legacy_adaptive':'Old adaptive neural controller', 'retuned_setup_spt':'Retuned setup-aware SPT',
              'spt':'Original SPT', 'learned_priority':'Learned CEM priority policies'}
    with plt.rc_context({'font.size':11}):
        fig,ax = plt.subplots(figsize=(11,5.3))
        means = data.mean_difference.to_numpy()
        y = np.arange(len(data))
        ax.errorbar(means,y,xerr=np.vstack([means-data.ci_low,data.ci_high-means]),fmt='o',
                    color='#245879',ecolor='#245879',capsize=4,lw=1.7)
        ax.axvline(0,color='#333333',linestyle='--',lw=1)
        ax.set_yticks(y,[labels[p] for p in data.reference])
        ax.invert_yaxis()
        ax.set_xlim(-15,10)
        ax.set_xlabel('Standalone neural PPO minus reference: normalised cost')
        ax.set_title('Shared neural scorer: improvement over old neural controller,\nbut no established superiority to strong rules',pad=17)
        ax.grid(axis='x',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
        fig.subplots_adjust(left=.31,right=.97,top=.79,bottom=.27)
        fig.text(.31,.14,'3 training seeds; 50 matched three-day sequences (24500-24549).',fontsize=9)
        fig.text(.31,.095,'Exploratory supplemental comparisons; unadjusted 95% seed/scenario bootstrap intervals.',fontsize=9)
        fig.text(.31,.05,'Negative favours the new neural scorer. Fixed public maintenance rule; no learned inspection.',fontsize=9)
        fig.savefig(ROOT/'standalone_comparisons.png',dpi=180)
        plt.close(fig)




if __name__ == '__main__':
    verify()
    decision_example(ROOT,'no_prior')
    plot_comparisons()
