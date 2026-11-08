"""Controlled inspection-value diagnostics, distinct from held-out policy results."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inspection import InspectionConfig, InspectionEnv
from .inspection_study import compare_actions, public_action


def information_value(prior,action_costs,sensor_likelihood,sensing_cost):
    """Standard finite-table expected sample information value, net of its cost.

    This analytical teaching/probe calculation is not a new algorithm or a
    measured factory outcome. Costs have rows=hidden states and columns=actions;
    sensor likelihood has rows=states and columns=possible observations.
    """
    prior=np.asarray(prior,dtype=float)
    costs=np.asarray(action_costs,dtype=float)
    likelihood=np.asarray(sensor_likelihood,dtype=float)
    if prior.ndim!=1 or costs.ndim!=2 or likelihood.ndim!=2 or len(prior)!=len(costs) or len(prior)!=len(likelihood):
        raise ValueError('State dimensions must match')
    if not np.isfinite(costs).all() or not np.isfinite(prior).all() or not np.isfinite(likelihood).all():
        raise ValueError('Finite cost and probability tables are required')
    if np.any(prior<0) or np.any(likelihood<0) or not np.isclose(prior.sum(),1) or not np.allclose(likelihood.sum(axis=1),1):
        raise ValueError('Prior and likelihood rows must be probability distributions')
    before=float(np.min(prior@costs))
    after=sum(float(np.min((prior*likelihood[:,o])@costs)) for o in range(likelihood.shape[1]))
    return before-after-float(sensing_cost)


def system_diagram(path):
    """Reproducible presentation figure; describes the implementation, not results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(16,9))
    fig.patch.set_facecolor('#f5f8fc'); ax.set_facecolor('#f5f8fc')
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    ax.text(.04,.94,'Learning when to inspect a factory machine',fontsize=25,weight='bold',color='#17324d')
    ax.text(.04,.895,'Implemented research system | synthetic factory | default observation: 307 numbers',fontsize=13,color='#50677e')
    def box(x,y,w,h,title,body,color='#ffffff'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                     facecolor=color,edgecolor='#abc0d2',linewidth=1.6))
        ax.text(x+.015,y+h-.035,title,fontsize=15,weight='bold',color='#17324d',va='top')
        ax.text(x+.015,y+h-.087,body,fontsize=12,color='#304d67',va='top',linespacing=1.55)
    def arrow(a,b,color='#3b7897',bend=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,linewidth=2,
                     color=color,connectionstyle=f'arc3,rad={bend}'))
    box(.05,.42,.24,.35,'Persistent factory','Arrived jobs and operations\nMachines, setups and service\nHidden stochastic wear\nFailures and repairs\nWork carries across shifts')
    box(.375,.55,.245,.22,'Condition belief','Particle estimate + uncertainty\nPublic operating history\nNoisy inspection measurements','#e9f3f8')
    box(.72,.55,.235,.22,'Masked controller','Public-rule imitation + PPO\nTwo 64-unit hidden layers\nInvalid choices masked','#e8f3ec')
    box(.72,.26,.235,.19,'One decision','Dispatch / inspect\nMaintain / wait','#e8f3ec')
    box(.05,.12,.24,.17,'Energy and time ledger','15-minute demand windows\nBilling peaks and late work')
    box(.375,.12,.245,.17,'Realised objective','Lateness + energy + demand\nService + inspection costs\nUnfinished-work charge')
    arrow((.30,.64),(.36,.64)); ax.text(.325,.68,'Observe',fontsize=11,color='#3b7897',ha='center')
    arrow((.635,.66),(.705,.66)); ax.text(.67,.705,'307 inputs',fontsize=11,color='#3b7897',ha='center')
    arrow((.837,.535),(.837,.465)); ax.text(.858,.495,'Choose',fontsize=11,color='#3b7897')
    arrow((.707,.35),(.305,.46),bend=-.12); ax.text(.46,.38,'Execute in simulation',fontsize=12,color='#3b7897')
    arrow((.17,.405),(.17,.305)); arrow((.305,.205),(.36,.205))
    for a,b in [((.635,.205),(.985,.205)),((.985,.205),(.985,.81)),
                ((.985,.81),(.837,.81)),((.837,.81),(.837,.785))]:
        arrow(a,b,color='#a36835')
    ax.text(.997,.49,'Negative cost increment',rotation=90,fontsize=11,color='#a36835',va='center',ha='left')
    ax.text(.05,.045,'Compact partial observation; approximate belief. Performance and novelty require experiments; the diagram does not imply superiority.',fontsize=11,color='#50677e')
    fig.savefig(path,dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor()); plt.close(fig)


def mechanism_probe(output, worlds=24, horizon=180):
    """Predeclared diagnostic grid, not a claim that any learned policy wins."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=InspectionConfig(days=3,billing_days=2,machines=3,machine_slots=5,
                            jobs=18,particles=128,initial_health_low=0.2)
    env=InspectionEnv(config,strict_actions=True)
    env.reset(seed=10010)  # Diagnostic validation scenario, not a held-out performance test.
    while env.time<90 or not env.candidates():
        env.step(public_action(env.policy_state(),config,'reactive'))
    rows,estimates=[],[]
    for headroom in (0.,60.):
        for noise in (.01,.1):
            for cost in (0.,25.,100.):
                parent=env.snapshot()
                parent.config=replace(config,inspection_noise=noise,inspection_cost=cost)
                parent.peak_kw+=headroom
                actions=[0,parent.maintenance_action,parent.inspection_action]
                actions=[a for a in actions if parent.action_masks()[a]]
                table=compare_actions(parent,actions,worlds=worlds,horizon=horizon,seed=720)
                table['extra_peak_headroom_kw']=headroom
                table['inspection_noise']=noise
                table['inspection_cost']=cost
                estimates.extend(table.to_dict('records'))
                lookup=table.set_index('action')
                inspect=np.array(json.loads(lookup.loc[parent.inspection_action,'cost_samples']))
                row=dict(extra_peak_headroom_kw=headroom,inspection_noise=noise,
                         inspection_cost=cost,worlds=worlds,horizon=horizon,
                         state_time=parent.time,diagnostic_scenario=10010)
                for name,action in [('process',0),('maintain',parent.maintenance_action)]:
                    diff=inspect-np.array(json.loads(lookup.loc[action,'cost_samples']))
                    se=diff.std(ddof=1)/np.sqrt(worlds) if worlds>1 else 0.
                    row[f'inspect_minus_{name}']=float(diff.mean())
                    row[f'{name}_ci_low']=float(diff.mean()-1.96*se)
                    row[f'{name}_ci_high']=float(diff.mean()+1.96*se)
                row['inspection_lower_than_both_means']=row['inspect_minus_process']<0 and row['inspect_minus_maintain']<0
                row['inspection_lower_with_both_intervals']=row['process_ci_high']<0 and row['maintain_ci_high']<0
                rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'inspection_value.csv',index=False)
    pd.DataFrame(estimates).to_csv(output/'action_estimates.csv',index=False)
    _plot_value(frame,output/'inspection_value.png')
    note=['# Controlled inspection-value diagnostic','',
          'This grid uses one predeclared validation snapshot after 90 calendar minutes, '
          'with a wider initial health prior. It is not held-out policy-performance evidence.',
          'Columns compare the estimated incremental cost of inspection with processing '
          'or maintenance. Negative values favour inspection under the stated model and horizon.',
          f'{worlds} matched latent worlds per action and a {horizon}-minute calendar horizon are used. '
          'Intervals describe Monte Carlo sampling error; they do not cover model error.',
          'Future arrivals are forecasts conditional on the public daily count/history. '
          'Future job attributes use training benchmark-duration pools, so held-out '
          'source-group mismatch is an explicit forecasting limitation.',
          'No residual-value approximation is added at the diagnostic horizon. A deferred '
          'breakdown beyond that horizon is not counted. Full policy experiments are separate.',
          f"Inspection had lower mean estimated cost than both alternatives in {int(frame.inspection_lower_than_both_means.sum())}/{len(frame)} grid cells.",
          f"Both paired upper interval bounds were below zero in {int(frame.inspection_lower_with_both_intervals.sum())}/{len(frame)} cells.",
          'If changing peak headroom barely changes these values, this diagnostic does '
          'not support a strong billing-history interaction. Do not hide that outcome.',
          'The value-of-information concept is established research. This diagnostic '
          'investigates a particular scheduling interaction, not a new general theorem.']
    (output/'report.md').write_text('\n\n'.join(note)+'\n',encoding='utf-8')
    return frame


def _plot_value(frame,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    magnitude=max(.01,float(np.abs(frame[['inspect_minus_process','inspect_minus_maintain']].to_numpy()).max()))
    for ax,(headroom,group) in zip(axes,frame.groupby('extra_peak_headroom_kw')):
        table=group.pivot(index='inspection_noise',columns='inspection_cost',values='inspect_minus_process')
        image=ax.imshow(table.to_numpy(),cmap='RdBu_r',vmin=-magnitude,vmax=magnitude,aspect='auto')
        ax.set_xticks(range(len(table.columns)),[str(x) for x in table.columns])
        ax.set_yticks(range(len(table.index)),[str(x) for x in table.index])
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center')
        ax.set_title(f'Added inherited peak: {headroom:g} kW')
        ax.set_xlabel('Inspection direct cost (synthetic INR)')
    axes[0].set_ylabel('Inspection noise standard deviation')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Cost difference: inspection minus processing')
    fig.suptitle('Controlled model diagnostic; not measured policy improvement')
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def explore_regimes(output,worlds=16):
    """Exploratory, fully exported grid; discoveries require new confirmation cases."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    identity=dict(worlds=worlds,loads=[18,30],state_minutes=[0,90,240],
                  durations=[1,5],headroom=[0,60],noise=[0,.1],cost=[0,25],horizon=180)
    metadata=output/'design.json'
    if metadata.exists() and json.loads(metadata.read_text())!=identity:
        raise ValueError('Exploration budget differs from cached grid')
    metadata.write_text(json.dumps(identity,indent=2))
    path=output/'regimes.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    for load in identity['loads']:
        config=InspectionConfig(days=3,billing_days=2,machines=3,machine_slots=5,
                                jobs=load,particles=128,initial_health_low=.2)
        env=InspectionEnv(config,strict_actions=True); env.reset(seed=10011)
        for target in identity['state_minutes']:
            while env.time<target or not env.candidates():
                env.step(public_action(env.policy_state(),config,'reactive'))
            for duration in identity['durations']:
                for headroom in identity['headroom']:
                    for noise in identity['noise']:
                        for cost in identity['cost']:
                            key=f'{load}/{target}/{duration}/{headroom}/{noise}/{cost}'
                            if any(r['key']==key for r in rows):
                                continue
                            parent=env.snapshot(); parent.peak_kw+=headroom
                            parent.config=replace(config,inspection_minutes=duration,inspection_noise=noise,inspection_cost=cost)
                            actions=[0,parent.maintenance_action,parent.inspection_action]
                            actions=[a for a in actions if parent.action_masks()[a]]
                            if parent.inspection_action not in actions or parent.maintenance_action not in actions:
                                rows.append(dict(key=key,status='service_masked',jobs=load,target=target))
                                continue
                            table=compare_actions(parent,actions,worlds=worlds,horizon=180,seed=721).set_index('action')
                            sensing=np.array(json.loads(table.loc[parent.inspection_action,'cost_samples']))
                            row=dict(key=key,status='complete',jobs=load,target=target,time=parent.time,
                                     inspection_minutes=duration,headroom=headroom,noise=noise,cost=cost,worlds=worlds)
                            for name,action in [('process',0),('maintain',parent.maintenance_action)]:
                                difference=sensing-np.array(json.loads(table.loc[action,'cost_samples']))
                                row[f'inspect_minus_{name}']=float(difference.mean())
                                row[f'{name}_ci_high']=float(difference.mean()+1.96*difference.std(ddof=1)/np.sqrt(worlds))
                            row['lower_mean_than_both']=row['inspect_minus_process']<0 and row['inspect_minus_maintain']<0
                            row['upper_bounds_below_zero']=row['process_ci_high']<0 and row['maintain_ci_high']<0
                            rows.append(row); pd.DataFrame(rows).to_csv(path,index=False)
    frame=pd.DataFrame(rows); frame.to_csv(path,index=False)
    complete=frame[frame.status=='complete']
    text=['# Exploratory inspection regimes','',
          'This is a declared exploratory grid of workload, state time, inspection duration, '
          'noise, cost and peak headroom. All cells, including unfavourable ones, are exported.',
          'It uses diagnostic validation scenario 10011 and a wide initial health prior [0.2,1], '
          'not the main experiment prior [0.6,1]. These are model assumptions, not measurements.',
          f"Completed {len(complete)} cells. Inspection had lower mean truncated cost than both alternatives in "
          f"{int(complete.lower_mean_than_both.sum())} cells. Both upper paired bounds were below zero in "
          f"{int(complete.upper_bounds_below_zero.sum())} cells.",
          'These are Monte Carlo diagnostic intervals, with no correction for scanning many cells. '
          'A favourable cell is a hypothesis for independent confirmation, not a statistically '
          'established discovery, RL improvement, or evidence of practical savings.',
          'The 180-minute estimate has no residual value. Longer-horizon policy experiments '
          'and model-mismatch checks are needed. No cell has been omitted because it was unfavourable.']
    (output/'report.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    return frame


def nested_interval(matrix,draws=2000):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim!=2 or not np.isfinite(matrix).all() or not matrix.size:
        raise ValueError('A finite scenario-by-world paired grid is required')
    rng=np.random.default_rng(989)
    values=[]
    for _ in range(draws):
        scenes=rng.integers(matrix.shape[0],size=matrix.shape[0])
        values.append(np.mean([matrix[s,rng.integers(matrix.shape[1],size=matrix.shape[1])].mean() for s in scenes]))
    return dict(mean=float(matrix.mean()),low=float(np.quantile(values,.0125)),
                high=float(np.quantile(values,.9875)),confidence=.975)


def confirm_regimes(output,worlds=128):
    """Fresh scenarios confirm/reject two disclosed economic regimes, not RL gains."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    regimes=[dict(name='fast_low_cost',duration=1,noise=.01,cost=25.),
             dict(name='default_inspection',duration=5,noise=.03,cost=100.)]
    design=dict(scenarios=list(range(10012,10017)),worlds=worlds,jobs=18,target=240,
                regimes=regimes,headroom=[0.,60.],horizon=180,initial_health_low=.2,
                provenance='Exploration-informed hypotheses, fresh validation scenarios, simulated model costs only')
    design_path=output/'design.json'
    if design_path.exists() and json.loads(design_path.read_text())!=design:
        raise ValueError('Confirmation design differs from cached results')
    design_path.write_text(json.dumps(design,indent=2))
    path=output/'action_estimates.csv'
    estimates=pd.read_csv(path).to_dict('records') if path.exists() else []
    for scenario in design['scenarios']:
        config=InspectionConfig(days=3,billing_days=2,machines=3,machine_slots=5,
                                jobs=18,particles=128,initial_health_low=.2)
        env=InspectionEnv(config,strict_actions=True); env.reset(seed=scenario)
        while env.time<240 or not env.candidates():
            env.step(public_action(env.policy_state(),config,'reactive'))
        for regime in regimes:
            for headroom in design['headroom']:
                key=f"{scenario}/{regime['name']}/{headroom}"
                if any(r['case']==key for r in estimates):
                    continue
                parent=env.snapshot(); parent.peak_kw+=headroom
                parent.config=replace(config,inspection_minutes=regime['duration'],
                                      inspection_noise=regime['noise'],inspection_cost=regime['cost'])
                actions=[0,parent.maintenance_action,parent.inspection_action]
                table=compare_actions(parent,actions,worlds,180,seed=scenario+800)
                table['case']=key; table['scenario']=scenario; table['regime']=regime['name']; table['headroom']=headroom
                estimates.extend(table.to_dict('records'))
                pd.DataFrame(estimates).to_csv(path,index=False)
    frame=pd.DataFrame(estimates)
    result=[]
    for regime in regimes:
        for headroom in design['headroom']:
            for name,action in [('process',0),('maintain',config.shortlist)]:
                paired=[]
                for scenario in design['scenarios']:
                    group=frame[(frame.scenario==scenario)&(frame.regime==regime['name'])&(frame.headroom==headroom)].set_index('action')
                    a=np.array(json.loads(group.loc[config.shortlist+1,'cost_samples']))
                    b=np.array(json.loads(group.loc[action,'cost_samples']))
                    paired.append(a-b)
                result.append(dict(regime=regime['name'],headroom=headroom,reference=name,
                                   scenarios=len(paired),worlds=worlds,**nested_interval(paired)))
    summary=pd.DataFrame(result); summary.to_csv(output/'paired_summary.csv',index=False)
    lines=['# Independent model-regime confirmation','',
           'Two regimes were fixed after exploration and evaluated on five fresh validation scenarios. '
           'The fast/low-cost regime is a hypothesis from the exploratory map; default inspection is a contrast.',
           'Both use a wider [0.2,1] initial-health prior than the main policy experiment. '
           'Neither is a claim about measured industrial inspection economics.',
           'Nested intervals resample scenarios and matched latent worlds. Each is 97.5%, '
           'chosen to allow two simultaneous inspection-versus-alternative comparisons within a regime/headroom pair. '
           'They do not cover all regime/headroom comparisons, forecast error or simulator error.',
           '|Regime|Extra peak headroom|Inspection minus alternative|Mean|97.5% interval|',
           '|---|---:|---|---:|---|']
    for row in result:
        lines.append(f"|{row['regime']}|{row['headroom']:g}|{row['reference']}|{row['mean']:.4f}|[{row['low']:.4f}, {row['high']:.4f}]|")
    lines += ['', 'These are 180-minute model-based incremental-cost comparisons with no residual value. '
              'A supported effect here is not evidence that PPO learned it or that the full policy beats rules. '
              'Read alongside the full-cycle held-out policy evaluation and negative results.']
    (output/'report.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def priority_audit(output='outputs/inspection_research/priority_audit'):
    """Supplementary validation-selected comparator and fixed-parameter ablations.

    Reuses the primary priority-search test grid; exploratory, not independent
    confirmation. No test outcome selects weights or comparator parameters.
    """
    import hashlib
    from .priority_search import INITIAL, _roll, _json, _csv, _table, TEST_START
    from .inspection_study import source_fingerprint, paired_statistics, _run_lock
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    source=Path('outputs/inspection_research/priority_search')
    primary=json.loads((source/'run.json').read_text())
    if primary['status']!='complete':raise ValueError('Wait for complete primary study')
    config=InspectionConfig(**primary['identity']['config'])
    policy_paths=[source/f'policy_s{s}.json' for s in range(primary['identity']['seeds'])]
    identity=dict(source=primary['identity'],source_hash=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()).encode()).hexdigest(),
        primary_episodes_sha256=hashlib.sha256((source/'episodes.csv').read_bytes()).hexdigest(),
        policies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in policy_paths},
        setup_weights=[0,1,2,4,8],maintenance_thresholds=[.55,.7,.85],
        interventions=['fixed_service','without_energy','without_urgency_age'])
    with _run_lock(root):
        manifest=root/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
            raise ValueError('Audit identity changed; use a new output stage')
        _json(manifest,dict(status='running',identity=identity))
        candidates=[]
        for setup in identity['setup_weights']:
            for threshold in identity['maintenance_thresholds']:
                w=INITIAL.copy();w[1]=setup/2;w[6]=threshold
                costs=[_roll(config,w,10100+j)[0]['full_objective'] for j in range(primary['identity']['validation'])]
                candidates.append(dict(setup_weight=setup,maintenance_threshold=threshold,
                    validation_cost=float(np.mean(costs)),weights=w.tolist()))
        _json(root/'baseline_validation.json',candidates)
        best=min(candidates,key=lambda x:x['validation_cost'])
        controllers=[('retuned_setup_spt',-1,np.array(best['weights']))]
        for seed,p in enumerate(policy_paths):
            w=np.array(json.loads(p.read_text())['weights'])
            fixed=w.copy();fixed[6]=.7;fixed[7]=1.
            no_energy=w.copy();no_energy[5]=0.
            no_urgency=w.copy();no_urgency[2:4]=0.
            controllers.extend([('fixed_service',seed,fixed),('without_energy',seed,no_energy),('without_urgency_age',seed,no_urgency)])
        scenarios=range(TEST_START,TEST_START+primary['identity']['episodes'])
        expected={(p,s,t) for p,s,_ in controllers for t in scenarios}
        path=root/'episodes.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        keys=[(r['policy'],r['training_seed'],r['scenario_seed']) for r in rows]
        if len(set(keys))!=len(keys) or not set(keys)<=expected:raise ValueError('Invalid audit cache')
        completed=set(keys)
        for policy,seed,w in controllers:
            for scenario in scenarios:
                if (policy,seed,scenario) in completed:continue
                metrics,_=_roll(config,w,scenario)
                rows.append(dict(policy=policy,training_seed=seed,scenario_seed=scenario,**metrics))
                _csv(path,pd.DataFrame(rows))
            print('Audited',policy,seed,flush=True)
        frame=pd.DataFrame(rows)
        assert len(frame)==len(expected) and frame.invalid_actions.eq(0).all()
        for r in rows:
            assert np.isclose(json.loads(r['ledger_daily'])[-1]['full'],r['full_objective'])
            assert np.isclose(sum(x['demand_cost'] for x in json.loads(r['ledger_billing'])),r['demand_cost'])
        combined=pd.concat([pd.read_csv(source/'episodes.csv'),frame],ignore_index=True)
        comparisons=pd.DataFrame([paired_statistics(combined,'learned_priority',reference)
            for reference in ['retuned_setup_spt','fixed_service','without_energy','without_urgency_age']])
        summary=combined.groupby('policy')[['full_objective','completion_rate','inspection_count','maintenance_count','breakdowns']].mean().reset_index()
        _csv(root/'paired_comparisons.csv',comparisons);_csv(root/'summary.csv',summary)
        text=['# Supplementary priority-controller audit','',
            'Exploratory reuse of the completed primary test grid. Baseline parameters selected using validation only. The setup-weighted SPT comparator searches 15 setup/maintenance combinations; this is a stronger comparator, not a new learning algorithm.',
            'Other comparisons change fixed parameters of each learned policy without retraining. They measure interventions on those fitted policies, not the best achievable separately trained ablations. Comparisons are exploratory and not adjusted across all examined contrasts.',
            f'Selected baseline: setup weight {best["setup_weight"]}, maintenance threshold {best["maintenance_threshold"]}.',
            '',_table(summary),'','## Learned minus reference cost with paired 95% intervals','',_table(comparisons)]
        (root/'report.md').write_text('\n\n'.join(text),encoding='utf-8')
        _json(manifest,dict(status='complete',identity=identity,rows=len(frame),invalid_actions=0,ledgers='reconciled'))
        print('COMPLETE',root,len(frame),'rows',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/inspection_research/mechanism')
    parser.add_argument('--worlds',type=int,default=24)
    parser.add_argument('--explore',action='store_true')
    parser.add_argument('--confirm',action='store_true')
    parser.add_argument('--priority-audit',action='store_true')
    args=parser.parse_args()
    if args.priority_audit:
        priority_audit(args.output)
    elif args.confirm:
        confirm_regimes(args.output,args.worlds)
    elif args.explore:
        explore_regimes(args.output,args.worlds)
    else:
        mechanism_probe(args.output,args.worlds)
