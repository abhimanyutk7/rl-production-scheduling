"""Small predeclared zero-shot transfer study for the fitted priority policies."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inspection import InspectionConfig
from .inspection_study import source_fingerprint, paired_statistics, _run_lock
from .priority_search import _roll, _json, _csv, _table, INITIAL


def run(output='outputs/inspection_research/priority_transfer'):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    source=Path('outputs/inspection_research/priority_search')
    audit=Path('outputs/inspection_research/priority_audit')
    assert json.loads((source/'run.json').read_text())['status']=='complete'
    assert json.loads((audit/'run.json').read_text())['status']=='complete'
    # Exact longer configuration used in the frozen full-cycle study. Controllers
    # were trained on three days and are NOT retrained on these transfer cases.
    config=InspectionConfig(**json.loads(Path('outputs/inspection_research/full_cycle/run.json').read_text())['config'])
    policies=[source/f'policy_s{s}.json' for s in range(3)]
    control_file=audit/'baseline_validation.json'
    best=min(json.loads(control_file.read_text()),key=lambda r:r['validation_cost'])
    files=policies+[control_file]
    identity=dict(config=vars(config),source_hash=hashlib.sha256((source_fingerprint()+
        Path(__file__).read_text()+Path('factory_rl/priority_search.py').read_text()).encode()).hexdigest(),
        artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scenarios=list(range(23300,23310)),training_seeds=[0,1,2],
        protocol='zero-shot 3-day to 60-day, 30-day billing; no parameter selection on transfer data')
    controllers=[('learned_priority',s,np.array(json.loads(p.read_text())['weights'])) for s,p in enumerate(policies)]
    controllers += [('spt',-1,INITIAL),('retuned_setup_spt',-1,np.array(best['weights']))]
    expected={(p,s,t) for p,s,_ in controllers for t in identity['scenarios']}
    with _run_lock(root):
        manifest=root/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
            raise ValueError('Transfer protocol differs; use a new named output')
        _json(manifest,dict(status='running',identity=identity))
        path=root/'episodes.csv'
        rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        keys=[(r['policy'],r['training_seed'],r['scenario_seed']) for r in rows]
        if len(set(keys))!=len(keys) or not set(keys)<=expected:
            raise ValueError('Invalid transfer cache')
        completed=set(keys)
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(_roll,config,w,t):(p,s,t) for p,s,w in controllers
                     for t in identity['scenarios'] if (p,s,t) not in completed}
            for future in as_completed(futures):
                policy,seed,scenario=futures[future]
                metrics,_=future.result()
                rows.append(dict(policy=policy,training_seed=seed,scenario_seed=scenario,**metrics))
                _csv(path,pd.DataFrame(rows))
                print('Transferred',len(rows),'of',len(expected),'rows',flush=True)
        frame=pd.DataFrame(rows)
        assert len(frame)==len(expected) and set(zip(frame.policy,frame.training_seed,frame.scenario_seed))==expected
        assert frame.invalid_actions.eq(0).all()
        for r in rows:
            assert np.isclose(json.loads(r['ledger_daily'])[-1]['full'],r['full_objective'])
            periods=json.loads(r['ledger_billing'])
            assert len(periods)==2 and [x['period'] for x in periods]==[0,1]
            assert np.isclose(sum(x['demand_cost'] for x in periods),r['demand_cost'])
        summary=frame.groupby('policy')[['full_objective','completion_rate','on_time_rate','inspection_count',
            'maintenance_count','breakdowns','belief_coverage90','belief_health_mae']].mean().reset_index()
        paired=pd.DataFrame([paired_statistics(frame,'learned_priority',r) for r in ['spt','retuned_setup_spt']])
        _csv(root/'summary.csv',summary);_csv(root/'paired_comparisons.csv',paired)
        text=['# Zero-shot longer-horizon transfer','',
            'Controllers trained/selected on three-day accelerated-billing episodes are evaluated without retraining on 60 simulated days and two genuine 30-day billing periods. New scenario sequences23300-23309; three fitted search seeds. The two simple rules are deterministic comparators, not three independently trained models.',
            'This is a small exploratory transfer check, not the separate original500-sequence neural study and not real-factory validation. Costs span60days and must not be compared directly with three-day costs. Intervals resample training seeds and paired scenario sequences; ten sequences give limited power.',
            '',_table(summary),'','## Learned minus reference cost, paired95% intervals','',_table(paired)]
        (root/'report.md').write_text('\n\n'.join(text),encoding='utf-8')
        _json(manifest,dict(status='complete',identity=identity,rows=len(frame),invalid_actions=0,
            ledgers='daily objectives and both 30-day period charges reconciled'))
        print('COMPLETE',root,len(frame),'rows',flush=True)


if __name__=='__main__':
    run()
