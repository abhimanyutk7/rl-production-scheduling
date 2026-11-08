"""Interpretable episodic policy search, separate from frozen PPO studies.

Cross-entropy policy search is an established optimisation method, not a new PPO
algorithm. Candidate coefficients are selected by training/validation cost only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .inspection import InspectionConfig, InspectionEnv
from .inspection_study import source_fingerprint, rollout, paired_statistics, _run_lock

FEATURES=['duration','setup','slack','age','operations_left','nominal_energy']
INITIAL=np.array([1.,.5,0.,0.,0.,0.,.7,1.])
LOW=np.array([.1,0.,0.,-2.,-2.,0.,.35,.03])
HIGH=np.array([3.,3.,3.,0.,2.,3.,.9,1.])
TEST_START=22300  # Engineering verification used 22100-22101 and 22200-22201.
VALIDATION_START=10100


def feature_terms(job,state,config):
    return np.array([job['duration']/60,job['setup']/30,
        np.clip(job['slack']/config.shift_minutes,-4,4),np.clip(job['age']/1440,0,3),
        job['operations_left']/config.max_operations,
        job['duration']*job['power']*config.tariff(state['time'])/60000],dtype=float)


def priority_action(state,config,weights):
    """Consumes only policy_state; never hidden condition or future work."""
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(8,) or not np.isfinite(weights).all():
        raise ValueError('Eight finite policy coefficients required')
    pm,inspect,wait=config.shortlist,config.shortlist+1,config.shortlist+2
    mask=state['action_mask']
    if state['machine']<0:return wait
    machine=state['machines'][state['machine']]
    if mask[inspect] and machine['std']>weights[7]:return inspect
    if mask[pm] and machine['mean']<weights[6]:return pm
    legal=[j for j in range(len(state['candidates'])) if mask[j]]
    if legal:
        return min(legal,key=lambda j:float(feature_terms(state['candidates'][j],state,config)@weights[:6]))
    return wait


def _roll(config,weights,scenario,example=False):
    env=InspectionEnv(config,strict_actions=True)
    env.reset(seed=int(scenario))
    first=None
    while not env.terminated:
        state=env.policy_state()
        action=priority_action(state,config,weights)
        if example and first is None and len(state['candidates'])>=2 and action<config.shortlist:
            first=dict(scenario=int(scenario),time=env.time,machine=state['machine'],chosen_action=action,
                belief=state['machines'][state['machine']],
                explanation='Exact terms of this readable priority rule; not a neural-network explanation or a causal estimate.',
                candidates=[dict(job=j['job'],features=dict(zip(FEATURES,feature_terms(j,state,config))),
                    weighted_terms=dict(zip(FEATURES,feature_terms(j,state,config)*weights[:6])),
                    score=float(feature_terms(j,state,config)@weights[:6])) for j in state['candidates']])
        env.step(action)
    metrics=env.metrics()
    for key in ('ledger_daily','ledger_billing'):metrics[key]=json.dumps(metrics[key])
    return metrics,first


def _fitness(job):
    config,weights,seeds=job
    config=InspectionConfig(**config)
    return float(np.mean([_roll(config,weights,seed)[0]['full_objective'] for seed in seeds]))


def _json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)


def _csv(path,frame):
    temp=path.with_suffix('.tmp');frame.to_csv(temp,index=False);temp.replace(path)


def bind_evaluation(root,policies,rules_hash):
    identity=dict(policy_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in policies},rules_sha256=rules_hash)
    path=root/'evaluation_identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:
        raise ValueError('Evaluation policy artifacts changed; use a new output')
    _json(path,identity)


def _table(frame):
    return '\n'.join(['|'+'|'.join(frame.columns)+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+
        ['|'+'|'.join(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+'|'
         for row in frame.itertuples(index=False,name=None)])


def run(args):
    import pandas as pd
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=Path('outputs/inspection_research/final_pilot')
    config=InspectionConfig(**json.loads((base/'run.json').read_text())['config'])
    rules_path=base/'selected_rules.json';rules=json.loads(rules_path.read_text())
    identity=dict(config=asdict(config),source_hash=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()).encode()).hexdigest(),
        rules_sha256=hashlib.sha256(rules_path.read_bytes()).hexdigest(),seeds=args.seeds,
        iterations=args.iterations,population=args.population,training_scenarios_per_iteration=args.train_scenarios,
        episodes=args.episodes,validation=args.validation,test_start=TEST_START,validation_start=VALIDATION_START,
        features=FEATURES,initial_weights=INITIAL.tolist(),lower=LOW.tolist(),upper=HIGH.tolist(),
        method='Gaussian cross-entropy direct policy search; 25 percent elites, smoothed updates and nonzero variance floor')
    with _run_lock(root):
        manifest=root/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
            raise ValueError('Protocol differs; use a new named output')
        _json(manifest,dict(status='running',identity=identity))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for seed in range(args.seeds):
                checkpoint=root/f'policy_s{seed}.json'
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_text())
                    if saved['identity']!=identity or saved['seed']!=seed:
                        raise ValueError('Policy identity differs')
                    continue
                started=time.perf_counter()
                rng=np.random.default_rng(seed+781)
                mean=INITIAL.copy();std=(HIGH-LOW)*.3
                incumbent=INITIAL.copy();best=INITIAL.copy()
                validation_seeds=list(range(VALIDATION_START,VALIDATION_START+args.validation))
                best_cost=_fitness((asdict(config),best,validation_seeds))
                initial_validation=best_cost
                history=[dict(iteration=-1,validation_cost=best_cost,weights=best.tolist())]
                for iteration in range(args.iterations):
                    candidates=np.clip(rng.normal(mean,std,size=(args.population,8)),LOW,HIGH)
                    candidates[0]=incumbent;candidates[1]=INITIAL;candidates[2]=mean
                    train_ids=rng.choice(1000,size=args.train_scenarios,replace=False).tolist()
                    costs=list(pool.map(_fitness,[(asdict(config),w,train_ids) for w in candidates]))
                    order=np.argsort(costs)
                    incumbent=candidates[order[0]].copy()
                    elite=candidates[order[:max(2,args.population//4)]]
                    mean=.3*mean+.7*elite.mean(axis=0)
                    std=np.maximum(.3*std+.7*elite.std(axis=0),(HIGH-LOW)*.03)
                    value=_fitness((asdict(config),incumbent,validation_seeds))
                    if value<best_cost:best,best_cost=incumbent.copy(),value
                    history.append(dict(iteration=iteration,training_scenarios=train_ids,
                        training_cost=float(costs[order[0]]),validation_cost=value,selected_validation_cost=best_cost,
                        nominee=incumbent.tolist(),selected_weights=best.tolist()))
                    _json(root/f'training_s{seed}.json',history)
                    print(f'seed {seed} iteration {iteration}: validation {value:.3f}; selected {best_cost:.3f}',flush=True)
                _json(checkpoint,dict(identity=identity,seed=seed,weights=best.tolist(),initial_validation_cost=initial_validation,
                    selected_validation_cost=best_cost,initial_retained=bool(np.array_equal(best,INITIAL)),
                    training_seconds=time.perf_counter()-started,
                    selection='Baseline and one training-selected nominee per iteration; validation only'))
        bind_evaluation(root,[root/f'policy_s{s}.json' for s in range(args.seeds)],identity['rules_sha256'])
        path=root/'episodes.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else []
        expected={(policy,seed,scenario) for policy in ['learned_priority','learned_no_inspection']
                  for seed in range(args.seeds) for scenario in range(TEST_START,TEST_START+args.episodes)}
        expected|={(policy,-1,scenario) for policy in rules for scenario in range(TEST_START,TEST_START+args.episodes)}
        keys=[(r['policy'],r['training_seed'],r['scenario_seed']) for r in rows]
        if len(set(keys))!=len(keys) or not set(keys)<=expected:raise ValueError('Cached evaluation grid differs')
        completed=set(keys)
        controllers=[]
        for seed in range(args.seeds):
            weights=np.array(json.loads((root/f'policy_s{seed}.json').read_text())['weights'])
            no_inspection=weights.copy();no_inspection[7]=1.
            controllers.extend([('learned_priority',seed,weights),('learned_no_inspection',seed,no_inspection)])
        controllers.extend((policy,-1,parameters) for policy,parameters in rules.items())
        for policy,seed,parameters in controllers:
            for scenario in range(TEST_START,TEST_START+args.episodes):
                if (policy,seed,scenario) in completed:continue
                if seed==-1:metrics,_=rollout(config,policy,scenario,parameters)
                else:metrics,_=_roll(config,parameters,scenario)
                rows.append(dict(policy=policy,training_seed=seed,scenario_seed=scenario,**metrics))
                _csv(path,pd.DataFrame(rows))
            print('Evaluated',policy,seed,flush=True)
        frame=pd.DataFrame(rows)
        assert len(frame)==len(expected) and set(zip(frame.policy,frame.training_seed,frame.scenario_seed))==expected
        assert frame.invalid_actions.eq(0).all()
        for r in rows:
            assert np.isclose(json.loads(r['ledger_daily'])[-1]['full'],r['full_objective'])
            assert np.isclose(sum(x['demand_cost'] for x in json.loads(r['ledger_billing'])),r['demand_cost'])
        summary=frame.groupby('policy')[['full_objective','completion_rate','inspection_count','maintenance_count','breakdowns','belief_coverage90']].mean().reset_index()
        comparisons=pd.DataFrame([paired_statistics(frame,'learned_priority',r) for r in ['spt','belief','uncertainty','reactive','learned_no_inspection']])
        _csv(root/'summary.csv',summary);_csv(root/'paired_comparisons.csv',comparisons)
        weights=json.loads((root/'policy_s0.json').read_text())['weights']
        _,example=_roll(config,weights,TEST_START,example=True)
        _json(root/'decision_example.json',example)
        text=['# Interpretable scheduling policy search','',
            'Exploratory three-day accelerated-billing simulation; synthetic data. Coefficients learned from whole-episode cost feedback using Gaussian cross-entropy search. This is direct black-box policy optimisation, not neural PPO or a newly invented RL algorithm.',
            f'Training scenario IDs are below 1000. Validation starts {VALIDATION_START}; untouched test starts {TEST_START}. Each training population shares the same sampled scenario IDs. Validation can retain the original SPT + belief-maintenance rule.',
            'The no-inspection comparison disables inspections in the learned policy without retraining. It is a fixed-parameter intervention, not a separately optimised no-inspection method. Scores and inspection thresholds are heuristics, not calculated value-of-information guarantees.',
            'This adds no real-factory validation and no claim of world-first novelty. Read costs with completion, paired uncertainty, seed counts and actual inspection use.',
            '',_table(summary),'','## Paired cost differences; negative favours learned priority','',_table(comparisons),'','## Learned weights','']
        for seed in range(args.seeds):text.append((root/f'policy_s{seed}.json').read_text())
        (root/'report.md').write_text('\n\n'.join(text),encoding='utf-8')
        _json(manifest,dict(status='complete',identity=identity,rows=len(frame),invalid_actions=0,ledgers='reconciled'))
        print('COMPLETE',root,len(frame),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/inspection_research/priority_search')
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--population',type=int,default=20)
    p.add_argument('--train-scenarios',type=int,default=4)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--validation',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if min(a.seeds,a.iterations,a.train_scenarios,a.episodes,a.validation,a.workers)<1 or a.population<4 or a.validation>400 or a.train_scenarios>1000 or a.episodes>500:
        p.error('Invalid budget or scenario partition')
    run(a)
