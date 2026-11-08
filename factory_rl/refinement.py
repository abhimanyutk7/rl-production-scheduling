"""Separate, resumable actor/critic calibration experiment; no frozen source edits.

Method motivation: Kernbach et al. (2026), arxiv.org/abs/2602.23804.
This is an adaptation, not a claim to invent actor-critic pretraining.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import gymnasium as gym
import numpy as np

from .inspection import InspectionConfig, InspectionEnv
from .inspection_study import source_fingerprint, rollout, load_model, paired_statistics

TEST_START = 21100  # Earlier engineering checks used 21000-21001.


def monte_carlo_returns(rewards):
    """Undiscounted full-episode returns, including the true terminal penalty."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError('Finite one-dimensional rewards required')
    return np.cumsum(rewards[::-1])[::-1].copy().astype(np.float32)


class ScaledReward(gym.RewardWrapper):
    """Fixed positive scaling changes gradient magnitudes, not objective ranking."""
    def __init__(self, env, scale):
        super().__init__(env)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Reward scale must be positive and finite')
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale

    def action_masks(self):
        return self.env.action_masks()


def fit_critic(model, observations, returns, updates=800, seed=0):
    """Fit only the separate value network; actor parameters remain unchanged."""
    import torch
    x = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    y = torch.as_tensor(returns, dtype=torch.float32, device=model.device)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise ValueError('Nonempty aligned observations and returns required')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Finite critic data required')
    params = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    def error():
        with torch.no_grad():
            return float(((model.policy.predict_values(x).flatten()-y)**2).mean())
    before = error()
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        indices = rng.integers(len(y), size=min(256,len(y)))
        loss = ((model.policy.predict_values(x[indices]).flatten()-y[indices])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
    return dict(examples=len(y), updates=updates, mse_before=before, mse_after=error())


def _json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def _csv(path, frame):
    temp = path.with_suffix('.tmp')
    frame.to_csv(temp,index=False)
    temp.replace(path)


def _table(frame):
    """Plain Markdown without an optional tabulate dependency."""
    def cell(value):
        return f'{value:.6g}' if isinstance(value,(float,np.floating)) else str(value).replace('|','/')
    return '\n'.join(['|'+'|'.join(map(str,frame.columns))+'|',
                      '|'+'|'.join(['---']*len(frame.columns))+'|']+
                     ['|'+'|'.join(map(cell,row))+'|' for row in frame.itertuples(index=False,name=None)])


def validate_cached_rows(rows, condition, seed, scenarios):
    if (len({r['scenario_seed'] for r in rows})!=len(rows)
        or any(r['scenario_seed'] not in scenarios or r['policy']!=condition
               or r['training_seed']!=seed for r in rows)):
        raise ValueError('Invalid cached evaluation identity/grid')


def _collect(model, config, seed, episodes=4):
    observations, returns = [], []
    scale = 1/config.days
    for j in range(episodes):
        env = InspectionEnv(config,strict_actions=True)
        obs,_ = env.reset(seed=(503+seed*31+j)%1000)
        episode_obs,rewards = [],[]
        while not env.terminated:
            episode_obs.append(obs.copy())
            action = int(model.predict(obs,action_masks=env.action_masks(),deterministic=False)[0])
            obs,reward,_,_,_ = env.step(action)
            rewards.append(reward*scale)
        assert np.isclose(-sum(rewards)/scale,env.metrics()['full_objective'])
        observations.extend(episode_obs)
        returns.extend(monte_carlo_returns(rewards))
    return np.asarray(observations,dtype=np.float32),np.asarray(returns,dtype=np.float32)


def _job(job):
    import torch
    import pandas as pd
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from .agents import ScenarioPool, TRAIN_SEEDS
    torch.set_num_threads(1)
    config = InspectionConfig(**job['config'])
    seed,condition = job['seed'],job['condition']
    directory = Path(job['output'])/'models'/f'{condition}_s{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    meta_path = directory/'metadata.json'
    model_path = directory/'model.zip'
    base_dir = Path(job['base'])/'models'/f'adaptive_s{seed}'
    source_model = base_dir/'imitation.zip'
    identity = {**job,'base_sha256':hashlib.sha256(source_model.read_bytes()).hexdigest()}
    env = InspectionEnv(config,strict_actions=True)
    history=[]
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['identity'] != identity:
            raise ValueError('Refinement job changed; use a new named output')
        if meta.get('status')=='complete':
            if hashlib.sha256(model_path.read_bytes()).hexdigest()!=meta['model_sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            model=MaskablePPO.load(model_path,device='cpu')
        else:
            model=None
    else:
        model=None
    if model is None:
        started=time.perf_counter()
        wrapper=ScaledReward(ScenarioPool(env,TRAIN_SEEDS,seed),1/config.days)
        model=MaskablePPO('MlpPolicy',wrapper,seed=seed,device='cpu',n_steps=1024,
            batch_size=128,n_epochs=4,gamma=1.,gae_lambda=.98,learning_rate=1e-4,
            clip_range=.1,target_kl=.01,ent_coef=0.,policy_kwargs={'net_arch':[64,64]},verbose=0)
        base=MaskablePPO.load(source_model,device='cpu')
        model.policy.load_state_dict(base.policy.state_dict())
        # Own-policy calibration samples actions; isolate its RNG consumption
        # from the matched PPO exploration stream used in both ablation arms.
        exploration_rng=torch.get_rng_state().clone()
        meta=dict(identity=identity,status='training',config=asdict(config),
            reward_scale=1/config.days,method='matched actor-only versus on-policy critic calibration',
            pretraining_source='own stochastic imitation-policy rollouts; training seeds only')
        _json(meta_path,meta)
        if condition=='actor_critic':
            x,y=_collect(model,config,seed)
            hx,hy=_collect(model,config,seed+100,episodes=1)
            with torch.no_grad():
                holdout_before=float(((model.policy.predict_values(torch.as_tensor(hx)).flatten()-torch.as_tensor(hy))**2).mean())
            actor_before={k:v.clone() for k,v in model.policy.state_dict().items() if 'value_net' not in k}
            meta['critic']=fit_critic(model,x,y,seed=seed)
            assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in actor_before.items())
            with torch.no_grad():
                meta['critic']['training_pool_holdout_mse_before']=holdout_before
                meta['critic']['training_pool_holdout_mse']=float(((model.policy.predict_values(torch.as_tensor(hx)).flatten()-torch.as_tensor(hy))**2).mean())
            meta['critic']['additional_simulation_decisions']=len(y)+len(hy)
        class Progress(BaseCallback):
            def __init__(self):
                super().__init__();self.best=float('inf');self.selected=0
            def validate(self):
                costs=[rollout(config,model,10020+j)[0]['full_objective'] for j in range(job['validation'])]
                diagnostics={k:float(v) for k,v in model.logger.name_to_value.items()
                             if k.startswith('train/') and np.isscalar(v)} if hasattr(model,'_logger') else {}
                history.append(dict(step=int(model.num_timesteps),mean_cost=float(np.mean(costs)),**diagnostics))
                if np.mean(costs)<self.best:
                    self.best=float(np.mean(costs));self.selected=int(model.num_timesteps)
                    model.save(model_path)
                _json(directory/'validation.json',history)
            def _on_step(self):
                if self.n_calls%8192==0:self.validate()
                return True
        progress=Progress();progress.validate()
        meta['initial_validation_cost']=history[0]['mean_cost']
        torch.set_rng_state(exploration_rng)
        model.learn(job['steps'],callback=progress)
        progress.validate()
        meta.update(status='complete',selected_steps=progress.selected,actual_steps=int(model.num_timesteps),
            validation_cost=progress.best,training_seconds=time.perf_counter()-started,
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
        _json(meta_path,meta)
        model=MaskablePPO.load(model_path,device='cpu')
        wrapper.close()
    path=directory/'test_episodes.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    expected=set(range(TEST_START,TEST_START+job['episodes']))
    validate_cached_rows(rows,condition,seed,expected)
    for scenario in sorted(expected-{r['scenario_seed'] for r in rows}):
        metrics,_=rollout(config,model,scenario)
        rows.append(dict(policy=condition,training_seed=seed,scenario_seed=scenario,**metrics))
        _csv(path,pd.DataFrame(rows))
    return rows


def run(args):
    import pandas as pd
    import torch
    from sb3_contrib import MaskablePPO
    from .inspection_study import _run_lock
    torch.set_num_threads(1)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    # A separate full run is not modified. Config and starting policies come from the completed pilot.
    base=Path(args.base)
    config=InspectionConfig(**json.loads((base/'run.json').read_text())['config'])
    fingerprint=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()).encode()).hexdigest()
    baseline_files=[base/'selected_rules.json']
    for seed in range(args.seeds):
        baseline_files.extend(base/'models'/f'adaptive_s{seed}'/name
                              for name in ('imitation.zip','model.zip','metadata.json'))
    baseline_hashes={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_files}
    identity=dict(config=asdict(config),source_hash=fingerprint,steps=args.steps,seeds=args.seeds,
        episodes=args.episodes,validation=args.validation,base=str(base.resolve()),
        test_start=TEST_START,validation_start=10020,
        conditions=['actor_only','actor_critic'],pretraining_episodes=4,critic_updates=800,
        baseline_sha256=baseline_hashes)
    with _run_lock(root):
        manifest=root/'run.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity']!=identity:
            raise ValueError('Protocol differs; choose a new named output stage')
        _json(manifest,dict(status='running',identity=identity))
        jobs=[dict(config=asdict(config),seed=s,condition=c,output=str(root),base=str(base),
            steps=args.steps,episodes=args.episodes,validation=args.validation,source_hash=fingerprint)
            for s in range(args.seeds) for c in identity['conditions']]
        rows=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_job,j) for j in jobs]
            for future in as_completed(futures):
                rows.extend(future.result());_csv(root/'episodes.csv',pd.DataFrame(rows))
                print('Completed learner job;',len(rows),'rows',flush=True)
        rules=json.loads((base/'selected_rules.json').read_text())
        for policy,parameters in rules.items():
            for scenario in range(TEST_START,TEST_START+args.episodes):
                metrics,_=rollout(config,policy,scenario,parameters)
                rows.append(dict(policy=policy,training_seed=-1,scenario_seed=scenario,**metrics))
        for s in range(args.seeds):
            directory=base/'models'/f'adaptive_s{s}'
            imitation=MaskablePPO.load(directory/'imitation.zip',device='cpu')
            legacy=load_model(directory,InspectionEnv(config))
            for policy,controller in [('imitation',imitation),('legacy_adaptive',legacy)]:
                for scenario in range(TEST_START,TEST_START+args.episodes):
                    metrics,_=rollout(config,controller,scenario)
                    rows.append(dict(policy=policy,training_seed=s,scenario_seed=scenario,**metrics))
        frame=pd.DataFrame(rows)
        for seed in range(args.seeds):
            initials=[json.loads((root/'models'/f'{c}_s{seed}'/'metadata.json').read_text())['initial_validation_cost']
                      for c in identity['conditions']]
            assert initials[0]==initials[1], 'Initial actors must have identical deterministic validation costs'
        expected={(c,s,t) for c in ['actor_only','actor_critic','imitation','legacy_adaptive'] for s in range(args.seeds) for t in range(TEST_START,TEST_START+args.episodes)}
        expected|={(c,-1,t) for c in rules for t in range(TEST_START,TEST_START+args.episodes)}
        actual=list(zip(frame.policy,frame.training_seed,frame.scenario_seed))
        assert len(actual)==len(expected) and set(actual)==expected
        assert frame.invalid_actions.eq(0).all()
        for row in rows:
            assert np.isclose(json.loads(row['ledger_daily'])[-1]['full'],row['full_objective'])
            assert np.isclose(sum(x['demand_cost'] for x in json.loads(row['ledger_billing'])),row['demand_cost'])
        _csv(root/'episodes.csv',frame)
        summary=frame.groupby('policy')[['full_objective','completion_rate','maintenance_count','inspection_count','breakdowns','belief_coverage90']].mean().reset_index()
        _csv(root/'summary.csv',summary)
        comparisons=[paired_statistics(frame,'actor_critic',reference) for reference in ['actor_only','imitation','legacy_adaptive','spt','belief','uncertainty']]
        _csv(root/'paired_comparisons.csv',pd.DataFrame(comparisons))
        lines=['# Controller calibration experiment','',
            f'Separate exploratory protocol. Three-day accelerated billing; fresh test scenarios starting {TEST_START}. No real factory data.',
            'Same imitation actor, fixed positive reward scaling 1/days, gamma=1, controlled PPO updates. Actor-critic adds full-episode on-policy value pretraining; actor-only omits it.',
            'Pretraining adds simulation/computation and is not a matched total-compute comparison. Calibration and PPO hyperparameters are adaptations, not a new PPO algorithm.',
            'A zero selected PPO step means the imitation actor was retained. Interpret cost with completion. Neither a lower point estimate nor a crossing interval proves superiority.',
            '',_table(summary),'','## Paired cost differences (candidate minus reference; lower is better)','',_table(pd.DataFrame(comparisons)),'','## Checkpoints','']
        for p in sorted((root/'models').glob('*/metadata.json')):
            m=json.loads(p.read_text());lines.append(f"- {p.parent.name}: selected PPO step {m['selected_steps']}; critic {m.get('critic','not pretrained')}")
        (root/'report.md').write_text('\n\n'.join(lines),encoding='utf-8')
        _json(manifest,dict(status='complete',identity=identity,rows=len(frame),invalid_actions=0,ledgers='reconciled'))
        print('COMPLETE',root,len(frame),'rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='outputs/inspection_research/final_pilot')
    p.add_argument('--output',default='outputs/inspection_research/critic_refinement')
    p.add_argument('--steps',type=int,default=65536)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--validation',type=int,default=5)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if min(a.steps,a.seeds,a.episodes,a.validation,a.workers)<1 or a.seeds>5 or a.episodes>500 or a.validation>480:
        p.error('Positive budgets required; max five existing seeds, 500 fresh tests, 480 validation sequences')
    run(a)
