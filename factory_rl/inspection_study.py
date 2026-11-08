"""Reproducible inspection controllers, experiments and model-based action audits."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .inspection import InspectionConfig, InspectionEnv
from .instances import generate_jobs


def source_fingerprint():
    files = ('inspection.py','belief.py','inspection_study.py','instances.py','config.py','agents.py')
    return hashlib.sha256(b''.join((Path(__file__).parent/f).read_bytes() for f in files)).hexdigest()


def public_action(state, config, kind='belief', threshold=0.55, interval=120, uncertainty=0.05):
    """A rule consumes only the permitted public state, never an environment."""
    mask, candidates = state['action_mask'], state['candidates']
    pm, inspect, wait = config.shortlist, config.shortlist+1, config.shortlist+2
    if state['machine'] < 0:
        return wait
    machine = state['machines'][state['machine']]
    if kind == 'periodic' and mask[inspect] and (not machine['measured'] or machine['measurement_age'] >= interval):
        return inspect
    if kind == 'uncertainty' and mask[inspect] and machine['std'] > uncertainty:
        return inspect
    if kind != 'reactive' and mask[pm] and machine['mean'] < threshold:
        return pm
    if candidates:
        if kind == 'spt':
            return int(np.argmin([j['duration']+j['setup'] for j in candidates]))
        return 0  # Candidate list is EDD ordered.
    return wait


def select_action(env, controller='belief', parameters=None):
    if isinstance(controller, str):
        return public_action(env.policy_state(), env.config, controller, **(parameters or {}))
    return int(controller.predict(env._observation(), action_masks=env.action_masks(), deterministic=True)[0])


def rollout(config, controller='belief', scenario=20000, parameters=None,
            exact_health=False, inspections_enabled=True, record=False):
    env = InspectionEnv(config, record=record, exact_health=exact_health,
                        inspections_enabled=inspections_enabled, strict_actions=True)
    env.reset(seed=int(scenario))
    started = time.perf_counter()
    while not env.terminated:
        env.step(select_action(env, controller, parameters))
    metrics=env.metrics()
    for key in ('ledger_daily','ledger_billing'):
        metrics[key]=json.dumps(metrics[key])
    return {**metrics, 'wall_seconds':time.perf_counter()-started}, env


def warm_start_model(model, config, seed=0, episodes=4, epochs=16, exact=False, inspections=True):
    """Imitate a disclosed public rule using training scenarios only; not a new RL algorithm."""
    import torch
    torch.set_num_threads(1)
    obs,masks,actions=[],[],[]
    for j in range(episodes):
        env=InspectionEnv(config,exact_health=exact,inspections_enabled=inspections,strict_actions=True)
        env.reset(seed=(seed*17+j)%1000)
        while not env.terminated:
            action=public_action(env.policy_state(),config,'uncertainty' if inspections and not exact else 'belief')
            obs.append(env._observation()); masks.append(env.action_masks()); actions.append(action)
            env.step(action)
    inputs=torch.as_tensor(np.asarray(obs),dtype=torch.float32)
    legal=np.asarray(masks)
    targets=torch.as_tensor(actions,dtype=torch.long)
    rng=np.random.default_rng(seed+909)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    counts=np.bincount(actions,minlength=config.shortlist+3)
    weights=torch.as_tensor(1/np.sqrt(np.maximum(1,counts)),dtype=torch.float32)
    epochs=max(epochs,math.ceil(400/max(1,math.ceil(len(inputs)/256))))
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(len(inputs)),max(1,math.ceil(len(inputs)/256))):
            distribution=model.policy.get_distribution(inputs[indices],action_masks=legal[indices])
            sample_weights=weights[targets[indices]]
            loss=-(distribution.log_prob(targets[indices])*sample_weights).sum()/sample_weights.sum()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.0)
            optimizer.step()
    with torch.no_grad():
        prediction=model.policy.get_distribution(inputs,action_masks=legal).get_actions(deterministic=True)
    recalls={str(action):float((prediction[targets==action]==action).float().mean()) for action in np.unique(actions)}
    return dict(examples=len(inputs),epochs=epochs,training_agreement=float((prediction==targets).float().mean()),
                training_action_recall=recalls,class_weighting='inverse square root of training action frequency',
                teacher='uncertainty_0.05_threshold_0.55' if inspections and not exact else 'belief_threshold_0.55',
                training_scenario_seeds=[(seed*17+j)%1000 for j in range(episodes)])


def _config_fingerprint(config):
    return hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode()).hexdigest()


def load_model(directory, env, allow_shift=False):
    from sb3_contrib import MaskablePPO
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata.get('schema_version') != env.schema_version or metadata.get('observation_schema') != env.observation_schema:
        raise ValueError('Model observation schema differs from inspection environment')
    if not allow_shift and metadata.get('config_hash') != _config_fingerprint(env.config):
        raise ValueError('Model configuration differs; explicitly opt into a distribution-shift evaluation')
    if metadata.get('exact_health') != env.exact_health or metadata.get('inspections_enabled') != env.inspections_enabled:
        raise ValueError('Model information condition differs from evaluation environment')
    if metadata.get('model_sha256') and hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Model checkpoint checksum differs from evaluated metadata')
    return MaskablePPO.load(directory/'model.zip',device='cpu')


def train_model(config, directory, condition='adaptive', seed=0, steps=16384, validation_episodes=2):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from .agents import ScenarioPool, TRAIN_SEEDS
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    exact, inspections = condition=='exact_health', condition!='no_inspection'
    env = InspectionEnv(config,exact_health=exact,inspections_enabled=inspections,strict_actions=True)
    metadata_path = directory/'metadata.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get('training_seed')!=seed or old.get('condition')!=condition:
            raise ValueError('Saved model training identity differs; use a new directory')
        if old.get('status')=='complete' and old.get('requested_steps')==steps:
            return load_model(directory,env),old
    torch.set_num_threads(1)
    history, validation = [],[]

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = math.inf
            self.best_steps = 0

        def validate(self):
            costs = [rollout(config,self.model,10000+j,exact_health=exact,
                             inspections_enabled=inspections)[0]['full_objective']
                     for j in range(validation_episodes)]
            mean = float(np.mean(costs))
            validation.append(dict(step=int(self.model.num_timesteps),mean_cost=mean))
            if mean < self.best:
                self.best,self.best_steps = mean,int(self.model.num_timesteps)
                self.model.save(directory/'model.zip')
            (directory/'validation.json').write_text(json.dumps(validation,indent=2))

        def _on_step(self):
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done:
                    history.append(dict(step=int(self.num_timesteps),**info['metrics']))
            if self.n_calls % 8192 == 0:
                self.validate()
                pd.DataFrame(history).to_csv(directory/'training.csv',index=False)
            return True

    wrapper = ScenarioPool(env,TRAIN_SEEDS,seed)
    model = MaskablePPO('MlpPolicy',wrapper,seed=seed,device='cpu',n_steps=256,
                       batch_size=64,n_epochs=4,gamma=1.0,ent_coef=0.01,
                       policy_kwargs={'net_arch':[64,64]},verbose=0)
    progress = Progress()
    started = time.perf_counter()
    imitation=warm_start_model(model,config,seed,exact=exact,inspections=inspections)
    if condition=='adaptive':
        model.save(directory/'imitation.zip')
    progress.model=model
    progress.validate()  # Preserve the feasible warm start if later PPO updates collapse.
    metadata = dict(status='running',condition=condition,training_seed=seed,
                    requested_steps=steps,config=asdict(config),config_hash=_config_fingerprint(config),
                    schema_version=env.schema_version,observation_schema=env.observation_schema,
                    exact_health=exact,inspections_enabled=env.inspections_enabled,
                    discount=1.0,horizon_is_terminal=True,imitation=imitation,
                    source_hash=source_fingerprint())
    metadata_path.write_text(json.dumps(metadata,indent=2))
    model.learn(total_timesteps=steps,callback=progress)
    progress.validate()
    pd.DataFrame(history).to_csv(directory/'training.csv',index=False)
    metadata.update(status='complete',actual_steps=int(model.num_timesteps),
                    selected_steps=progress.best_steps,validation_cost=progress.best,
                    training_seconds=time.perf_counter()-started,
                    model_sha256=hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(metadata,indent=2))
    wrapper.close()
    return load_model(directory,env),metadata


def paired_statistics(frame, candidate, reference, draws=2000):
    a = frame[frame.policy==candidate][['training_seed','scenario_seed','full_objective']]
    b = frame[frame.policy==reference][['training_seed','scenario_seed','full_objective']]
    shared_seeds = set(a.training_seed)==set(b.training_seed)
    if shared_seeds:
        paired = a.merge(b,on=['training_seed','scenario_seed'],suffixes=('_a','_b'),validate='one_to_one')
    elif set(b.training_seed)=={-1}:
        b = b.groupby('scenario_seed',as_index=False).full_objective.mean()
        paired = a.merge(b,on='scenario_seed',suffixes=('_a','_b'),validate='many_to_one')
    else:
        raise ValueError('Learned-policy seed grids differ')
    if len(paired) != len(a) or a.empty:
        raise ValueError('Incomplete paired evaluation grid')
    paired['difference'] = paired.full_objective_a-paired.full_objective_b
    matrix = paired.pivot(index='training_seed',columns='scenario_seed',values='difference').to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError('Hierarchical bootstrap requires a complete finite grid')
    rng = np.random.default_rng(702)
    boot = np.array([matrix[rng.integers(matrix.shape[0],size=matrix.shape[0])][:,
                     rng.integers(matrix.shape[1],size=matrix.shape[1])].mean() for _ in range(draws)])
    return dict(candidate=candidate,reference=reference,mean_difference=float(matrix.mean()),
                ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
                training_seeds=matrix.shape[0],scenarios=matrix.shape[1],
                uncertainty='hierarchical training-seed and paired-scenario bootstrap')


def compare_actions(env, actions=None, worlds=16, horizon=120, seed=0, privileged=False):
    """Estimate truncated incremental cost, not total future optimal cost or causal truth.

    Belief-model mode samples hidden worlds independently of the realised truth.
    The continuation is the public belief rule. No policy is allowed future data.
    """
    actions = list(actions if actions is not None else np.flatnonzero(env.action_masks()))
    if worlds < 1 or horizon < 1 or any(not env.action_masks()[a] for a in actions):
        raise ValueError('Positive rollout budget and legal first actions are required')
    if privileged:
        worlds = 1  # A realised hindsight path is not a sample of possible worlds.
    rows = []
    for action in actions:
        costs = []
        for world in range(worlds):
            branch = env.snapshot()
            branch.record = False
            rng = np.random.default_rng(np.random.SeedSequence([seed,world,601]))
            if not privileged:
                branch.exact_health=False
                _sample_future_orders(branch,rng)
                for m,belief in enumerate(branch.beliefs):
                    index = int(rng.choice(len(belief.health),p=belief.weights))
                    branch.machines[m].health = float(belief.health[index])
                    branch.true_rates[m] = float(belief.rate[index])
                shape = branch.failure_shocks.shape
                branch.failure_shocks = rng.random(shape)
                cv = branch.config.wear_noise_cv
                branch.wear_shocks = rng.gamma(1/cv**2,cv**2,shape) if cv else np.ones(shape)
                branch.sensor_shocks = rng.normal(size=shape)
            branch.rollout_limit = min(branch.config.horizon,branch.time+horizon)
            before = branch.cost_components()['full']
            branch.step(int(action))
            while not branch.terminated and branch.time < branch.rollout_limit:
                branch.step(public_action(branch.policy_state(),branch.config,'belief'))
            costs.append(branch.cost_components()['full']-before)
        array = np.asarray(costs)
        se = float(array.std(ddof=1)/np.sqrt(worlds)) if worlds>1 else 0.
        rows.append(dict(action=int(action),mean_incremental_cost=float(array.mean()),
                         ci_low=float(array.mean()-1.96*se),ci_high=float(array.mean()+1.96*se),
                         worlds=worlds,horizon_calendar_minutes=horizon,
                         information_mode='privileged_hindsight' if privileged else 'belief_model',
                         continuation='belief_threshold_0.55',
                         cost_samples=json.dumps(array.tolist()),
                         limitation='truncated realised-cost estimate; no residual-value approximation'))
    return pd.DataFrame(rows)


def _sample_future_orders(branch,rng):
    """Conditional synthetic arrival forecast. Never read unreleased order attributes.

    Daily counts and the arrival law are public. Future operation attributes use
    the training duration-pool prior; this is a documented forecast approximation
    under the held-out benchmark-source distribution shift.
    """
    c=branch.config
    day_now=branch.time//1440
    for day in range(day_now,c.days):
        indices=[j for j in range(day*c.jobs,(day+1)*c.jobs) if branch.arrivals[j]>branch.time]
        if not indices:
            continue
        while True:
            daily=generate_jobs(c,rng,int(rng.integers(1000)))
            if daily[-1].arrival < c.shift_minutes:
                break
        if day==day_now:
            for _ in range(10000):
                arrivals=branch.time + np.ceil(np.cumsum(rng.exponential(c.arrival_minutes/max(1,c.jobs-1),len(indices)))).astype(int)
                if arrivals[-1] < day*1440+c.shift_minutes:
                    break
            else:
                raise ValueError('Conditional future-arrival forecast cannot fit remaining shift')
        else:
            arrivals=np.array([j.arrival+day*1440 for j in daily])
        for slot,index in enumerate(indices):
            job=daily[slot]
            allowance=job.due-job.arrival
            job.arrival=int(arrivals[slot]); job.due=job.arrival+allowance
            branch.jobs[index]=job
            branch.arrivals[index]=job.arrival; branch.dues[index]=job.due


def tune_rules(config, directory, episodes=3):
    directory = Path(directory)
    path = directory/'rule_validation.csv'
    identity=dict(config_hash=_config_fingerprint(config),episodes=episodes,source_hash=source_fingerprint())
    identity_path=directory/'rule_validation_identity.json'
    if path.exists() and (not identity_path.exists() or json.loads(identity_path.read_text())!=identity):
        raise ValueError('Rule-validation cache identity differs; use a new output stage')
    identity_path.write_text(json.dumps(identity,indent=2))
    candidates = [('reactive',{})]
    candidates += [('belief',{'threshold':t}) for t in (0.4,0.55,0.7)]
    candidates += [('spt',{'threshold':t}) for t in (0.4,0.55,0.7)]
    candidates += [('periodic',{'threshold':0.55,'interval':i}) for i in (60,120,240)]
    candidates += [('uncertainty',{'threshold':0.55,'uncertainty':u}) for u in (.02,.05,.08)]
    records = []
    if path.exists():
        records = pd.read_csv(path).to_dict('records')
    for kind,parameters in candidates:
        key = kind+json.dumps(parameters,sort_keys=True)
        if any(r['key']==key for r in records):
            continue
        costs = [rollout(config,kind,10000+j,parameters)[0]['full_objective'] for j in range(episodes)]
        records.append(dict(key=key,policy=kind,parameters=json.dumps(parameters,sort_keys=True),
                            mean_cost=float(np.mean(costs)),validation_episodes=episodes))
        pd.DataFrame(records).to_csv(path,index=False)
    frame = pd.DataFrame(records)
    selected = {}
    for kind in frame.policy.unique():
        row = frame[frame.policy==kind].sort_values('mean_cost').iloc[0]
        selected[kind] = json.loads(row.parameters)
    (directory/'selected_rules.json').write_text(json.dumps(selected,indent=2))
    return selected


def summarise(frame):
    fields = ['full_objective','completion_rate','on_time_rate','average_tardiness_min','breakdowns',
              'maintenance_count','inspection_count','energy_cost','demand_cost','belief_health_mae',
              'belief_coverage90','wall_seconds']
    summary = frame.groupby('policy')[fields].mean()
    summary['scenario_sequences'] = frame.groupby('policy').scenario_seed.nunique()
    summary['training_seeds'] = frame.groupby('policy').training_seed.apply(lambda s:int(s[s>=0].nunique()))
    summary['worst10_cost'] = frame.groupby('policy').full_objective.apply(
        lambda s: float(s.sort_values().iloc[-max(1,math.ceil(len(s)*.1)):].mean()))
    return summary.reset_index()


def validate_grid(frame,seeds,episodes):
    scenarios=range(20000,20000+episodes)
    expected={(policy,-1,j) for policy in ('reactive','belief','spt','periodic','uncertainty') for j in scenarios}
    expected.update((policy,s,j) for policy in ('adaptive','no_inspection','exact_health','imitation')
                    for s in range(seeds) for j in scenarios)
    actual=list(frame[['policy','training_seed','scenario_seed']].itertuples(index=False,name=None))
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Evaluation grid has duplicate, missing or undeclared rows')


def evaluation_job(job):
    """Independent resumable unit; safe to execute in a separate CPU process."""
    config = InspectionConfig(**job['config'])
    directory = Path(job['directory'])
    directory.mkdir(parents=True,exist_ok=True)
    settings = {k:v for k,v in job.items() if k not in ('episodes','directory')}
    settings['source_hash']=source_fingerprint()
    settings_path = directory/'evaluation_settings.json'
    condition = job['condition']
    exact, inspections = condition=='exact_health', condition!='no_inspection'
    if condition:
        controller,_ = train_model(config,directory,condition,job['seed'],job['steps'])
        settings['model_hash']=hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest()
        if condition=='adaptive':
            settings['imitation_hash']=hashlib.sha256((directory/'imitation.zip').read_bytes()).hexdigest()
    else:
        controller = job['label']
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Evaluation cache settings differ')
    settings_path.write_text(json.dumps(settings,indent=2))
    path = directory/'test_episodes.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    finished = {(r['policy'],r['scenario_seed']) for r in rows}
    for scenario in range(20000,20000+job['episodes']):
        if (job['label'],scenario) in finished:
            continue
        metrics,_ = rollout(config,controller,scenario,job['parameters'],exact,inspections)
        rows.append(dict(policy=job['label'],training_seed=job['seed'],scenario_seed=scenario,**metrics))
        pd.DataFrame(rows).to_csv(path,index=False)
    if condition=='adaptive':
        from sb3_contrib import MaskablePPO
        imitation=MaskablePPO.load(directory/'imitation.zip',device='cpu')
        for scenario in range(20000,20000+job['episodes']):
            if ('imitation',scenario) in finished:
                continue
            metrics,_=rollout(config,imitation,scenario)
            rows.append(dict(policy='imitation',training_seed=job['seed'],scenario_seed=scenario,**metrics))
            pd.DataFrame(rows).to_csv(path,index=False)
    return pd.read_csv(path).query('scenario_seed < @job["episodes"] + 20000').to_dict('records')


@contextmanager
def _run_lock(output):
    """An OS lock releases automatically on crash; stale files do not block resume."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    handle = (output/'run.lock').open('a+b')
    handle.seek(0,2)
    if handle.tell()==0:
        handle.write(b' ')
        handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('An inspection study is already running in this output directory')
    try:
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps(dict(pid=os.getpid())).encode())
        handle.flush()
        yield
    finally:
        handle.seek(0)
        if os.name=='nt':
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(handle,fcntl.LOCK_UN)
        handle.close()


def run_study(config, output, steps=16384, seeds=2, episodes=20, stage='pilot', workers=1):
    with _run_lock(output):
        return _run_study(config,output,steps,seeds,episodes,stage,workers)


def _run_study(config, output, steps, seeds, episodes, stage, workers):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config_hash = _config_fingerprint(config)
    manifest_path = output/'run.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        identity=dict(config_hash=config_hash,requested_steps=steps,training_seeds=list(range(seeds)),source_hash=source_fingerprint())
        if any(old.get(key)!=value for key,value in identity.items()):
            raise ValueError('Existing run identity differs; preserve it and use another named stage directory')
    manifest = dict(status='running',stage=stage,config=asdict(config),config_hash=config_hash,
                    requested_steps=steps,training_seeds=list(range(seeds)),test_scenarios=list(range(20000,20000+episodes)),
                    source_hash=source_fingerprint(),
                    provenance='Synthetic multi-shift scenarios, Taillard-derived processing durations; no factory calibration',
                    versions={n:importlib.metadata.version(n) for n in ('numpy','gymnasium','torch','sb3-contrib')})
    manifest_path.write_text(json.dumps(manifest,indent=2))
    selected = tune_rules(config,output,episodes=min(3,episodes))
    print('Validation-selected rules ready',flush=True)
    path = output/'episodes.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    rows=[r for r in rows if r['scenario_seed'] in manifest['test_scenarios'] and
          (r['training_seed']==-1 or r['training_seed'] in manifest['training_seeds'])]

    def evaluate(label,controller,training_seed=-1,parameters=None,exact=False,inspections=True):
        if isinstance(controller,str):
            checkpoint_hash=hashlib.sha256(json.dumps([controller,parameters],sort_keys=True).encode()).hexdigest()
        else:
            checkpoint_hash=hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes()
                for _,value in sorted(controller.policy.state_dict().items()))).hexdigest()
        matching=[r for r in rows if r['policy']==label and r['training_seed']==training_seed]
        if any(r.get('checkpoint_hash')!=checkpoint_hash for r in matching):
            raise ValueError('Cached evaluations belong to a different controller/checkpoint')
        for scenario in range(20000,20000+episodes):
            if any(r['policy']==label and r['training_seed']==training_seed and r['scenario_seed']==scenario for r in rows):
                continue
            metrics,_ = rollout(config,controller,scenario,parameters,exact,inspections)
            rows.append(dict(policy=label,training_seed=training_seed,scenario_seed=scenario,
                             checkpoint_hash=checkpoint_hash,**metrics))
            pd.DataFrame(rows).to_csv(path,index=False)
        print(f'Evaluated {label}, training seed {training_seed}',flush=True)

    if workers==1:
        for kind,parameters in selected.items():
            evaluate(kind,kind,parameters=parameters)
        for seed in range(seeds):
            for condition in ('adaptive','no_inspection','exact_health'):
                print(f'Training {condition}, seed {seed}, {steps} steps',flush=True)
                model,_ = train_model(config,output/'models'/f'{condition}_s{seed}',condition,seed,steps)
                evaluate(condition,model,seed,exact=condition=='exact_health',inspections=condition!='no_inspection')
                if condition=='adaptive':
                    from sb3_contrib import MaskablePPO
                    imitation=MaskablePPO.load(output/'models'/f'{condition}_s{seed}'/'imitation.zip',device='cpu')
                    evaluate('imitation',imitation,seed)
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        jobs = [dict(config=asdict(config),directory=str(output/'rules'/kind),label=kind,
                     parameters=parameters,condition=None,seed=-1,steps=0,episodes=episodes)
                for kind,parameters in selected.items()]
        jobs += [dict(config=asdict(config),directory=str(output/'models'/f'{condition}_s{seed}'),
                      label=condition,parameters={},condition=condition,seed=seed,steps=steps,episodes=episodes)
                 for seed in range(seeds) for condition in ('adaptive','no_inspection','exact_health')]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluation_job,job):job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(path,index=False)
                print(f"Finished {job['label']}, seed {job['seed']}: {len(rows)} aggregate rows",flush=True)
    frame = pd.DataFrame(rows)
    validate_grid(frame,seeds,episodes)
    summary = summarise(frame)
    summary.to_csv(output/'summary.csv',index=False)
    references=['reactive','belief','periodic','uncertainty','no_inspection']
    if 'imitation' in frame.policy.unique():
        references.append('imitation')
    comparisons = [paired_statistics(frame,'adaptive',reference) for reference in references]
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv',index=False)
    manifest.update(status='complete',rows=len(frame),completed_models=seeds*3,
                    evidence_level='accelerated cycle / pilot' if stage!='research' else 'declared research configuration')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    write_report(output)
    return frame


def write_report(output):
    output = Path(output)
    manifest = json.loads((output/'run.json').read_text())
    frame = pd.read_csv(output/'episodes.csv')
    summary = summarise(frame)
    display = summary[['policy','full_objective','completion_rate','inspection_count','maintenance_count','breakdowns','belief_coverage90']]
    lines = ['# Adaptive inspection research results','',f"Evidence stage: **{manifest['stage']}**.",'',
             f"{manifest['config']['days']} simulated days; billing period {manifest['config']['billing_days']} days; "
             f"{len(manifest['training_seeds'])} training seeds; {len(manifest['test_scenarios'])} held-out scenario sequences.",'',
             'All jobs, health, failures and prices are simulated. Processing durations are benchmark-derived. '
             'The combined objective is normalised and is not an electricity bill or accuracy percentage.','',
             '|Controller|Mean cost|Completion|Inspections|PM actions|Breakdowns|90% interval coverage|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in display.to_dict('records'):
        lines.append(f"|{row['policy']}|{row['full_objective']:.3f}|{row['completion_rate']:.1%}|{row['inspection_count']:.2f}|"
                     f"{row['maintenance_count']:.2f}|{row['breakdowns']:.2f}|{row['belief_coverage90']:.1%}|")
    lines += ['', '## Interpretation and limitations','',
              'Adaptive inspection is a hypothesis, not an assumed improvement. Read cost together with completion, '
              'breakdowns and inspection spending. The exact-health policy has privileged information and is not an optimal bound.',
              'The paired-comparison file includes uncertainty over both training seeds and matched scenarios. '
              'Small seed/scenario counts and short training are preliminary evidence.',
              'Condition intervals are approximate particle-filter intervals; empirical coverage must be assessed. '
              'Poor coverage is an inference limitation, not proof that the policy knows the true health.',
              'This is a finite-horizon study. Long-run factory performance and real-world savings are not established.',
              'Candidate contribution: the interaction of paid condition information, job dispatch and persistent billing history. '
              'Individual inspection, maintenance, PPO and electricity-scheduling methods already exist.',
              '', '## Neural training and checkpoint selection','',
              'The current neural protocol uses a public-rule imitation warm start from training scenarios only, '
              'with inverse-square-root action-frequency weighting. PPO then trains on the same realised-cost objective. '
              'The imitation-only policy is evaluated separately. Validation may select the checkpoint before any PPO updates; '
              'a selected step of zero means imitation was retained, not that PPO improved it.',
              '', '|Condition / seed|Selected PPO step|Teacher|Imitation training agreement|',
              '|---|---:|---|---:|']
    for path in sorted((output/'models').glob('*/metadata.json')):
        model=json.loads(path.read_text())
        imitation=model.get('imitation',{})
        lines.append(f"|{path.parent.name}|{model.get('selected_steps','unrecorded')}|"
                     f"{imitation.get('teacher','no imitation recorded')}|{imitation.get('training_agreement',0):.1%}|")
    comparison_path=output/'paired_comparisons.csv'
    if comparison_path.exists():
        paired=pd.read_csv(comparison_path)
        lines += ['', '## Paired cost differences','',
                  'Negative values favour the candidate. These are uncertainty intervals, not a guarantee of future performance.',
                  '', '|Candidate minus reference|Mean difference|95% interval|', '|---|---:|---|']
        for row in paired.to_dict('records'):
            lines.append(f"|{row['candidate']} minus {row['reference']}|{row['mean_difference']:.3f}|"
                         f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]|")
    lines += [
              '', '## Research foundations','',
              '- [Inspection and maintenance under erroneous observations, 2024](https://link.springer.com/article/10.1186/s43065-024-00098-9)',
              '- [Maintenance and value of information, 2025](https://doi.org/10.1016/j.array.2025.100454)',
              '- [Energy-aware scheduling with preventive maintenance, 2026](https://www.sciencedirect.com/science/article/pii/S0957417425046391)',
              '- [Tariff-aware manufacturing load shifting, 2026](https://www.mdpi.com/1424-8220/26/6/1858)',
              '', '## Files','', 'See summary.csv, episodes.csv, paired_comparisons.csv, selected_rules.json and run.json.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].barh(summary.policy,summary.full_objective,color='#2b6f8e')
    axes[0].set_xlabel('Normalised combined cost (lower is better)')
    axes[1].barh(summary.policy,summary.completion_rate*100,color='#409b83')
    axes[1].set_xlabel('Jobs completed (%)')
    fig.suptitle(f"{manifest['stage']}: {manifest['config']['days']} simulated days, {len(manifest['training_seeds'])} training seeds")
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)


def add_arguments(parser):
    parser.add_argument('--config')
    parser.add_argument('--output',default='outputs/inspection_research/pilot')
    parser.add_argument('--steps',type=int,default=16384)
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--days',type=int)
    parser.add_argument('--billing-days',type=int)
    parser.add_argument('--particles',type=int)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--stage',choices=['smoke','pilot','research','report'],default='pilot')


def run_cli(args):
    if args.stage=='report':
        write_report(args.output)
        return
    if not 1 <= args.episodes <= 500 or args.steps<1 or args.seeds<1 or not 1 <= args.workers <= 4:
        raise ValueError('Positive steps/seeds and 1..500 test scenarios are required')
    config = InspectionConfig.load(args.config)
    changes = {key:getattr(args,key) for key in ('days','billing_days','particles') if getattr(args,key) is not None}
    config = replace(config,**changes)
    run_study(config,args.output,args.steps,args.seeds,args.episodes,args.stage,args.workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run_cli(parser.parse_args())
