"""A separate physical-budget study which groups forced public-gate actions."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from .agents import ScenarioPool, TRAIN_SEEDS
from .inspection import InspectionConfig
from .inspection_study import source_fingerprint, paired_statistics, _run_lock
from .priority_search import _roll, INITIAL, _json, _csv, _table
from .residual_dispatch import (DispatchEnv, ResidualPolicy, UpdateGPUOnlyPPO,
    evaluation_policy, _roll_dispatch, validate_rows, bind_identity)


class DecisionEnv(gym.Wrapper):
    """Reset stays at time zero; every subsequent forced action/cost is retained."""
    def __init__(self, config):
        super().__init__(DispatchEnv(config))
        self.physical_steps = self.choice_steps = self.episodes = 0

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        reward = 0.
        actions = []
        self.choice_steps += int(self.action_masks().sum() > 1)
        while True:
            actions.append(int(action))
            obs, r, terminal, truncated, info = self.env.step(action)
            reward += r
            self.physical_steps += 1
            if terminal or truncated:
                self.episodes += 1
                break
            mask = self.action_masks()
            if mask.sum() > 1:
                break
            action = int(np.flatnonzero(mask)[0])
        return obs, reward, terminal, truncated, {**info, 'aggregated_physical_actions': actions}


def _train(config, root, condition, seed, args):
    directory = root/'models'/f'{condition}_s{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory/'metadata.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['status'] == 'complete':
            assert hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest() == meta['model_sha256']
            return MaskablePPO.load(directory/'model.zip', device='cpu'), meta
    env = DummyVecEnv([lambda i=i: ScenarioPool(DecisionEnv(config), TRAIN_SEEDS, seed*100+i) for i in range(4)])
    model = UpdateGPUOnlyPPO(ResidualPolicy, env, seed=seed, device='cuda', learning_rate=1e-4,
        n_steps=128, batch_size=256, n_epochs=4, gamma=1., gae_lambda=.98,
        ent_coef=0., clip_range=.1, target_kl=.015,
        policy_kwargs=dict(shortlist=config.shortlist, use_prior=condition == 'event_residual'), verbose=0)
    meta = dict(status='training', condition=condition, seed=seed, config=asdict(config),
        device='CPU collection and CUDA updates', physical_budget=args.physical_steps,
        method='forced-action aggregation; fixed public service; shared job scorer; no learned service')
    _json(meta_path, meta)
    started = time.perf_counter()

    def counts():
        return {key:int(sum(env.get_attr(key))) for key in ['physical_steps','choice_steps','episodes']}

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history, self.best, self.selected = [], float('inf'), 0
            self.next_validation, self.stop = 8192, False
            self.completed_rollout_physical = 0

        def validate(self):
            actor = evaluation_policy(model.policy)
            rows = [_roll_dispatch(config, actor, args.validation_start+j) for j in range(args.validation)]
            cost = float(np.mean([r['full_objective'] for r in rows]))
            row = dict(outer_steps=int(model.num_timesteps), **counts(), mean_cost=cost,
                       changed_dispatches=float(np.mean([r['changed_dispatches'] for r in rows])))
            self.history.append(row)
            if cost < self.best:
                self.best, self.selected = cost, int(model.num_timesteps)
                model.save(directory/'model.zip')
            _json(directory/'validation.json', self.history)
            print(f'{condition} seed{seed} outer{model.num_timesteps} physical{row["physical_steps"]} validation{cost:.4f}', flush=True)

        def _on_rollout_end(self):
            physical = counts()['physical_steps']
            self.completed_rollout_physical = physical
            if physical >= self.next_validation:
                self.validate()
                self.next_validation = (physical//8192+1)*8192
            if physical >= args.physical_steps:
                self.stop = True

        def _on_step(self):
            return not self.stop

    progress = Progress()
    progress.validate()
    # The callback stops after the first completed rollout reaching the physical
    # budget. SB3 then probes one vector step before stopping; record that too.
    model.learn(args.physical_steps+4096, callback=progress)
    progress.validate()
    meta.update(status='complete', selected_steps=progress.selected, validation_cost=progress.best,
        outer_steps=int(model.num_timesteps), **counts(),
        trained_rollout_physical_steps=progress.completed_rollout_physical,
        physical_overshoot=progress.completed_rollout_physical-args.physical_steps,
        gpu_update_calls=model.gpu_update_calls,
        training_seconds=time.perf_counter()-started,
        model_sha256=hashlib.sha256((directory/'model.zip').read_bytes()).hexdigest())
    assert meta['physical_steps'] >= args.physical_steps and meta['gpu_update_calls'] > 0
    _json(meta_path, meta)
    env.close()
    return MaskablePPO.load(directory/'model.zip', device='cpu'), meta


def run(args):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This declared study requires the verified CUDA Python')
    base = Path('outputs/inspection_research/residual_dispatch')
    main = json.loads((base/'run.json').read_text())
    if main['status'] != 'complete':
        raise RuntimeError('Wait for the frozen reference study to finish; do not duplicate its GPU work')
    config = InspectionConfig(**main['identity']['config'])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    controls = [base/'models'/f'{p}_s{s}'/'model.zip' for s in range(args.seeds) for p in ['residual','no_prior']]
    identity = dict(config=asdict(config), settings=vars(args), base_identity=main['identity'],
        controls_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in controls},
        source_hash=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()
            +Path('factory_rl/residual_dispatch.py').read_text()+Path('factory_rl/priority_search.py').read_text()).encode()).hexdigest(),
        budget='physical transitions; complete-rollout overshoot and final discarded probe are recorded',
        gamma=1., gae_lambda=.98, gae_time='aggregated outer decision steps', torch=torch.__version__)
    with _run_lock(root):
        bind_identity(root, identity)
        _json(root/'run.json', dict(status='running', identity=identity))
        jobs = [(p,s) for s in range(args.seeds) for p in ['event_residual','event_no_prior','plain_residual','plain_no_prior']]
        jobs += [('retuned_setup_spt',-1),('spt',-1)]
        scenarios = set(range(args.test_start,args.test_start+args.episodes))
        all_rows, metadata = [], []
        for policy, seed in jobs:
            if policy.startswith('event_'):
                model, meta = _train(config, root, policy, seed, args)
                metadata.append(meta)
            elif policy.startswith('plain_'):
                model = MaskablePPO.load(base/'models'/f'{policy[6:]}_s{seed}'/'model.zip',device='cpu')
            else:
                model = None
            path = root/f'{policy}_s{seed}.csv'
            rows = pd.read_csv(path).to_dict('records') if path.exists() else []
            validate_rows(rows, policy, seed, scenarios)
            for scenario in sorted(scenarios-{r['scenario_seed'] for r in rows}):
                metrics = _roll(config,INITIAL,scenario)[0] if policy == 'spt' else _roll_dispatch(config,model,scenario)
                rows.append(dict(policy=policy,training_seed=seed,scenario_seed=scenario,**metrics))
                _csv(path,pd.DataFrame(rows))
            all_rows.extend(rows)
            print(f'Finished {policy} seed{seed}: {len(all_rows)} rows',flush=True)
        frame = pd.DataFrame(all_rows)
        actual = list(frame[['policy','training_seed','scenario_seed']].itertuples(index=False,name=None))
        expected = {(p,s,j) for p,s in jobs for j in scenarios}
        assert len(actual) == len(set(actual)) and set(actual) == expected
        _csv(root/'episodes.csv',frame)
        fields = ['full_objective','completion_rate','on_time_rate','inspection_count','maintenance_count','breakdowns','belief_coverage90','changed_dispatches']
        summary = frame.groupby('policy')[fields].mean().reset_index()
        _csv(root/'summary.csv',summary)
        pairs = [('event_residual',p) for p in ['event_no_prior','plain_residual','plain_no_prior','retuned_setup_spt','spt']]
        pairs += [('event_no_prior',p) for p in ['plain_no_prior','retuned_setup_spt','spt']]
        paired = pd.DataFrame([paired_statistics(frame,a,b) for a,b in pairs])
        _csv(root/'paired_comparisons.csv',paired)
        report = ['# Physical-budget event-dispatch study','',
            'Same physical factory and public service gate; only consecutive forced actions are grouped for learning.',
            'All costs remain included; gamma1 preserves the undiscounted objective. GAE lambda now decays per aggregated choice.',
            'Physical-budget targets are matched to the preceding study; recorded rollout overshoots and update counts differ.',
            'This is a complete representation/training-protocol comparison, not an isolated causal claim about one parameter.',
            f'{len(frame)} exact held-out rows. Fresh validation{args.validation_start}+, test{args.test_start}+.', '',
            _table(summary),'',_table(paired),'','## Training and selected checkpoints','',
            _table(pd.DataFrame([{k:m[k] for k in ['condition','seed','selected_steps','physical_steps','trained_rollout_physical_steps','physical_overshoot','choice_steps','outer_steps','episodes','gpu_update_calls']} for m in metadata])), '',
            'Stepzero means the starting policy was retained; report it explicitly. Negative cost differences favour the candidate.',
            'Learned service/inspection value, real-factory calibration, monetary savings and new-algorithm novelty are unestablished.']
        (root/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        _json(root/'run.json',dict(status='complete',identity=identity,rows=len(frame),completed_models=len(metadata)))
        print(summary.to_string(index=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/inspection_research/event_dispatch')
    parser.add_argument('--physical-steps',type=int,default=65536)
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--validation',type=int,default=10)
    parser.add_argument('--validation-start',type=int,default=10220)
    parser.add_argument('--test-start',type=int,default=24700)
    args = parser.parse_args()
    if min(args.physical_steps,args.seeds,args.episodes,args.validation) < 1 or args.seeds > 3:
        parser.error('Positive budgets and one to three seeds required')
    run(args)
