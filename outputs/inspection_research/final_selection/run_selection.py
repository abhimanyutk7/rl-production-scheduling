"""Select fixed saved policies on common validation, then evaluate untouched tests."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from factory_rl.inspection import InspectionConfig
from factory_rl.inspection_study import source_fingerprint, paired_statistics, _run_lock
from factory_rl.priority_search import _roll, INITIAL, _json, _csv, _table
from factory_rl.residual_dispatch import PRIOR, _roll_dispatch, bind_identity

ROOT = Path(__file__).resolve().parent
VALIDATION = list(range(10240,10250))
TEST = list(range(24900,24950))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    parents = {name:PROJECT/'outputs/inspection_research'/name for name in ['residual_dispatch','event_dispatch']}
    deadline = time.monotonic()+7200
    _json(ROOT/'queue.json',dict(status='waiting_for_completed_parent_studies'))
    while any(not (p/'run.json').exists() or json.loads((p/'run.json').read_text())['status'] != 'complete' for p in parents.values()):
        if time.monotonic() > deadline:
            raise RuntimeError('Parent study did not finish within two hours; inspect logs before resuming')
        time.sleep(10)
    _json(ROOT/'queue.json',dict(status='started'))
    manifests = {k:json.loads((p/'run.json').read_text()) for k,p in parents.items()}
    config_dict = manifests['residual_dispatch']['identity']['config']
    assert manifests['event_dispatch']['identity']['config'] == config_dict
    config = InspectionConfig(**config_dict)
    candidates = []
    for family,condition in [('residual_dispatch','no_prior'),('event_dispatch','event_no_prior')]:
        for seed in range(3):
            directory = parents[family]/'models'/f'{condition}_s{seed}'
            meta = json.loads((directory/'metadata.json').read_text())
            assert meta['status'] == 'complete' and sha(directory/'model.zip') == meta['model_sha256']
            candidates.append(dict(name=f'{family}_s{seed}',kind='neural',path=str(directory.relative_to(PROJECT)),
                sha256=sha(directory/'model.zip'),selected_steps=meta['selected_steps']))
    for seed in range(3):
        path = PROJECT/'outputs/inspection_research/priority_search'/f'policy_s{seed}.json'
        candidates.append(dict(name=f'cem_s{seed}',kind='cem',path=str(path.relative_to(PROJECT)),
            sha256=sha(path),weights=json.loads(path.read_text())['weights']))
    candidates += [dict(name='retuned_setup_spt',kind='rule',weights=PRIOR.tolist()),
                   dict(name='spt',kind='rule',weights=INITIAL.tolist())]
    helpers = PROJECT/'outputs/inspection_research/residual_dispatch/verify_results.py'
    identity = dict(config=config_dict,candidates=candidates,validation=VALIDATION,test=TEST,
        parents={k:v['identity'] for k,v in manifests.items()},
        source_hash=hashlib.sha256((source_fingerprint()+Path(__file__).read_text()
            +(PROJECT/'factory_rl/residual_dispatch.py').read_text()+(PROJECT/'factory_rl/event_dispatch.py').read_text()
            +(PROJECT/'factory_rl/priority_search.py').read_text()).encode()).hexdigest(),
        helpers_sha256=sha(helpers),torch=torch.__version__,
        selection='minimum common validation cost within neural/CEM; global validation winner also recorded; test never selects')

    def evaluate(candidate, scenario):
        if candidate['kind'] == 'neural':
            if candidate['name'] not in models:
                models[candidate['name']] = MaskablePPO.load(PROJECT/candidate['path']/'model.zip',device='cpu')
            return _roll_dispatch(config,models[candidate['name']],scenario)
        return _roll(config,np.array(candidate['weights']),scenario)[0]

    with _run_lock(ROOT):
        bind_identity(ROOT,identity)
        _json(ROOT/'run.json',dict(status='running',phase='common_validation',identity=identity))
        models = {}
        path = ROOT/'validation.csv'
        rows = pd.read_csv(path).to_dict('records') if path.exists() else []
        expected = {(c['name'],j) for c in candidates for j in VALIDATION}
        actual = [(r['candidate'],r['scenario_seed']) for r in rows]
        assert len(actual) == len(set(actual)) and set(actual) <= expected
        for candidate in candidates:
            for scenario in VALIDATION:
                if (candidate['name'],scenario) in set(actual):
                    continue
                rows.append(dict(candidate=candidate['name'],scenario_seed=scenario,**evaluate(candidate,scenario)))
                _csv(path,pd.DataFrame(rows))
            print('Validated',candidate['name'],flush=True)
        validation = pd.DataFrame(rows)
        assert len(validation) == len(expected) == 110
        costs = validation.groupby('candidate').full_objective.mean().to_dict()
        best_neural = min([c for c in candidates if c['kind'] == 'neural'],key=lambda c:costs[c['name']])
        best_cem = min([c for c in candidates if c['kind'] == 'cem'],key=lambda c:costs[c['name']])
        overall = min(candidates,key=lambda c:costs[c['name']])
        selection = dict(selected_neural=best_neural,selected_cem=best_cem,best_overall=overall,
            common_validation_costs=costs,validation_scenarios=VALIDATION,
            locked_before_test=True,test_scenarios=TEST,
            note='One selected neural policy is a deployment choice; this is not a multi-training-seed benchmark.')
        locked = ROOT/'selection.json'
        if locked.exists():
            old = json.loads(locked.read_text())
            for field in ['selected_neural','selected_cem','best_overall']:
                assert old[field] == selection[field], 'Never change the locked choice after testing'
            selection = old
        else:
            _json(locked,selection)
        _json(ROOT/'run.json',dict(status='running',phase='locked_test',identity=identity,selection_sha256=sha(locked)))
        controls = [('selected_neural',best_neural),('selected_cem',best_cem)]+[(c['name'],c) for c in candidates if c['kind'] == 'rule']
        path = ROOT/'episodes.csv'
        rows = pd.read_csv(path).to_dict('records') if path.exists() else []
        expected = {(p,-1,j) for p,c in controls for j in TEST}
        actual = [(r['policy'],r['training_seed'],r['scenario_seed']) for r in rows]
        assert len(actual) == len(set(actual)) and set(actual) <= expected
        for label,candidate in controls:
            for scenario in TEST:
                if (label,-1,scenario) in set(actual):
                    continue
                rows.append(dict(policy=label,training_seed=-1,scenario_seed=scenario,
                    fixed_candidate=candidate['name'],**evaluate(candidate,scenario)))
                _csv(path,pd.DataFrame(rows))
            print('Tested fixed',label,flush=True)
        frame = pd.DataFrame(rows)
        assert len(frame) == len(expected) == 200
        accounting = runpy.run_path(str(helpers))['accounting']
        checks = accounting(frame,config_dict)
        validation_checks = accounting(validation,config_dict)
        summary = frame.groupby('policy')[['full_objective','completion_rate','on_time_rate','inspection_count','breakdowns','belief_coverage90']].mean().reset_index()
        paired = pd.DataFrame([paired_statistics(frame,'selected_neural',p) for p in ['selected_cem','retuned_setup_spt','spt']])
        paired['uncertainty'] = 'paired scenario bootstrap for fixed validation-selected policies; no across-training-seed uncertainty'
        paired['training_seeds'] = 0
        _csv(ROOT/'summary.csv',summary)
        _csv(ROOT/'paired_comparisons.csv',paired)
        _json(ROOT/'verification.json',dict(status='passed',exact_grid=True,selection_sha256=sha(locked),
            episodes_sha256=sha(path),validation_rows=len(validation),validation_accounting=validation_checks,**checks))
        report = ['# Locked validation selection and fresh fixed-policy test','',
            '11 candidates on ten common validation sequences10240-10249; choices locked before test24900-24949.',
            f'Selected neural: {best_neural["name"]}. Selected CEM: {best_cem["name"]}. Overall validation winner: {overall["name"]}.',
            'This evaluates single fixed selected policies on50 fresh three-day scenarios, not an algorithm average over training seeds.',
            'All service choices use disclosed public rules; inspection benefit is not tested. Normalised cost is not accuracy or money savings.','',
            _table(summary),'',_table(paired),'',
            'Selection uncertainty and broader training-seed variation are not captured by these fixed-policy scenario intervals.',
            'All earlier studies and negative comparisons remain part of the research evidence. No test result selects the model.']
        (ROOT/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        _json(ROOT/'run.json',dict(status='complete',identity=identity,selection_sha256=sha(locked),rows=len(frame)))
        runpy.run_path(str(helpers))['decision_example'](ROOT,'selected_neural',PROJECT/best_neural['path'])
        _json(ROOT/'queue.json',dict(status='complete'))
        print(summary.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
