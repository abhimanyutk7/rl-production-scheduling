"""Read saved final-test evidence; optionally replay eight neural decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    return ROOT.joinpath(*str(value).replace('\\', '/').split('/'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Load neural policy and run eight CPU decisions')
    args = parser.parse_args()
    study = ROOT / 'outputs/inspection_research/final_selection'
    selection = json.loads((study / 'selection.json').read_text())
    neural = selection['selected_neural']
    checkpoint = project_path(neural['path']) / 'model.zip'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != neural['sha256']:
        raise ValueError('Selected neural checkpoint hash differs from locked selection')
    with (study / 'episodes.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    policies = ['selected_neural', 'selected_cem', 'retuned_setup_spt', 'spt']
    actual = [(r['policy'], int(r['scenario_seed'])) for r in rows]
    expected = {(p, s) for p in policies for s in range(24900, 24950)}
    if len(actual) != 200 or len(set(actual)) != 200 or set(actual) != expected:
        raise ValueError('Final-test rows do not match the locked 200-row grid')
    if any(float(r['invalid_actions']) != 0 for r in rows):
        raise ValueError('Invalid action recorded in final evidence')
    print('Verified selected neural SHA-256 and 200 unique final-test rows; zero invalid actions.')
    print('Fixed policy             Mean cost   Completion   On time')
    for policy in policies:
        part = [r for r in rows if r['policy'] == policy]
        mean = lambda name: sum(float(r[name]) for r in part) / len(part)
        print(f'{policy:24} {mean("full_objective"):9.3f}   {100*mean("completion_rate"):8.2f}%  {100*mean("on_time_rate"):7.2f}%')
    print('Overall validation winner:', selection['best_overall']['name'])
    print('Normalised simulation costs; fixed-policy test, not classification accuracy.')
    if args.replay:
        sys.path.insert(0, str(ROOT))
        import torch
        from sb3_contrib import MaskablePPO
        from factory_rl.inspection import InspectionConfig
        from factory_rl.residual_dispatch import DispatchEnv

        torch.set_num_threads(1)
        metadata = json.loads((checkpoint.parent / 'metadata.json').read_text())
        env = DispatchEnv(InspectionConfig(**metadata['config']))
        model = MaskablePPO.load(checkpoint, device='cpu')
        obs, _ = env.reset(seed=24900)
        print('Actual saved-model CPU replay: first eight decisions on scenario 24900.')
        for step in range(8):
            mask = env.action_masks()
            action = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
            if not mask[action]:
                raise ValueError('Model produced an illegal action')
            obs, reward, terminated, truncated, _ = env.step(action)
            print(f'  Decision {step+1}: action={action}, scaled_reward={reward:.6f}')
            if terminated or truncated:
                break
        env.close()
        print('Replay passed. This partial replay is distinct from the saved complete test.')


if __name__ == '__main__':
    main()
