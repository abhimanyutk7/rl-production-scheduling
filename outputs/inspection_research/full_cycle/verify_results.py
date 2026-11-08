"""Recheck frozen full-cycle artifacts; run from the project root with its Python."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from factory_rl.inspection_study import paired_statistics, source_fingerprint

ROOT = Path(__file__).resolve().parent
KEY = ['policy', 'training_seed', 'scenario_seed']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    assert np.isclose(a, b, rtol=1e-10, atol=1e-7), (a, b)


def verify():
    manifest = json.loads((ROOT / 'run.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['source_hash'] == source_fingerprint()
    cfg = manifest['config']
    assert cfg['days'] == 60 and cfg['billing_days'] == 30
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert config_hash == manifest['config_hash']
    rules = ['reactive', 'belief', 'spt', 'periodic', 'uncertainty']
    learned = ['adaptive', 'imitation', 'no_inspection', 'exact_health']
    seeds, scenarios = manifest['training_seeds'], manifest['test_scenarios']
    assert seeds == list(range(5)) and scenarios == list(range(20000, 20500))
    expected = {(p, -1, j) for p in rules for j in scenarios}
    expected |= {(p, s, j) for p in learned for s in seeds for j in scenarios}
    compact, actual = [], []
    for frame in pd.read_csv(ROOT / 'episodes.csv', chunksize=500):
        actual.extend(frame[KEY].itertuples(index=False, name=None))
        numeric = frame.select_dtypes(include='number')
        assert np.isfinite(numeric.to_numpy()).all()
        assert frame.invalid_actions.eq(0).all()
        assert frame.released_jobs.eq(1800).all()
        for row in frame.to_dict('records'):
            daily, billing = json.loads(row['ledger_daily']), json.loads(row['ledger_billing'])
            assert [d['day'] for d in daily] == list(range(60))
            assert [b['period'] for b in billing] == [0, 1]
            close(sum(b['demand_cost'] for b in billing), row['demand_cost'])
            for bill in billing:
                close(bill['peak_kw'] * cfg['demand_rate'], bill['demand_cost'])
            prior = 0
            for day in daily:
                close(sum(day[k] for k in ['lateness', 'energy', 'demand', 'downtime', 'inspection', 'terminal']), day['full'])
                assert day['full'] >= prior - 1e-7
                prior = day['full']
            last = daily[-1]
            close(last['full'], row['full_objective'])
            for key in ['energy_cost', 'demand_cost', 'total_tardiness_min']:
                close(last[key], row[key])
            close(row['completed_jobs'] + row['unfinished_jobs'], row['released_jobs'])
            close(row['completion_rate'], row['completed_jobs'] / row['released_jobs'])
            objective = (cfg['tardiness_weight'] * row['total_tardiness_min'] / cfg['tardiness_scale']
                + cfg['energy_weight'] * row['energy_cost'] / cfg['energy_scale']
                + cfg['demand_weight'] * row['demand_cost'] / cfg['demand_scale']
                + cfg['maintenance_weight'] * (row['maintenance_downtime_min'] + row['repair_downtime_min'] + row['inspection_downtime_min']) / cfg['downtime_scale']
                + row['inspection_direct_cost'] / cfg['energy_scale']
                + cfg['unfinished_penalty'] * row['unfinished_jobs'])
            close(objective, row['full_objective'])
        compact.append(frame.drop(columns=['ledger_daily', 'ledger_billing']))
    assert len(actual) == len(set(actual)) == len(expected) == 12500
    assert set(actual) == expected
    frame = pd.concat(compact, ignore_index=True)
    checkpoints = []
    for condition in ['adaptive', 'no_inspection', 'exact_health']:
        for seed in seeds:
            directory = ROOT / 'models' / f'{condition}_s{seed}'
            meta = json.loads((directory / 'metadata.json').read_text())
            settings = json.loads((directory / 'evaluation_settings.json').read_text())
            assert meta['status'] == 'complete' and meta['actual_steps'] == 65536
            assert meta['condition'] == condition and meta['training_seed'] == seed
            assert meta['source_hash'] == settings['source_hash'] == manifest['source_hash']
            assert meta['config_hash'] == config_hash and meta['config'] == settings['config'] == cfg
            assert meta['exact_health'] == (condition == 'exact_health')
            assert meta['inspections_enabled'] == (condition != 'no_inspection')
            assert settings['condition'] == settings['label'] == condition and settings['seed'] == seed
            model_hash = sha(directory / 'model.zip')
            assert model_hash == meta['model_sha256'] == settings['model_hash']
            validation = json.loads((directory / 'validation.json').read_text())
            best = min(validation, key=lambda r: r['mean_cost'])
            assert best['step'] == meta['selected_steps']
            close(best['mean_cost'], meta['validation_cost'])
            data = json.loads(zipfile.ZipFile(directory / 'model.zip').read('data'))
            assert data['num_timesteps'] == meta['selected_steps'] and data['gamma'] == 1
            if condition == 'adaptive':
                assert settings['imitation_hash'] == sha(directory / 'imitation.zip')
                if meta['selected_steps'] == 0:
                    with zipfile.ZipFile(directory / 'model.zip') as a, zipfile.ZipFile(directory / 'imitation.zip') as b:
                        assert a.read('policy.pth') == b.read('policy.pth')
                    pair = frame.query('training_seed == @seed and policy in ["adaptive", "imitation"]')
                    pivot = pair.pivot(index='scenario_seed', columns='policy', values='full_objective')
                    np.testing.assert_allclose(pivot.adaptive, pivot.imitation, rtol=0, atol=1e-10)
            cached = pd.read_csv(directory / 'test_episodes.csv').drop(columns=['ledger_daily', 'ledger_billing'])
            labels = [condition, 'imitation'] if condition == 'adaptive' else [condition]
            subset = frame[frame.policy.isin(labels) & frame.training_seed.eq(seed)]
            pd.testing.assert_frame_equal(cached.sort_values(KEY).reset_index(drop=True),
                subset.sort_values(KEY).reset_index(drop=True), check_like=True, rtol=1e-10, atol=1e-8)
            checkpoints.append(dict(condition=condition, seed=seed, selected_steps=meta['selected_steps'],
                validation_cost=meta['validation_cost'], model_sha256=model_hash))
    selected = json.loads((ROOT / 'selected_rules.json').read_text())
    for rule in rules:
        directory = ROOT / 'rules' / rule
        settings = json.loads((directory / 'evaluation_settings.json').read_text())
        assert settings['source_hash'] == manifest['source_hash'] and settings['config'] == cfg
        assert settings['label'] == rule and settings['seed'] == -1 and settings['parameters'] == selected[rule]
        cached = pd.read_csv(directory / 'test_episodes.csv').drop(columns=['ledger_daily', 'ledger_billing'])
        pd.testing.assert_frame_equal(cached.sort_values(KEY).reset_index(drop=True),
            frame[frame.policy.eq(rule)].sort_values(KEY).reset_index(drop=True), check_like=True, rtol=1e-10, atol=1e-8)
    paired = pd.read_csv(ROOT / 'paired_comparisons.csv')
    for row in paired.to_dict('records'):
        result = paired_statistics(frame, row['candidate'], row['reference'])
        for key in ['mean_difference', 'ci_low', 'ci_high']:
            close(row[key], result[key])
    supplemental = pd.DataFrame([paired_statistics(frame, 'adaptive', 'spt')])
    supplemental['analysis_status'] = 'supplemental comparison after primary run; same fixed bootstrap'
    supplemental.to_csv(ROOT / 'paired_spt_supplement.csv', index=False)
    result = dict(date='2026-11-02', status='passed', rows=len(frame), models=len(checkpoints),
        exact_grid=True, invalid_actions=0, daily_records=60 * len(frame), billing_records=2 * len(frame),
        daily_and_billing_accounting='all reconciled', objective_formula='all rows reconciled',
        source_unchanged=True, source_hash=manifest['source_hash'], config_hash=config_hash,
        episodes_sha256=sha(ROOT / 'episodes.csv'), all_cache_rows_match=True,
        primary_bootstrap_reproduced=True, supplemental_spt=supplemental.to_dict('records')[0],
        checkpoints=checkpoints, numeric_policy_means=frame.groupby('policy').mean(numeric_only=True).to_dict('index'))
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['checkpoints', 'numeric_policy_means']}, indent=2))


def plot_results():
    """All six primary comparisons plus the explicitly supplemental SPT comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paired = pd.concat([pd.read_csv(ROOT / 'paired_spt_supplement.csv'),
                        pd.read_csv(ROOT / 'paired_comparisons.csv')], ignore_index=True)
    names = {'spt': 'SPT + belief maintenance *', 'belief': 'EDD + belief maintenance',
             'periodic': 'Periodic inspection', 'uncertainty': 'Uncertainty inspection',
             'no_inspection': 'No-inspection neural policy', 'imitation': 'Imitation only',
             'reactive': 'Reactive EDD'}
    with plt.rc_context({'font.size': 11}):
        fig, ax = plt.subplots(figsize=(11, 6.4))
        y = np.arange(len(paired))
        mean = paired.mean_difference.to_numpy()
        ax.errorbar(mean, y, xerr=np.vstack([mean-paired.ci_low, paired.ci_high-mean]),
                    fmt='o', color='#245879', ecolor='#245879', capsize=4, lw=1.8)
        for i, row in enumerate(paired.itertuples()):
            ax.text(425, i, f'{row.mean_difference:+.1f} [{row.ci_low:+.1f}, {row.ci_high:+.1f}]',
                    va='center', fontsize=10)
        ax.axvline(0, color='#333333', linestyle='--', lw=1)
        ax.set_yticks(y, [names[r] for r in paired.reference])
        ax.invert_yaxis()
        ax.set_xlim(-220, 700)
        ax.set_xticks([-200, -100, 0, 100, 200, 300, 400])
        ax.set_xlabel('Adaptive neural policy minus reference: normalised combined cost')
        ax.set_title('Original 60-day study: adaptive policy versus every primary comparator\n'
                     '5 training seeds; 500 matched sequences; 95% hierarchical bootstrap intervals', pad=18)
        ax.grid(axis='x', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        fig.subplots_adjust(left=.29, right=.97, bottom=.23, top=.81)
        fig.text(.29, .115, 'Negative favours adaptive; positive favours reference. Intervals include seed and scenario uncertainty.', fontsize=9)
        fig.text(.29, .077, '* SPT is a supplemental comparison after the primary run. 2,000 draws; fixed bootstrap seed 702.', fontsize=9)
        fig.text(.29, .039, 'Four of five adaptive checkpoints retained imitation. Simulated cost units; no real-world savings claim.', fontsize=9)
        fig.savefig(ROOT / 'paired_cost_differences.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    verify()
    plot_results()
