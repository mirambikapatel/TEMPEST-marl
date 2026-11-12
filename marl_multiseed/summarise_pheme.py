"""
summarise_pheme.py — produce tables comparing TEMPEST/SIR/type_prior
on PHEME real-network events (mean ± SEM, paired t-tests vs TEMPEST).

Usage:
  python3 summarise_pheme.py
Appends to results_v2/pheme/summary.md and writes results_v2/pheme/summary.json.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHEME_OUT = HERE / 'results_v2' / 'pheme'
EVENTS = ['charliehebdo', 'ottawashooting', 'sydneysiege', 'ferguson', 'germanwings-crash']

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

METRICS_SIM = [
    ('peak_I', 'Peak I', 0),
    ('sustained_I', 'Mean I (t≥T)', 0),
    ('final_R', 'Final R', 0),
    ('sp_contain', 'Signed pol. contain', 3),
]
METRICS_REAL = [
    ('I_pearson_r', 'I Pearson r', 3),
    ('I_rmse', 'I RMSE', 3),
    ('R_pearson_r', 'R Pearson r', 3),
    ('R_rmse', 'R RMSE', 3),
    ('peak_t_error', 'Peak-t error', 1),
    ('sim_refuter_share', 'Sim R/(R+I)', 3),
    ('real_refuter_share', 'Real R/(R+I)', 3),
]


def fmt_mean_sem(arr, d):
    arr = np.asarray([x for x in arr if not math.isnan(x)], float)
    if len(arr) == 0:
        return '–'
    sem = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return f'{arr.mean():.{d}f} ± {sem:.{d}f}'


def load_event_results(event):
    results = []
    for f in sorted((PHEME_OUT / event).glob('seed_*.json')):
        results.append(json.loads(f.read_text()))
    return results


def sim_table(results, ref='tempest'):
    methods = [m for m in results[0]['methods'] if not m.endswith('_meta')]
    rows = []
    head = '| Method | n | ' + ' | '.join(lbl for _, lbl, _ in METRICS_SIM) + ' |'
    rows += [head, '|' + '---|' * (len(METRICS_SIM) + 2)]
    vals_by_method = {}
    for m in methods:
        vals = {k: [r['methods'][m].get(k, float('nan')) for r in results if m in r['methods']]
                for k, _, _ in METRICS_SIM}
        vals_by_method[m] = vals
        cells = [fmt_mean_sem(vals[k], d) for k, _, d in METRICS_SIM]
        rows.append(f'| {m} | {len(vals["peak_I"])} | ' + ' | '.join(cells) + ' |')
    if scipy_stats and all(ref in r['methods'] for r in results) and len(results) > 2:
        rows.append('')
        rows.append(f'Paired t-tests vs {ref}: Δ mean (other−{ref}), t, p, Cohen d_z')
        for m in methods:
            if m == ref:
                continue
            for k, lbl, d in [('sustained_I', 'Mean I (t≥T)', 0),
                               ('sp_contain', 'Signed pol. contain', 3)]:
                a = np.array([r['methods'].get(ref, {}).get(k, float('nan')) for r in results])
                b = np.array([r['methods'].get(m, {}).get(k, float('nan')) for r in results])
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 3:
                    continue
                diff = b[ok] - a[ok]
                t, p = scipy_stats.ttest_rel(b[ok], a[ok])
                dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
                rows.append(f'- {m} · {lbl}: Δ={diff.mean():+.{d}f}, t={t:+.2f}, p={p:.2g}, d_z={dz:+.2f}')
    return '\n'.join(rows)


def real_table(results):
    methods = [m for m in results[0].get('vs_real', {}).keys()]
    if not methods:
        return 'No vs_real data.'
    rows = []
    head = '| Method | n | ' + ' | '.join(lbl for _, lbl, _ in METRICS_REAL) + ' |'
    rows += [head, '|' + '---|' * (len(METRICS_REAL) + 2)]
    for m in methods:
        vals = {k: [r['vs_real'].get(m, {}).get(k, float('nan')) for r in results]
                for k, _, _ in METRICS_REAL}
        cells = [fmt_mean_sem(vals[k], d) for k, _, d in METRICS_REAL]
        n = sum(1 for r in results if m in r.get('vs_real', {}))
        rows.append(f'| {m} | {n} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def main():
    md = ['# PHEME TEMPEST experiment summary\n']
    js = {}
    meta_all = {}

    for event in EVENTS:
        results = load_event_results(event)
        if not results:
            md.append(f'\n## {event}: no results yet\n')
            continue
        meta = json.loads((PHEME_OUT / event / 'meta.json').read_text())
        meta_all[event] = meta
        seeds = sorted(r['seed'] for r in results)
        md.append(f'\n## {event}')
        md.append(f'n_nodes={meta["n_nodes"]}, n_edges={meta["n_edges"]}, '
                  f'K={meta["K_use"]}, T={results[0]["contain_t"]}, '
                  f'seeds={seeds}')
        md.append(f'Cascade: {meta["n_sharers"]} sharers, {meta["n_refuters"]} refuters '
                  f'(of {meta["total_participants"]} annotated participants)\n')
        md.append('### Simulation metrics (mean ± SEM)')
        md.append(sim_table(results))
        md.append('\n### Comparison to real cascade (mean ± SEM)')
        md.append(real_table(results))
        js[event] = {
            'meta': meta,
            'seeds': seeds,
            'n_seeds': len(seeds),
        }

    PHEME_OUT.mkdir(parents=True, exist_ok=True)
    (PHEME_OUT / 'summary.md').write_text('\n'.join(md))
    (PHEME_OUT / 'summary.json').write_text(json.dumps(js, indent=2))
    print('\n'.join(md))
    print(f'\nWritten: {PHEME_OUT / "summary.md"}')


if __name__ == '__main__':
    main()
