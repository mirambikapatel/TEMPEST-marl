"""
Summarise results_v2/ into paper-ready tables.

    python summarise_v2.py                  # all experiments found
    python summarise_v2.py --outdir results_v2_smoke

Writes results_v2/summary.md (human-readable) and results_v2/summary.json.
Each table: mean ± SEM over seeds; for every method other than TEMPEST, a paired
t-test vs TEMPEST on the same seeds (when TEMPEST ran in that job).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except ImportError:          # scipy optional
    stats = None

HERE = Path(__file__).resolve().parent
METRICS = [('peak_I', 'Peak I', 0), ('sustained_I', 'Mean I (t≥T)', 0), ('final_R', 'Final R', 0),
           ('final_R_refuted', 'Final R refuted', 0), ('final_R_spont', 'Final R spontaneous', 0),
           ('sp_growth', 'Signed pol. growth', 3), ('sp_contain', 'Signed pol. contain', 3),
           ('sp_ref_contain', 'Signed pol. contain (refuted R)', 3)]


def load(outdir):
    data = defaultdict(lambda: defaultdict(list))   # (exp, variant) -> list of results
    for f in sorted(Path(outdir).glob('*/*/seed_*.json')):
        r = json.loads(f.read_text())
        if 'variant' not in r:
            continue   # skip pheme/ and other non-revision jobs
        data[r['exp']][r['variant']['name']].append(r)
    return data


def fmt(arr, d):
    arr = np.asarray(arr, float)
    if len(arr) == 0: return '–'
    sem = (arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return f'{arr.mean():.{d}f} ± {sem:.{d}f}'


def table(results, ref='tempest'):
    methods = [m for m in results[0]['methods'] if not m.endswith('_meta')]
    seeds = [r['seed'] for r in results]
    rows, js = [], {}
    head = '| Method | n | ' + ' | '.join(lbl for _, lbl, _ in METRICS) + ' |'
    rows += [head, '|' + '---|' * (len(METRICS) + 2)]
    for m in methods:
        vals = {k: [r['methods'][m].get(k, np.nan) for r in results if m in r['methods']]
                for k, _, _ in METRICS}
        js[m] = {k: {'mean': float(np.nanmean(v)), 'sd': float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                     'n': len(v)} for k, v in vals.items() if len(v)}
        cells = [fmt([x for x in vals[k] if x == x], d) for k, _, d in METRICS]
        rows.append(f'| {m} | {len(vals["peak_I"])} | ' + ' | '.join(cells) + ' |')
    # paired tests vs reference
    if stats is not None and all(ref in r['methods'] for r in results) and len(results) > 2:
        rows.append('')
        rows.append(f'Paired t-tests vs {ref} (same seeds): mean difference (other − {ref}), t, p, Cohen d_z')
        for m in methods:
            if m == ref: continue
            for k, lbl, d in (('sustained_I', 'Mean I (t≥T)', 0), ('final_R_refuted', 'Final R refuted', 0),
                              ('sp_contain', 'Signed pol. contain', 3)):
                a = np.array([r['methods'][ref].get(k, np.nan) for r in results])
                b = np.array([r['methods'].get(m, {}).get(k, np.nan) for r in results])
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 3: continue
                diff = b[ok] - a[ok]
                t, p = stats.ttest_rel(b[ok], a[ok])
                dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
                rows.append(f'- {m} · {lbl}: Δ={diff.mean():+.{d}f}, t={t:+.2f}, p={p:.2g}, d_z={dz:+.2f}')
    return '\n'.join(rows), js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_v2')
    args = ap.parse_args()
    outdir = HERE / args.outdir
    data = load(outdir)
    md, js = [f'# TEMPEST revision results ({outdir.name})\n'], {}
    for exp in data:
        md.append(f'\n## {exp}\n')
        for var, results in sorted(data[exp].items()):
            r0 = results[0]
            md.append(f"\n### {exp} / {var}  (graph={r0['gtype']}, n={r0['n_nodes']}, K={r0['K']}, "
                      f"T={r0['contain_t']}, seeds={sorted(r['seed'] for r in results)})\n")
            t, j = table(results)
            md.append(t)
            js[f'{exp}/{var}'] = j
            if 'detection' in r0:
                for who in ('sir', 'tempest'):
                    dt = [r['detection'][who]['mean_detect_t'] for r in results]
                    md.append(f'\nDetection time {who}: {fmt(dt, 2)}')
    (outdir / 'summary.md').write_text('\n'.join(md))
    (outdir / 'summary.json').write_text(json.dumps(js, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
