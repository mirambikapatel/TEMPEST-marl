"""
Train Centralized DQN baseline on the existing 20-seed graphs and add
metrics to each seed_NNN.json in-place.  Does NOT retrain TEMPEST.

Fair-compute guarantee: the single network receives 4 gradient updates per
env step, matching TEMPEST which updates K=4 community networks once each.

Usage:
    python add_centralized_to_existing.py --indir multiseed_results --workers 4
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np


# ── Worker (runs in a separate process) ───────────────────────────────────────
def _worker(args_tuple):
    indir_str, sf_str, n_evals, n_eps, max_time = args_tuple

    # Per-worker stdout goes straight to console (no log redirect needed,
    # parallel workers each print their own prefix)
    sys.path.insert(0, str(Path(__file__).parent))

    import torch
    import warnings
    warnings.filterwarnings('ignore')

    from marl_core import (
        make_graph, GraphCache, CONTAIN_T,
        train_centralized, eval_centralized_avg,
    )

    sf = Path(sf_str)
    with open(sf) as f:
        data = json.load(f)

    seed     = data['seed']
    n_nodes  = 5000
    gtype    = 'ba'
    t_cut    = min(CONTAIN_T, max_time)
    I_sir    = np.array(data['I_curve_sir'])

    print(f'[seed={seed:03d}] reconstructing graph …', flush=True)
    G  = make_graph(gtype, n_nodes, seed)
    gc = GraphCache(G)

    # Deterministic, separate from TEMPEST stream (seed*1000+17 vs seed*1000)
    np.random.seed(seed * 1000 + 17)
    random.seed(seed * 1000 + 17)
    torch.manual_seed(seed * 1000 + 17)

    print(f'[seed={seed:03d}] training Centralized DQN ({n_eps} eps, 4× updates/step) …',
          flush=True)
    t0 = time.time()
    cmarl = train_centralized(gc, n_eps=n_eps, max_time=max_time)
    train_time = time.time() - t0
    print(f'[seed={seed:03d}] training done in {train_time:.1f}s', flush=True)

    print(f'[seed={seed:03d}] evaluating ({n_evals} rollouts) …', flush=True)
    h_cen = eval_centralized_avg(cmarl, max_time=max_time, n_evals=n_evals)
    I_c = np.array(h_cen['I']); R_c = np.array(h_cen['R'])

    sust_sir  = I_sir[CONTAIN_T:].mean()
    sust_cen  = I_c[CONTAIN_T:].mean()
    sust_red  = (sust_sir - sust_cen) / (sust_sir + 1e-9) * 100

    new_keys = {
        'peak_I_centralized'      : int(I_c.max()),
        'final_R_centralized'     : int(R_c[-1]),
        'truth_amp_centralized'   : float(R_c[:t_cut].mean() / (I_c[:t_cut].mean() + 1e-9)),
        'polarization_centralized': float(np.mean(np.abs(I_c - R_c) / (I_c + R_c + 1e-9))),
        'spread_red_centralized'  : float((I_sir.max() - I_c.max()) / (I_sir.max() + 1e-9)),
        'I_curve_centralized'     : list(map(int, I_c.tolist())),
        'R_curve_centralized'     : list(map(int, R_c.tolist())),
    }

    data.update(new_keys)
    with open(sf, 'w') as f:
        json.dump(data, f, indent=2,
                  default=lambda o: int(o) if hasattr(o, 'item') else float(o))

    print(
        f'[seed={seed:03d}] peak_I_sir={data["peak_I_sir"]}  '
        f'peak_I_cen={new_keys["peak_I_centralized"]}  '
        f'peak_I_marl={data["peak_I_marl"]}  '
        f'sred_cen={new_keys["spread_red_centralized"]*100:.1f}%  '
        f'sust_red={sust_red:.1f}%',
        flush=True,
    )

    return {
        'seed'               : seed,
        'peak_I_sir'         : data['peak_I_sir'],
        'peak_I_centralized' : new_keys['peak_I_centralized'],
        'peak_I_marl'        : data['peak_I_marl'],
        'sred_cen'           : new_keys['spread_red_centralized'] * 100,
        'sust_red_cen'       : sust_red,
        'train_time'         : train_time,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir',   default='multiseed_results')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--n-evals', type=int, default=5)
    ap.add_argument('--eps',     type=int, default=80,
                    help='Training episodes (must match original run, default 80)')
    ap.add_argument('--max-time',type=int, default=60)
    args = ap.parse_args()

    indir = Path(args.indir)
    seed_files = sorted(indir.glob('seed_*.json'))
    assert seed_files, f'No seed JSONs in {indir}'
    print(f'Found {len(seed_files)} seed JSONs')
    print(f'Workers={args.workers}  n_evals={args.n_evals}  '
          f'eps={args.eps}  max_time={args.max_time}')
    print()

    tasks = [
        (str(indir), str(sf), args.n_evals, args.eps, args.max_time)
        for sf in seed_files
    ]

    t0 = time.time()
    ctx = mp.get_context('spawn')
    rows = []
    with ctx.Pool(args.workers) as pool:
        for row in pool.imap_unordered(_worker, tasks):
            rows.append(row)
            elapsed = time.time() - t0
            print(f'  [{len(rows)}/{len(seed_files)} done  elapsed={elapsed:.0f}s]',
                  flush=True)

    rows.sort(key=lambda r: r['seed'])

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f'{"Seed":>5}  {"peak_I_sir":>10}  {"peak_I_cen":>10}  '
          f'{"peak_I_marl":>11}  {"sred_cen%":>9}  {"sust_red_cen%":>13}')
    print('-' * 67)
    for r in rows:
        print(f'{r["seed"]:>5}  {r["peak_I_sir"]:>10}  '
              f'{r["peak_I_centralized"]:>10}  {r["peak_I_marl"]:>11}  '
              f'{r["sred_cen"]:>8.1f}%  {r["sust_red_cen"]:>12.1f}%')
    print('-' * 67)

    def ms(vals):
        a = np.array(vals, dtype=float)
        return f'{a.mean():.1f} ± {a.std(ddof=1)/np.sqrt(len(a)):.1f}'

    print(f'\nAggregate across {len(rows)} seeds (mean ± SEM):')
    print(f'  peak_I_sir         : {ms([r["peak_I_sir"]         for r in rows])}')
    print(f'  peak_I_centralized : {ms([r["peak_I_centralized"] for r in rows])}')
    print(f'  peak_I_marl        : {ms([r["peak_I_marl"]        for r in rows])}')
    print(f'  spread_red_cen     : {ms([r["sred_cen"]           for r in rows])}%')
    print(f'  sust_red_cen       : {ms([r["sust_red_cen"]       for r in rows])}%')
    print(f'  train_time/seed    : {ms([r["train_time"]         for r in rows])}s')
    print(f'\nTotal wall-clock: {time.time()-t0:.0f}s')

    # ── Re-aggregate ──────────────────────────────────────────────────────────
    print(f'\nRe-aggregating {indir} …')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'run_seeds', Path(__file__).parent / 'run_seeds.py')
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    rs.aggregate_results(indir)
    print('Re-aggregation complete.')


if __name__ == '__main__':
    main()
