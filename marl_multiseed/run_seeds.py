"""
Parallel multi-seed runner for MARL-DQN rumour spreading.

Runs N independent seeds in parallel (one process per seed), each doing the
full train → eval → metrics pipeline. Saves per-seed results to JSON and
aggregates mean ± std for the paper.

Usage:
    python run_seeds.py --seeds 20 --workers 8 --eps 80 --nodes 5000
    python run_seeds.py --seeds 5  --workers 5 --eps 30 --nodes 1000  # smoke test

Outputs (in --outdir):
    seed_<N>.json   per-seed results
    aggregate.json  mean/std across all seeds
    aggregate.txt   human-readable summary table

Designed to be launched by Claude Code (or any terminal) while you write
the paper. Logs go to <outdir>/logs/seed_<N>.log so you can tail progress.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np


def run_one_seed(args_tuple):
    """
    Worker function: runs the full pipeline for one seed.
    Imports happen inside so each worker gets a fresh torch context.
    """
    seed, cfg = args_tuple
    outdir = Path(cfg['outdir'])
    log_path = outdir / 'logs' / f'seed_{seed:03d}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Redirect stdout/stderr for this worker to its log file
    log_f = open(log_path, 'w', buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    t0 = time.time()
    print(f'[seed={seed}] starting at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    try:
        # Import inside worker so each gets fresh state
        import random
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        import networkx as nx
        from collections import deque, namedtuple
        import warnings
        warnings.filterwarnings('ignore')

        # Seed everything
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Import the experiment module that wraps the notebook's code
        # (we keep marl_core.py side-by-side with this script)
        sys.path.insert(0, str(Path(__file__).parent))
        from marl_core import run_full_experiment

        result = run_full_experiment(seed=seed, cfg=cfg)
        result['seed'] = seed
        result['runtime_seconds'] = time.time() - t0

        # Save per-seed result
        out_path = outdir / f'seed_{seed:03d}.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, default=_json_default)
        print(f'[seed={seed}] DONE in {result["runtime_seconds"]:.1f}s → {out_path}')
        return {'seed': seed, 'status': 'ok', 'path': str(out_path)}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f'[seed={seed}] FAILED:\n{tb}')
        return {'seed': seed, 'status': 'error', 'error': str(e), 'traceback': tb}
    finally:
        log_f.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def _json_default(o):
    """Make numpy types JSON-serialisable."""
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(f'Not serialisable: {type(o)}')


def aggregate_results(outdir: Path):
    """Read all seed_*.json files and compute mean ± std for each metric."""
    seed_files = sorted(outdir.glob('seed_*.json'))
    if not seed_files:
        print('No seed results found.')
        return None

    all_results = []
    for sf in seed_files:
        with open(sf) as f:
            all_results.append(json.load(f))

    print(f'Aggregating {len(all_results)} seeds...')

    # Scalar metrics — compute mean/std/sem
    scalar_keys = [
        'spread_reduction', 'peak_I_sir', 'peak_I_marl',
        'truth_amp_sir', 'truth_amp_marl',
        'polarization_sir', 'polarization_marl',
        'influential_efficiency',
        'rumours_generated', 'rumours_detected_sir', 'rumours_detected_marl',
        'detection_rate_sir', 'detection_rate_marl',
        'mean_detect_t_sir', 'mean_detect_t_marl',
        'mean_peak_I_per_rumour_sir', 'mean_peak_I_per_rumour_marl',
        'final_R_sir', 'final_R_marl',
        # Greedy heuristic baseline
        'peak_I_greedy', 'final_R_greedy',
        'truth_amp_greedy', 'polarization_greedy', 'spread_red_greedy',
        # Centralized DQN baseline
        'peak_I_centralized', 'final_R_centralized',
        'truth_amp_centralized', 'polarization_centralized', 'spread_red_centralized',
    ]
    agg = {'n_seeds': len(all_results), 'seeds_used': [r['seed'] for r in all_results]}
    for k in scalar_keys:
        vals = [r[k] for r in all_results if k in r and r[k] is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[k] = {
            'mean': float(arr.mean()),
            'std':  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'sem':  float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            'min':  float(arr.min()),
            'max':  float(arr.max()),
            'n':    int(len(arr)),
        }

    # Curve metrics — stack and compute mean ± std per timestep
    curve_keys = [
        'I_curve_sir', 'I_curve_marl', 'R_curve_sir', 'R_curve_marl',
        'I_curve_greedy', 'R_curve_greedy',
        'I_curve_centralized', 'R_curve_centralized',
    ]
    for k in curve_keys:
        stacks = [r[k] for r in all_results if k in r]
        if not stacks:
            continue
        # Pad to same length if needed
        max_len = max(len(c) for c in stacks)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in stacks])
        agg[k] = {
            'mean': padded.mean(axis=0).tolist(),
            'std':  padded.std(axis=0, ddof=1).tolist() if len(padded) > 1 else [0.0]*max_len,
            'n':    len(padded),
        }

    # Save aggregate JSON
    agg_path = outdir / 'aggregate.json'
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'Aggregate saved → {agg_path}')

    # Save human-readable summary
    txt_path = outdir / 'aggregate.txt'
    with open(txt_path, 'w') as f:
        f.write(f'MARL-DQN Multi-Seed Aggregate (n={agg["n_seeds"]} seeds)\n')
        f.write('=' * 78 + '\n\n')
        f.write(f'Seeds: {agg["seeds_used"]}\n\n')
        f.write(f'{"Metric":<35} {"mean":>10} {"std":>10} {"sem":>10} {"min":>8} {"max":>8}\n')
        f.write('-' * 78 + '\n')
        for k in scalar_keys:
            if k not in agg: continue
            s = agg[k]
            f.write(f'{k:<35} {s["mean"]:>10.4f} {s["std"]:>10.4f} '
                    f'{s["sem"]:>10.4f} {s["min"]:>8.3f} {s["max"]:>8.3f}\n')
    print(f'Summary table saved → {txt_path}')
    return agg


def main():
    ap = argparse.ArgumentParser(description='Parallel multi-seed MARL runner')
    ap.add_argument('--seeds',   type=int, default=20,   help='Number of seeds (default 20)')
    ap.add_argument('--workers', type=int, default=4,    help='Parallel workers (default 4)')
    ap.add_argument('--eps',     type=int, default=80,   help='Training episodes per seed (default 80)')
    ap.add_argument('--nodes',   type=int, default=5000, help='Graph size (default 5000)')
    ap.add_argument('--max-time',type=int, default=60,   help='Episode length / sim horizon (default 60)')
    ap.add_argument('--gtype',   type=str, default='ba', help='Graph type: ba/er/ws/sbm (default ba)')
    ap.add_argument('--n-rumours',type=int, default=10,  help='Rumours per seed for detection metrics (default 10)')
    ap.add_argument('--n-evals', type=int, default=5,    help='Eval rollouts per seed, averaged (default 5)')
    ap.add_argument('--outdir',  type=str, default='multiseed_results',
                    help='Output directory (default multiseed_results/)')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='Skip running; just re-aggregate existing seed JSONs')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'logs').mkdir(exist_ok=True)

    if args.aggregate_only:
        aggregate_results(outdir)
        return

    n_workers = args.workers if args.workers is not None else min(args.seeds, mp.cpu_count())
    print(f'Launching {args.seeds} seeds across {n_workers} workers')
    print(f'  nodes={args.nodes}  eps={args.eps}  max_time={args.max_time}  gtype={args.gtype}')
    print(f'  outdir={outdir.resolve()}')
    print(f'  logs: tail -f {outdir / "logs" / "seed_*.log"}')

    cfg = {
        'eps':       args.eps,
        'nodes':     args.nodes,
        'max_time':  args.max_time,
        'gtype':     args.gtype,
        'n_rumours': args.n_rumours,
        'n_evals':   args.n_evals,
        'outdir':    str(outdir),
    }

    tasks = [(s, cfg) for s in range(args.seeds)]
    t0 = time.time()

    # 'spawn' avoids torch + fork issues on macOS / some Linux setups
    ctx = mp.get_context('spawn')
    with ctx.Pool(n_workers) as pool:
        results = []
        for r in pool.imap_unordered(run_one_seed, tasks):
            results.append(r)
            done = len(results)
            ok = sum(1 for x in results if x['status'] == 'ok')
            elapsed = time.time() - t0
            print(f'  Progress: {done}/{args.seeds}  (ok={ok})  elapsed={elapsed:.1f}s')

    fail = [r for r in results if r['status'] != 'ok']
    print(f'\nAll done. ok={len(results)-len(fail)}, failed={len(fail)}')
    if fail:
        print('Failures:')
        for r in fail:
            print(f'  seed={r["seed"]}: {r["error"]}')

    aggregate_results(outdir)


if __name__ == '__main__':
    main()
