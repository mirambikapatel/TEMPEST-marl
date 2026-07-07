"""
Add Greedy Heuristic metrics to the existing 20-seed JSONs WITHOUT retraining.

For each seed:
  1. Reconstruct the exact graph used in that seed's run
  2. Run eval_greedy_policy for n_evals=5 rollouts (RNG seeded deterministically)
  3. Compute greedy metrics and add to seed_NNN.json in-place
  4. Report a per-seed summary table

Then re-aggregate with run_seeds.py --aggregate-only.

Usage:
    python add_greedy_to_existing.py --indir multiseed_results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))

import marl_core
from marl_core import (
    make_graph, GraphCache, eval_greedy_policy, run_greedy_avg,
    CONTAIN_T, INIT_INFECTED
)

N_EVALS = 5


def greedy_metrics(gc, seed, max_time, t_cut, I_sir, n_evals=N_EVALS):
    """Run greedy baseline and return metric dict."""
    np.random.seed(seed * 1000 + 7)
    h = run_greedy_avg(gc, max_time=max_time, n_evals=n_evals)
    I_g = np.array(h['I']); R_g = np.array(h['R'])
    return {
        'peak_I_greedy'      : int(I_g.max()),
        'final_R_greedy'     : int(R_g[-1]),
        'truth_amp_greedy'   : float(R_g[:t_cut].mean() / (I_g[:t_cut].mean() + 1e-9)),
        'polarization_greedy': float(np.mean(np.abs(I_g - R_g) / (I_g + R_g + 1e-9))),
        'spread_red_greedy'  : float((I_sir.max() - I_g.max()) / (I_sir.max() + 1e-9)),
        'I_curve_greedy'     : list(map(int, I_g.tolist())),
        'R_curve_greedy'     : list(map(int, R_g.tolist())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='multiseed_results')
    ap.add_argument('--n-evals', type=int, default=N_EVALS)
    args = ap.parse_args()

    indir = Path(args.indir)
    seed_files = sorted(indir.glob('seed_*.json'))
    assert seed_files, f"No seed JSONs found in {indir}"
    print(f"Found {len(seed_files)} seed JSONs in {indir}")
    print(f"Running Greedy baseline with n_evals={args.n_evals} per seed\n")

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"{'Seed':>5}  {'peak_I_sir':>10}  {'peak_I_greedy':>13}  {'peak_I_marl':>11}  "
          f"{'sred_greedy%':>12}  {'sust_red_greedy%':>16}")
    print('-' * 75)

    rows = []
    for sf in seed_files:
        with open(sf) as f:
            data = json.load(f)

        seed      = data['seed']
        gtype     = 'ba'          # all 20-seed runs used BA graph
        n_nodes   = 5000
        max_time  = len(data['I_curve_sir'])
        t_cut     = min(CONTAIN_T, max_time)
        I_sir     = np.array(data['I_curve_sir'])

        # Reconstruct the exact graph (same seed → same topology)
        G  = make_graph(gtype, n_nodes, seed)
        gc = GraphCache(G)

        metrics = greedy_metrics(gc, seed, max_time, t_cut, I_sir, args.n_evals)

        # Sustained reduction
        I_g_arr = np.array(metrics['I_curve_greedy'])
        I_s_arr = np.array(data['I_curve_sir'])
        s_sir   = I_s_arr[CONTAIN_T:].mean()
        s_grdy  = I_g_arr[CONTAIN_T:].mean()
        sust_red = (s_sir - s_grdy) / (s_sir + 1e-9) * 100

        print(f"{seed:>5}  {data['peak_I_sir']:>10}  {metrics['peak_I_greedy']:>13}  "
              f"{data['peak_I_marl']:>11}  "
              f"{metrics['spread_red_greedy']*100:>11.1f}%  {sust_red:>15.1f}%")

        # Update JSON in-place
        data.update(metrics)
        with open(sf, 'w') as f:
            json.dump(data, f, indent=2, default=lambda o:
                      int(o) if hasattr(o, 'item') else float(o))

        rows.append({
            'seed':         seed,
            'peak_I_sir':   data['peak_I_sir'],
            'peak_I_greedy': metrics['peak_I_greedy'],
            'peak_I_marl':  data['peak_I_marl'],
            'sred_greedy':  metrics['spread_red_greedy'] * 100,
            'sust_red_greedy': sust_red,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print('-' * 75)
    sreds  = np.array([r['sred_greedy']     for r in rows])
    susts  = np.array([r['sust_red_greedy'] for r in rows])
    peakGs = np.array([r['peak_I_greedy']   for r in rows])
    peakSs = np.array([r['peak_I_sir']      for r in rows])
    peakMs = np.array([r['peak_I_marl']     for r in rows])

    def ms(arr):
        return f"{arr.mean():.1f} ± {arr.std(ddof=1)/np.sqrt(len(arr)):.1f}"

    print(f"\nAggregate across {len(rows)} seeds (mean ± SEM):")
    print(f"  peak_I_sir    : {ms(peakSs)}")
    print(f"  peak_I_greedy : {ms(peakGs)}")
    print(f"  peak_I_marl   : {ms(peakMs)}")
    print(f"  spread_red_greedy  : {ms(sreds)}%")
    print(f"  sust_red_greedy    : {ms(susts)}%")

    # ── Re-aggregate using run_seeds.py ───────────────────────────────────────
    print(f"\nRe-aggregating {indir} ...")
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        'run_seeds', Path(__file__).parent / 'run_seeds.py')
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    agg = rs.aggregate_results(indir)
    print("Re-aggregation complete.")
    return rows


if __name__ == '__main__':
    main()
