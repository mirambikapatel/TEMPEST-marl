"""
pheme_runner.py — train TEMPEST / run SIR / run type_prior on PHEME event networks.

Reads:
  results_v2/pheme/{event}/network.gpickle
  results_v2/pheme/{event}/meta.json
  results_v2/pheme/{event}/real_cascade.csv

Writes:
  results_v2/pheme/{event}/seed_XXX.json  (same format as revision_runner.py)

Usage:
  python3 pheme_runner.py --smoke                     # 1 seed, tiny training
  python3 pheme_runner.py --seeds 20 --workers 5      # full run
  python3 pheme_runner.py --events charliehebdo --seeds 1  # one event
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHEME_OUT = HERE / 'results_v2' / 'pheme'
QUALIFIED_EVENTS = ['charliehebdo', 'ottawashooting', 'sydneysiege', 'ferguson', 'germanwings-crash']
MAX_TIME = 60
N_EVALS = 5
T_CONTAIN = 30
N_EPS = 80


def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def job_path(event, seed):
    return PHEME_OUT / event / f'seed_{seed:03d}.json'


def load_event(event):
    out_dir = PHEME_OUT / event
    with open(out_dir / 'network.gpickle', 'rb') as f:
        G = pickle.load(f)
    meta = json.loads((out_dir / 'meta.json').read_text())
    real_cascade = {}
    with open(out_dir / 'real_cascade.csv') as f:
        import csv
        for row in csv.DictReader(f):
            for k in row:
                real_cascade.setdefault(k, []).append(
                    int(row[k]) if k in ('t', 'I_count', 'R_count') else float(row[k]))
    return G, meta, real_cascade


def pearson_rmse(sim, real):
    """Pearson r and RMSE between two equal-length arrays."""
    s = np.asarray(sim, float); r = np.asarray(real, float)
    if s.std() < 1e-9 or r.std() < 1e-9:
        corr = float(np.nan)
    else:
        corr = float(np.corrcoef(s, r)[0, 1])
    rmse = float(np.sqrt(np.mean((s - r) ** 2)))
    return corr, rmse


def compare_to_real(hist, real_cascade, total_participants):
    """Compute comparison metrics between simulated and real cascade."""
    n = max(total_participants, 1)
    # Normalised simulated curves
    I_sim = np.asarray(hist['I'], float) / n
    R_sim = np.asarray(hist['R'], float) / n
    I_real = np.asarray(real_cascade['I_norm'], float)
    R_real = np.asarray(real_cascade['R_norm'], float)

    I_r, I_rmse = pearson_rmse(I_sim, I_real)
    R_r, R_rmse = pearson_rmse(R_sim, R_real)

    # Peak time error
    t_peak_sim = int(np.argmax(I_sim))
    real_I_count = np.asarray(real_cascade['I_count'], float)
    t_peak_real = int(np.argmax(real_I_count)) if real_I_count.max() > 0 else 0
    peak_t_err = abs(t_peak_sim - t_peak_real)

    # Final refuter share
    final_I_sim = float(I_sim[-1]); final_R_sim = float(R_sim[-1])
    sim_refuter_share = final_R_sim / (final_I_sim + final_R_sim + 1e-9)
    real_I_final = float(I_real[-1]); real_R_final = float(R_real[-1])
    real_refuter_share = real_R_final / (real_I_final + real_R_final + 1e-9)

    return {
        'I_pearson_r': I_r, 'I_rmse': I_rmse,
        'R_pearson_r': R_r, 'R_rmse': R_rmse,
        'peak_t_sim': t_peak_sim, 'peak_t_real': t_peak_real,
        'peak_t_error': peak_t_err,
        'sim_refuter_share': sim_refuter_share,
        'real_refuter_share': real_refuter_share,
    }


def run_one(arg):
    event, seed, cfg = arg
    out = job_path(event, seed)
    if out.exists():
        return {'job': f'{event}/seed_{seed:03d}', 'status': 'skipped'}

    log_path = PHEME_OUT / event / 'logs' / f'seed_{seed:03d}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, 'w', buffering=1)
    sys.stdout = log_f; sys.stderr = log_f
    t0 = time.time()
    try:
        import warnings; warnings.filterwarnings('ignore')
        import torch
        import random
        torch.set_num_threads(cfg['threads'])
        sys.path.insert(0, str(HERE))
        import marl_core as mc
        import baselines as B

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        G, meta, real_cascade = load_event(event)
        gc = mc.GraphCache(G)
        np.random.seed(seed)   # node types depend only on seed

        K_use = min(meta['K_communities'], 4)
        mc.N_COMMUNITIES = K_use
        mc.CONTAIN_T = T_CONTAIN

        init_nodes = np.array(meta['init_nodes'], dtype=np.int64)
        # Clamp init_nodes to graph size
        init_nodes = init_nodes[init_nodes < gc.n]
        if len(init_nodes) == 0:
            init_nodes = np.array([0], dtype=np.int64)
        n_init = len(init_nodes)

        n_eps = cfg['eps']
        max_time = cfg['max_time']
        n_evals = cfg['n_evals']
        total_p = meta['total_participants']

        print(f'[pheme/{event}/seed {seed}] n={gc.n} K={K_use} T={T_CONTAIN} init_nodes={n_init}')

        r_cause = np.zeros(gc.n, dtype=np.int8)

        def fixed_init_rollout(step_fn):
            """Rollout starting from the real initial infected nodes."""
            states = np.zeros(gc.n, dtype=np.int64)
            states[init_nodes] = 1
            rc = np.zeros(gc.n, dtype=np.int8)
            carry = None
            hist = {k: [] for k in ('S', 'I', 'R', 'R_refuted', 'R_spont', 'frac_refute_action')}
            for t in range(max_time):
                actions, carry = step_fn(states, t, carry)
                new = mc.env_step(states, actions, gc, t, init_nodes=init_nodes)
                B.tag_transitions(states, new, actions, t, init_nodes, rc)
                states = new
                R = (states == 2)
                hist['S'].append(int((states == 0).sum()))
                hist['I'].append(int((states == 1).sum()))
                hist['R'].append(int(R.sum()))
                hist['R_refuted'].append(int((R & (rc == 1)).sum()))
                hist['R_spont'].append(int((R & (rc == 2)).sum()))
                hist['frac_refute_action'].append(float((actions == 2).mean()))
                assert hist['R'][-1] == hist['R_refuted'][-1] + hist['R_spont'][-1]
            return hist

        def avg_rollouts(step_fn):
            return B.average_hists([fixed_init_rollout(step_fn) for _ in range(n_evals)])

        res = {
            'exp': 'pheme', 'event': event, 'seed': seed,
            'gtype': 'pheme_follow', 'n_nodes': gc.n, 'n_edges': G.number_of_edges(),
            'K': K_use, 'contain_t': T_CONTAIN,
            'init_nodes': init_nodes.tolist(),
            'methods': {}, 'curves': {}, 'vs_real': {},
        }

        def record(name, h):
            res['methods'][name] = B.summarise(h, T_CONTAIN)
            res['curves'][name] = h
            res['vs_real'][name] = compare_to_real(h, real_cascade, total_p)
            print(f'  {name}: ' + ', '.join(f'{a}={b:.3f}'
                  for a, b in res['methods'][name].items() if isinstance(b, float)))

        # ── TEMPEST ────────────────────────────────────────────────────────────
        print('training TEMPEST')
        t1 = time.time()
        marl = mc.train_marl(gc, n_eps=n_eps, max_time=max_time)
        res['train_minutes_tempest'] = (time.time() - t1) / 60

        def tempest_step(states, t, h):
            import torch
            if h is None: h = torch.zeros(gc.n, mc.HIDDEN)
            nets = marl.online if isinstance(marl.online, list) else [marl.online]
            for net in nets: net.eval()
            with torch.no_grad():
                a, h = marl.select_actions(gc.features(states, t), h, eps=0.0)
            return a, h

        record('tempest', avg_rollouts(tempest_step))

        # ── SIR ───────────────────────────────────────────────────────────────
        h_sir = mc.run_sir_avg(gc, max_time, n_evals=n_evals, beta=0.2, gamma=0.02)
        h_sir = {kk: h_sir[kk] for kk in ('S', 'I', 'R')}
        h_sir['R_refuted'] = [0.0] * max_time
        h_sir['R_spont'] = list(h_sir['R'])
        record('sir', h_sir)

        # ── type_prior ────────────────────────────────────────────────────────
        def tp_step(states, t, c):
            return B.background_actions(states, t, T_CONTAIN, 'type_prior', gc.node_types), c
        record('type_prior', avg_rollouts(tp_step))

        res['runtime_seconds'] = time.time() - t0
        tmp = out.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(res, f, default=_json_default)
        os.replace(tmp, out)
        print(f'DONE in {res["runtime_seconds"] / 60:.1f} min')
        return {'job': str(out), 'status': 'ok', 'minutes': res['runtime_seconds'] / 60}

    except Exception as e:
        print(traceback.format_exc())
        return {'job': str(out), 'status': 'error', 'error': repr(e)}
    finally:
        log_f.close(); sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', nargs='*', default=QUALIFIED_EVENTS)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        args.seeds = 1
        n_eps = 3
        n_evals = 1
        max_time = MAX_TIME
    else:
        n_eps = N_EPS
        n_evals = N_EVALS
        max_time = MAX_TIME

    cfg = dict(eps=n_eps, max_time=max_time, n_evals=n_evals, threads=args.threads)

    jobs = [(ev, s, cfg) for ev in args.events
            for s in range(args.seeds)
            if PHEME_OUT / ev / 'meta.json' in [PHEME_OUT / ev / 'meta.json']
            and (PHEME_OUT / ev / 'meta.json').exists()]
    todo = [(ev, s, c) for ev, s, c in jobs if not job_path(ev, s).exists()]

    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run')
    for ev in args.events:
        n = sum(1 for e, s, _ in todo if e == ev)
        if n:
            print(f'  pheme/{ev}: {n} seeds')

    if args.list or not todo:
        return

    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        done = 0
        for r in pool.imap(run_one, todo):
            done += 1
            el = (time.time() - t0) / 60
            msg = f"[{done}/{len(todo)}] {r['status']}  {r['job']}"
            if r['status'] == 'ok': msg += f"  ({r['minutes']:.1f} min)"
            if r['status'] == 'error': msg += f"  ERROR {r['error']}"
            print(f'{msg}   elapsed {el:.0f} min', flush=True)

    print('Finished. Re-run to retry any failed jobs.')


if __name__ == '__main__':
    main()
