"""
TCSS major-revision experiment runner (TEMPEST).

One command runs every revision experiment in priority order, in parallel,
and is RESUMABLE: a job whose JSON already exists is skipped, so if the laptop
sleeps or a job crashes, just run the same command again.

    python revision_runner.py --smoke                 # quick check (~10-15 min)
    python revision_runner.py --workers 5             # full plan
    python revision_runner.py --only main ablation    # subset of experiments
    python revision_runner.py --list                  # print the job plan only

Outputs:  results_v2/<experiment>/<variant>/seed_XXX.json   (+ logs/, models/)
Does NOT touch multiseed_results/ (the numbers in the submitted paper).

What is new compared with run_seeds.py
  * R-state tagging: every I->R transition is tagged refuted (agent chose Refute)
    or spontaneous (anything else), computed outside env_step so the kernel's RNG
    sequence is unchanged. R_refuted + R_spont == R is asserted.
  * Signed polarization (R-I)/(R+I+1e-9): full run, growth (t<T), containment (t>=T),
    and the same using refuted-only R.
  * Curves saved as floats (no int truncation).
  * Trained models saved (TEMPEST community nets, centralised DQN, QDOM Q-table,
    DRLE policy) so later re-evaluation needs no retraining.
  * New baselines: no-learning type-prior population, QDOM-adapted, DRLE-adapted.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Experiment plan (priority order). Each experiment: list of variants, seeds,
# and which methods to run. Overrides are module-level constants of marl_core.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {'PEER_ALPHA': 0.70, 'BASE_SPREAD': 0.04,
                  'REFUTE_PROB_CONTAIN': 0.40, 'REFUTE_CASCADE': 0.12}

PLAN = [
    # 1. Baselines for refs 36/37 on the SAME 20 graphs and user types as the
    #    submitted results (multiseed_results/). TEMPEST/SIR/greedy/centralised DQN
    #    are NOT rerun -- their submitted numbers are compared seed-by-seed.
    dict(exp='baselines', seeds=20, variants=[dict(name='ba_k4_T30')],
         methods=['type_prior', 'qdom', 'drle', 'linkcut_is']),
    # 2. Ablation, 20 seeds (Full = submitted results, same graphs).
    dict(exp='ablation', seeds=20, variants=[dict(name='ba_k4_T30')],
         methods=['abl_no_contagion', 'abl_random_policy', 'abl_no_egt', 'abl_no_ignition']),
    # 3. Other network types + one real network.
    dict(exp='networks', seeds=20,
         variants=[dict(name='er', gtype='er'), dict(name='ws', gtype='ws'),
                   dict(name='facebook', gtype='facebook')],
         methods=['tempest', 'sir', 'greedy', 'type_prior']),
    # 4. One-at-a-time sensitivity for every parameter Reviewer 1 listed
    #    (alpha_peer, beta_base, p_contain, p_growth, p_casc, node-type prior, payoff matrix M).
    dict(exp='sensitivity', seeds=10,
         variants=[dict(name=f'{p}_{v}', overrides={p: v}) for p, vals in [
             ('PEER_ALPHA', [0.50, 0.90]), ('BASE_SPREAD', [0.02, 0.06]),
             ('REFUTE_PROB_CONTAIN', [0.25, 0.55]), ('REFUTE_PROB_GROWTH', [0.05, 0.15]),
             ('REFUTE_CASCADE', [0.06, 0.18]), ('PAYOFF_SCALE', [0.75, 1.25])]
             for v in vals]
         + [dict(name='TYPES_more_informed', overrides={'NODE_TYPE_PROBS': [0.45, 0.40, 0.15]}),
            dict(name='TYPES_fewer_informed', overrides={'NODE_TYPE_PROBS': [0.15, 0.60, 0.25]})],
         methods=['tempest', 'sir', 'type_prior']),
    # 5. Number of community partitions K (4 = main), up to 12, plus no merging.
    dict(exp='k_sweep', seeds=10,
         variants=[dict(name=f'k{k}', k=k) for k in (2, 6, 8, 10, 12)]
         + [dict(name='k_nomerge', k='nomerge')],
         methods=['tempest', 'sir']),
    # 6. Baselines in their native timing (t=0) and with other user backgrounds.
    dict(exp='baselines_extra', seeds=10,
         variants=[dict(name='t0_type_prior', intervene_t=0),
                   dict(name='T_growth_share', background='growth_share'),
                   dict(name='T_passive', background='passive')],
         methods=['qdom', 'drle', 'linkcut_is']),
    # 7. OPTIONAL (later): SIR parameter grid (no training; cheap).
    dict(exp='sir_grid', seeds=20, optional=True,
         variants=[dict(name=f'b{b}_g{g}', sir_beta=b, sir_gamma=g)
                   for b in (0.1, 0.2, 0.3) for g in (0.02, 0.05, 0.1)],
         methods=['sir']),
    # 8. OPTIONAL: retrain with other containment start times (moves baselines too).
    dict(exp='T_sweep', seeds=10, optional=True,
         variants=[dict(name=f'T{T}', contain_t=T) for T in (20, 25, 35, 40)],
         methods=['tempest', 'sir', 'greedy', 'type_prior', 'qdom']),
    # 10. OPTIONAL: separate tagged TEMPEST run, used ONLY to report what share of
    #     TEMPEST's R is active refutation vs spontaneous (Reviewer 3 #2). Does not
    #     replace any table in the paper.
    dict(exp='r_composition', seeds=10, optional=True, variants=[dict(name='ba_k4_T30')],
         methods=['tempest', 'greedy', 'sir']),
    # 9. OPTIONAL: exploration schedule that actually reaches eps=0.05.
    dict(exp='eps_decay', seeds=10, optional=True,
         variants=[dict(name='decay0963', overrides={'EPS_DECAY': 0.963})],
         methods=['tempest', 'sir']),
]


def build_jobs(args):
    jobs = []
    for e in PLAN:
        if args.only and e['exp'] not in args.only:
            continue
        if e.get('optional') and not args.include_optional and not (args.only and e['exp'] in args.only):
            continue
        n_seeds = args.smoke_seeds if args.smoke else (args.seeds or e['seeds'])
        for v in e['variants']:
            for s in range(n_seeds):
                jobs.append(dict(exp=e['exp'], variant=v, methods=e['methods'], seed=s))
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def _json_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def job_paths(job, cfg):
    d = Path(cfg['outdir']) / job['exp'] / job['variant']['name']
    return d / f"seed_{job['seed']:03d}.json", d / 'logs' / f"seed_{job['seed']:03d}.log", d / 'models'


def run_job(arg):
    job, cfg = arg
    out_path, log_path, model_dir = job_paths(job, cfg)
    if out_path.exists():
        return dict(job=f"{job['exp']}/{job['variant']['name']}/{job['seed']}", status='skipped')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, 'w', buffering=1)
    sys.stdout = log_f; sys.stderr = log_f
    t0 = time.time()
    try:
        import warnings; warnings.filterwarnings('ignore')
        import torch
        torch.set_num_threads(cfg['threads'])
        sys.path.insert(0, str(HERE))
        result = _run(job, cfg, model_dir)
        result['runtime_seconds'] = time.time() - t0
        tmp = out_path.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(result, f, default=_json_default)
        os.replace(tmp, out_path)                       # atomic: no half-written JSON
        print(f'DONE in {result["runtime_seconds"]/60:.1f} min')
        return dict(job=str(out_path), status='ok', minutes=result['runtime_seconds'] / 60)
    except Exception as e:
        print(traceback.format_exc())
        return dict(job=str(out_path), status='error', error=repr(e))
    finally:
        log_f.close(); sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__


def _make_graph(mc, gtype, n, seed):
    import networkx as nx
    if gtype == 'facebook':
        path = HERE / 'OUR_program' / 'outbreak' / 'facebook_combined.txt'
        G = nx.read_edgelist(str(path), nodetype=int)
        G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
        return G
    return mc.make_graph(gtype, n, seed)


def _rollout(mc, B, gc, max_time, step_fn, init_infected):
    """Generic tagged rollout. step_fn(states, t, carry) -> (actions, carry)."""
    n = gc.n
    states = np.zeros(n, dtype=np.int64)
    init_nodes = np.random.choice(n, init_infected, replace=False)
    states[init_nodes] = 1
    r_cause = np.zeros(n, dtype=np.int8)
    carry = None
    hist = {k: [] for k in ('S', 'I', 'R', 'R_refuted', 'R_spont', 'frac_refute_action')}
    for t in range(max_time):
        actions, carry = step_fn(states, t, carry)
        new = mc.env_step(states, actions, gc, t, init_nodes=init_nodes)
        B.tag_transitions(states, new, actions, t, init_nodes, r_cause)
        states = new
        R = (states == 2)
        hist['S'].append(int((states == 0).sum())); hist['I'].append(int((states == 1).sum()))
        hist['R'].append(int(R.sum()))
        hist['R_refuted'].append(int((R & (r_cause == 1)).sum()))
        hist['R_spont'].append(int((R & (r_cause == 2)).sum()))
        hist['frac_refute_action'].append(float((actions == 2).mean()))
        assert hist['R'][-1] == hist['R_refuted'][-1] + hist['R_spont'][-1]
    return hist


def _avg(B, hs):
    return B.average_hists(hs)


def _run(job, cfg, model_dir):
    import random
    import torch
    import marl_core as mc
    import baselines as B

    v = job['variant']; seed = job['seed']
    # ---- configure module constants for this job --------------------------
    for k, val in v.get('overrides', {}).items():
        if k == 'PAYOFF_SCALE':            # scale the whole EGT payoff matrix M
            mc._TYPE_PAYOFFS = (mc._TYPE_PAYOFFS * val).astype(np.float32)
        else:
            setattr(mc, k, val)
    contain_t = v.get('contain_t', 30)
    mc.CONTAIN_T = contain_t
    gtype = v.get('gtype', 'ba')
    max_time, n_evals, eps = cfg['max_time'], cfg['n_evals'], cfg['eps']

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G = _make_graph(mc, gtype, cfg['nodes'], seed)
    np.random.seed(seed)                       # node types depend only on seed
    gc = mc.GraphCache(G)

    k = v.get('k', 4)
    if k == 'nomerge':
        from networkx.algorithms.community import greedy_modularity_communities
        k = len(list(greedy_modularity_communities(G)))
    mc.N_COMMUNITIES = int(k)

    res = {'exp': job['exp'], 'variant': v, 'seed': seed, 'gtype': gtype,
           'n_nodes': gc.n, 'n_edges': G.number_of_edges(), 'K': int(k),
           'contain_t': contain_t, 'methods': {}, 'curves': {}}
    print(f"[{job['exp']}/{v['name']}/seed {seed}] n={gc.n} m={G.number_of_edges()} K={k} T={contain_t}")

    def record(name, h):
        res['methods'][name] = B.summarise(h, contain_t)
        res['curves'][name] = h
        print(f'  {name}: ' + ', '.join(f'{a}={b:.3f}' for a, b in res['methods'][name].items()))

    model_dir.mkdir(parents=True, exist_ok=True)
    marl = None
    methods = job['methods']

    # ---- TEMPEST ----------------------------------------------------------
    def tempest_eval(system):
        nets = system.online if isinstance(system.online, list) else [system.online]
        for net in nets:
            net.eval()
        def step(states, t, h):
            if h is None: h = torch.zeros(gc.n, mc.HIDDEN)
            with torch.no_grad():
                a, h = system.select_actions(gc.features(states, t), h, eps=0.0)
            return a, h
        return _avg(B, [_rollout(mc, B, gc, max_time, step, mc.INIT_INFECTED) for _ in range(n_evals)])

    if 'tempest' in methods or 'detection' in methods:
        print('training TEMPEST'); t1 = time.time()
        marl = mc.train_marl(gc, n_eps=eps, max_time=max_time)
        res['train_minutes_tempest'] = (time.time() - t1) / 60
        torch.save([n.state_dict() for n in marl.online], model_dir / f'tempest_seed{seed:03d}.pt')
        record('tempest', tempest_eval(marl))

    if 'sir' in methods:
        b = v.get('sir_beta', 0.2); g = v.get('sir_gamma', 0.02)
        h = mc.run_sir_avg(gc, max_time, n_evals=n_evals, beta=b, gamma=g)
        h = {kk: h[kk] for kk in ('S', 'I', 'R')}
        h['R_refuted'] = [0.0] * max_time; h['R_spont'] = list(h['R'])     # SIR: no refutation channel
        res['sir_params'] = {'beta': b, 'gamma': g}
        record('sir', h)

    if 'greedy' in methods:
        pr_thr = np.sort(gc.pr)[int(gc.n * 0.80)]
        def gstep(states, t, c):
            a = np.zeros(gc.n, dtype=np.int64)
            if t >= mc.CONTAIN_T:
                a[states == 1] = 2
                inf = states == 1
                for i in np.where(states == 0)[0]:
                    if any(inf[j] for j in gc.nbrs[i]): a[i] = 2
            else:
                a[(states == 0) & (gc.pr >= pr_thr)] = 1
                if t > 5: a[states == 1] = 2
            return a, c
        np.random.seed(seed * 1000 + 7)
        record('greedy', _avg(B, [_rollout(mc, B, gc, max_time, gstep, mc.INIT_INFECTED) for _ in range(n_evals)]))

    if 'type_prior' in methods:
        def tstep(states, t, c):
            return B.background_actions(states, t, mc.CONTAIN_T, 'type_prior', gc.node_types), c
        record('type_prior', _avg(B, [_rollout(mc, B, gc, max_time, tstep, mc.INIT_INFECTED) for _ in range(n_evals)]))

    if 'centralized' in methods:
        np.random.seed(seed * 1000 + 17); random.seed(seed * 1000 + 17); torch.manual_seed(seed * 1000 + 17)
        print('training centralised DQN')
        cm = mc.train_centralized(gc, n_eps=eps, max_time=max_time)
        torch.save(cm.online.state_dict(), model_dir / f'centralized_seed{seed:03d}.pt')
        record('centralized', tempest_eval(cm))

    if 'detection' in methods and marl is not None:
        rs = mc.simulate_multi_rumour(gc, None, cfg['n_rumours'], max_time)
        rm = mc.simulate_multi_rumour(gc, marl, cfg['n_rumours'], max_time)
        res['detection'] = {'sir': {kk: rs[kk] for kk in ('detection_rate', 'mean_detect_t', 'mean_peak_I')},
                            'tempest': {kk: rm[kk] for kk in ('detection_rate', 'mean_detect_t', 'mean_peak_I')}}
        IE, mr, sr = mc.influential_efficiency(gc, res['curves']['tempest'], res['curves'].get('sir', res['curves']['tempest']))
        res['influential_efficiency'] = {'IE': IE, 'tempest_rate': mr, 'sir_rate': sr}

    # ---- intervention baselines -------------------------------------------
    for name in ('qdom', 'drle', 'linkcut_is'):
        if name not in methods: continue
        bcfg = {'eps': eps, 'max_time': max_time, 'n_evals': n_evals, 'contain_t': contain_t,
                'intervene_t': v.get('intervene_t', 'contain'),
                'background': v.get('background', 'type_prior'), 'methods': [name]}
        out = B.run_baselines(seed, bcfg, gc=gc)
        record(name, out['curves'][name])
        res['methods'][name + '_meta'] = {kk: out[kk] for kk in out
                                          if kk.startswith(name) or kk in ('intervene_t', 'background')}
        record(f"no_intervention_{bcfg['background']}", out['curves']['no_intervention'])

    # ---- ablations (same graph & seed as main) ----------------------------
    if 'abl_no_contagion' in methods:
        # passive kernel: BASE_SPREAD contagion only, 3% recovery, no agent actions
        def passive_once():
            s = np.zeros(gc.n, dtype=np.int64); s[np.random.choice(gc.n, mc.INIT_INFECTED, replace=False)] = 1
            hh = {kk: [] for kk in ('S', 'I', 'R')}
            for t in range(max_time):
                inten = mc.rumour_intensity(t); new = s.copy()
                for u in np.where(s == 1)[0]:
                    for w in gc.nbrs[u]:
                        if s[w] == 0 and np.random.rand() < mc.BASE_SPREAD * inten: new[w] = 1
                    if np.random.rand() < 0.03: new[u] = 2
                s = new
                hh['S'].append(int((s == 0).sum())); hh['I'].append(int((s == 1).sum())); hh['R'].append(int((s == 2).sum()))
            hh['R_refuted'] = [0] * max_time; hh['R_spont'] = list(hh['R'])
            return hh
        record('abl_no_contagion', _avg(B, [passive_once() for _ in range(n_evals)]))
    if 'abl_random_policy' in methods:
        record('abl_random_policy', tempest_eval(mc.MARLSystem(gc)))
    if 'abl_no_egt' in methods:
        orig = mc.reward_fn
        mc.reward_fn = lambda s, a, g, t, flat=False: orig(s, a, g, t, flat=True)
        try:
            m2 = mc.train_marl(gc, n_eps=eps, max_time=max_time)
        finally:
            mc.reward_fn = orig
        record('abl_no_egt', tempest_eval(m2))
    if 'abl_no_ignition' in methods:
        orig_step = mc.env_step
        mc.env_step = lambda s, a, g, t, init_nodes=None: orig_step(s, a, g, t, init_nodes=None)
        try:
            m3 = mc.train_marl(gc, n_eps=eps, max_time=max_time)
            record('abl_no_ignition', tempest_eval(m3))
        finally:
            mc.env_step = orig_step
    return res


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--threads', type=int, default=2, help='torch threads per worker')
    ap.add_argument('--eps', type=int, default=80)
    ap.add_argument('--nodes', type=int, default=5000)
    ap.add_argument('--max-time', type=int, default=60)
    ap.add_argument('--n-evals', type=int, default=5)
    ap.add_argument('--n-rumours', type=int, default=10)
    ap.add_argument('--seeds', type=int, default=None, help='override seeds per experiment')
    ap.add_argument('--only', nargs='*', default=None, help='experiments to run')
    ap.add_argument('--include-optional', action='store_true')
    ap.add_argument('--outdir', default='results_v2')
    ap.add_argument('--smoke', action='store_true', help='tiny run of every experiment')
    ap.add_argument('--smoke-seeds', type=int, default=1)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        args.nodes, args.eps, args.n_evals, args.n_rumours = 500, 3, 1, 2
        args.outdir = args.outdir + '_smoke'
    cfg = dict(eps=args.eps, nodes=args.nodes, max_time=args.max_time, n_evals=args.n_evals,
               n_rumours=args.n_rumours, outdir=str(HERE / args.outdir), threads=args.threads)
    jobs = build_jobs(args)
    todo = [j for j in jobs if not job_paths(j, cfg)[0].exists()]
    print(f'{len(jobs)} jobs in plan, {len(jobs) - len(todo)} already done, {len(todo)} to run')
    by = {}
    for j in todo:
        by[f"{j['exp']}/{j['variant']['name']}"] = by.get(f"{j['exp']}/{j['variant']['name']}", 0) + 1
    for kk, n in by.items():
        print(f'  {kk}: {n}')
    if args.list or not todo:
        return
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        done = 0
        for r in pool.imap(run_job, [(j, cfg) for j in todo]):
            done += 1
            el = (time.time() - t0) / 60
            msg = f"[{done}/{len(todo)}] {r['status']}  {r['job']}"
            if r['status'] == 'ok': msg += f"  ({r['minutes']:.1f} min)"
            if r['status'] == 'error': msg += f"  ERROR {r['error']}"
            print(f'{msg}   elapsed {el:.0f} min', flush=True)
    print('Finished. Re-run the same command to retry any failed jobs.')


if __name__ == '__main__':
    main()
