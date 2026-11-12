"""
Adapted intervention baselines for the TCSS revision (Reviewer 1, comment 5).

Both baselines are *centrally chosen interventions* run inside the TEMPEST
environment (same graph, same contagion kernel `env_step`, same CONTAIN_T,
same seeds, same metrics). Ordinary users do not learn; by default they follow
their type's highest-payoff EGT action (Informed->Refute, Uninformed->Share,
Indifferent->Ignore), i.e. the same heterogeneous population as TEMPEST without
learning. See `background_actions` for the robustness variants.

  QDOM-adapted  -- He et al., "Reinforcement-learning-based dynamic opinion
                   maximization framework in signed social networks",
                   IEEE TCDS 15(1):54-64, 2023.
                   Q-learning chooses, for each of m seed placements, one of the
                   four seeding strategies from the paper (MaxDegree, Blocking,
                   Subgreedy, Mixstrategy). Seeds become permanent refuters.

  DRLE-adapted  -- Su et al., "Rumor Mitigation in Social Media Platforms with
                   Deep Reinforcement Learning", WWW '24 Companion.
                   A learned edge-scoring policy (node/edge centrality features,
                   rumour-source information, community information) removes a
                   budget of edges (10% of edges, <=40% of any node's edges, as in
                   the released config `facebook_w.yaml`). Trained with a clipped
                   policy-gradient update (PPO-style: clip 0.2, 4 epochs, lr 4e-4,
                   entropy 0.01 -- the released hyper-parameters).

Documented deviations from the originals (state these in the paper):
  * Both act at t = intervene_t (default CONTAIN_T) inside the TEMPEST kernel,
    instead of their native simulators (signed LT opinion model / plain SIR).
  * QDOM: "Blocking" and "Subgreedy" are not formally defined in the paper; we use
    Blocking = candidate with the most infected neighbours, Subgreedy = best of a
    random candidate subset under a one-step exposure-reduction proxy.
    The Q-update uses the episodic containment return (Monte-Carlo) because
    per-seed propagation rollouts are too costly at N = 5000.
  * DRLE: edges are selected in one batch (Plackett-Luce / Gumbel top-k) rather
    than one cut per environment step, because the original recomputes all
    centralities after every single cut, which is infeasible at N = 5000.
    Betweenness is approximated with k-sample estimates.
  * Training budget: both baselines get the same number of training episodes as
    TEMPEST (cfg['eps'], default 80).

R-state tagging: every I->R transition is tagged 'refuted' if the agent chose
Refute at that step, otherwise 'spontaneous'. Computed outside env_step, so the
RNG sequence of the kernel is unchanged.
"""
import math
import random

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

import marl_core as mc

IGNORE, SHARE, REFUTE = 0, 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
class _NbrView:
    """Minimal stand-in for GraphCache used by env_step (n, nbrs, node_types)."""
    def __init__(self, gc, nbrs):
        self.n = gc.n
        self.nbrs = nbrs
        self.node_types = gc.node_types


_TYPE_PRIOR_ACTION = mc._TYPE_PAYOFFS.argmax(axis=1).astype(np.int64)  # [Refute, Share, Ignore]


def background_actions(states, t, contain_t, mode, node_types):
    """Fixed, non-learned behaviour of ordinary users.

    type_prior (default): each user plays the action with the highest base EGT
        payoff for its type (Informed->Refute, Uninformed->Share,
        Indifferent->Ignore). Same heterogeneous population as TEMPEST, without
        learning; yields an outbreak of comparable size at contain_t.
    growth_share: susceptible users Share before contain_t, then Ignore.
    passive: everyone Ignores throughout (users as in the original papers).
    """
    if mode == 'type_prior':
        return _TYPE_PRIOR_ACTION[node_types].copy()
    a = np.zeros(len(states), dtype=np.int64)
    if mode == 'growth_share' and t < contain_t:
        a[states == 0] = SHARE
    return a


def tag_transitions(prev, new, actions, t, init_nodes, r_cause):
    """Tag I->R transitions (1 = chose Refute, 2 = spontaneous/other)."""
    eff = actions
    if init_nodes is not None and t < mc.PEAK_T:      # mirror env_step's override
        eff = actions.copy()
        for i in init_nodes:
            if prev[i] == 1:
                eff[i] = SHARE
    moved = np.where((prev == 1) & (new == 2))[0]
    r_cause[moved] = np.where(eff[moved] == REFUTE, 1, 2)


def summarise(hist, contain_t, eps=1e-9):
    I = np.asarray(hist['I'], float); R = np.asarray(hist['R'], float)
    s = (R - I) / (R + I + eps)
    out = {
        'peak_I': float(I.max()), 'final_R': float(R[-1]),
        'sustained_I': float(I[contain_t:].mean()),
        'sp': float(s.mean()), 'sp_growth': float(s[:contain_t].mean()),
        'sp_contain': float(s[contain_t:].mean()),
    }
    if 'R_refuted' in hist:
        Rr = np.asarray(hist['R_refuted'], float)
        sr = (Rr - I) / (Rr + I + eps)
        out.update({'final_R_refuted': float(Rr[-1]),
                    'final_R_spont': float(np.asarray(hist['R_spont'])[-1]),
                    'sp_ref': float(sr.mean()),
                    'sp_ref_growth': float(sr[:contain_t].mean()),
                    'sp_ref_contain': float(sr[contain_t:].mean())})
    return out


def run_episode(gc, max_time, contain_t, background, intervene_t=None,
                intervene=None, init_infected=mc.INIT_INFECTED):
    """Roll out the TEMPEST kernel with a background population and an optional
    one-shot intervention applied at t == intervene_t.

    intervene(states, view) -> (view, seeds) where `view` may carry modified
    neighbour lists (edge removal) and `seeds` is an int array of permanent
    refuters (may be empty).
    """
    old_ct = mc.CONTAIN_T
    mc.CONTAIN_T = contain_t
    try:
        n = gc.n
        states = np.zeros(n, dtype=np.int64)
        init_nodes = np.random.choice(n, init_infected, replace=False)
        states[init_nodes] = 1
        view = _NbrView(gc, gc.nbrs)
        seeds = np.zeros(0, dtype=np.int64)
        r_cause = np.zeros(n, dtype=np.int8)
        hist = {k: [] for k in ('S', 'I', 'R', 'R_refuted', 'R_spont')}
        for t in range(max_time):
            if intervene is not None and t == intervene_t:
                view, seeds = intervene(states, view)
                newly = seeds[states[seeds] != 2]
                r_cause[newly] = 1                     # debunkers count as refuted
                states[seeds] = 2
            actions = background_actions(states, t, contain_t, background, gc.node_types)
            if len(seeds):
                actions[seeds] = REFUTE
            new = mc.env_step(states, actions, view, t, init_nodes=init_nodes)
            tag_transitions(states, new, actions, t, init_nodes, r_cause)
            states = new
            hist['S'].append(int((states == 0).sum()))
            hist['I'].append(int((states == 1).sum()))
            hist['R'].append(int((states == 2).sum()))
            hist['R_refuted'].append(int(((states == 2) & (r_cause == 1)).sum()))
            hist['R_spont'].append(int(((states == 2) & (r_cause == 2)).sum()))
        return hist
    finally:
        mc.CONTAIN_T = old_ct


def average_hists(hists):
    keys = hists[0].keys()
    return {k: np.mean([h[k] for h in hists], axis=0).tolist() for k in keys}


def containment_return(hist, ref_sustained, contain_t):
    """Reward = relative reduction in mean infected after contain_t."""
    sus = float(np.mean(hist['I'][contain_t:]))
    return (ref_sustained - sus) / (ref_sustained + 1e-9)


def paired_episode(gc, max_time, contain_t, background, intervene_t, intervene):
    """Common-random-numbers training episode (variance reduction).

    Runs the same episode twice from the same RNG state: once without and once
    with the intervention. Up to the intervention both runs are identical, so the
    return isolates the intervention's own effect instead of outbreak-to-outbreak
    noise. Returns (hist_with_intervention, return).
    """
    s = np.random.randint(2**31 - 1)
    np.random.seed(s); random.seed(s)
    h0 = run_episode(gc, max_time, contain_t, background)
    np.random.seed(s); random.seed(s)
    h1 = run_episode(gc, max_time, contain_t, background, intervene_t, intervene)
    np.random.seed((s * 7919 + 1) % (2**31 - 1))       # move on to fresh randomness
    ref = float(np.mean(h0['I'][contain_t:]))
    return h1, containment_return(h1, ref, contain_t)


# ─────────────────────────────────────────────────────────────────────────────
# QDOM-adapted (ref 36)
# ─────────────────────────────────────────────────────────────────────────────
QDOM_STRATS = ('MaxDegree', 'Blocking', 'Subgreedy', 'Mixstrategy')


class QDOM:
    def __init__(self, gc, budget_frac=0.02, alpha=0.1, eps_start=0.9,
                 eps_end=0.05, eps_decay=0.95, n_bins=5, subgreedy_k=20):
        self.gc = gc
        self.m = max(1, int(round(budget_frac * gc.n)))
        self.alpha = alpha
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.n_bins = n_bins
        self.Q = np.zeros((n_bins, n_bins, len(QDOM_STRATS)))
        self.deg = np.array([len(x) for x in gc.nbrs])
        self.subgreedy_k = subgreedy_k

    def _state(self, k, states):
        sb = min(self.n_bins - 1, int(self.n_bins * k / self.m))
        ib = min(self.n_bins - 1, int(self.n_bins * (states == 1).mean() / 0.6))
        return sb, ib

    def _pick(self, strat, states, chosen):
        gc = self.gc
        avail = np.ones(gc.n, bool); avail[list(chosen)] = False
        if strat == 'Mixstrategy':
            strat = random.choice(QDOM_STRATS[:3])
        if strat == 'MaxDegree':
            score = self.deg.astype(float)
        elif strat == 'Blocking':
            inf = (states == 1)
            score = np.array([inf[nb].sum() if nb else 0 for nb in gc.nbrs], float)
        else:  # Subgreedy: best of a random candidate subset (one-step proxy)
            inf = (states == 1); sus = (states == 0)
            cand = np.random.choice(np.where(avail)[0],
                                    min(self.subgreedy_k, avail.sum()), replace=False)
            best, best_v = cand[0], -1.0
            for c in cand:
                nb = gc.nbrs[c]
                v = float(inf[nb].sum()) * (1.0 + float(sus[nb].sum())) if nb else 0.0
                if v > best_v:
                    best, best_v = c, v
            return int(best)
        score = np.where(avail, score + 1e-6 * np.random.rand(gc.n), -np.inf)
        return int(np.argmax(score))

    def select(self, states, greedy=False):
        chosen, traj = [], []
        for k in range(self.m):
            s = self._state(k, states)
            if not greedy and random.random() < self.eps:
                a = random.randrange(len(QDOM_STRATS))
            else:
                a = int(np.argmax(self.Q[s] + 1e-9 * np.random.rand(len(QDOM_STRATS))))
            chosen.append(self._pick(QDOM_STRATS[a], states, set(chosen)))
            traj.append((s, a))
        return np.array(chosen, dtype=np.int64), traj

    def update(self, traj, G):
        for s, a in traj:
            self.Q[s][a] += self.alpha * (G - self.Q[s][a])
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def strategy_usage(self):
        best = self.Q.argmax(axis=2).ravel()
        return {QDOM_STRATS[i]: int((best == i).sum()) for i in range(len(QDOM_STRATS))}


def train_qdom(gc, cfg, contain_t, intervene_t, background, ref_sustained):
    agent = QDOM(gc, budget_frac=cfg.get('qdom_budget', 0.02))
    for ep in range(cfg['eps']):
        box = {}
        def iv(states, view):
            seeds, traj = agent.select(states)
            box['traj'] = traj
            return view, seeds
        h, G = paired_episode(gc, cfg['max_time'], contain_t, background, intervene_t, iv)
        agent.update(box['traj'], G)
        if (ep + 1) % 20 == 0:
            print(f'  [QDOM] ep {ep+1}/{cfg["eps"]}  sustained_I={np.mean(h["I"][contain_t:]):.0f}  return={G:.3f}')
    return agent


def eval_qdom(agent, cfg, contain_t, intervene_t, background, n_evals):
    def iv(states, view):
        seeds, _ = agent.select(states, greedy=True)
        return view, seeds
    return average_hists([run_episode(agent.gc, cfg['max_time'], contain_t, background,
                                      intervene_t, iv) for _ in range(n_evals)])


# ─────────────────────────────────────────────────────────────────────────────
# DRLE-adapted (ref 37)
# ─────────────────────────────────────────────────────────────────────────────
def _norm(x):
    x = np.asarray(x, float); m = x.mean()
    return x / m if m > 0 else x


class EdgeFeatures:
    """Static node features computed once per graph (as in news.py
    `_cal_node_static`), dynamic source-dependent features per episode."""
    def __init__(self, gc, k_btw=256, seed=0):
        G = gc.G; n = gc.n
        self.gc = gc
        self.edges = np.array([(u, v) if u < v else (v, u) for u, v in G.edges()], dtype=np.int64)
        self.deg = _norm([G.degree(i) for i in range(n)])
        btw = nx.betweenness_centrality(G, k=min(k_btw, n), normalized=False, seed=seed)
        self.btw = _norm([btw[i] for i in range(n)])
        try:
            eig = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eig = {i: 0.0 for i in range(n)}
        self.eig = _norm([eig[i] for i in range(n)])
        clo = nx.harmonic_centrality(G, nbunch=None) if n <= 2000 else None
        if clo is None:      # sampled closeness for large graphs
            sample = np.random.RandomState(seed).choice(n, min(256, n), replace=False)
            acc = np.zeros(n)
            for s in sample:
                for v, d in nx.single_source_shortest_path_length(G, int(s)).items():
                    if d > 0: acc[v] += 1.0 / d
            clo = {i: acc[i] for i in range(n)}
        self.clo = _norm([clo[i] for i in range(n)])
        cl = nx.clustering(G)
        self.clu = _norm([cl[i] for i in range(n)])
        from networkx.algorithms.community import greedy_modularity_communities
        comm = np.zeros(n, dtype=np.int64)
        for c, nodes in enumerate(greedy_modularity_communities(G)):
            for v in nodes: comm[v] = c
        self.cross = (comm[self.edges[:, 0]] != comm[self.edges[:, 1]]).astype(float)
        self.e_index = {(int(u), int(v)): k for k, (u, v) in enumerate(self.edges)}
        self.dim = 2 * 9 + 4

    def dynamic(self, states, n_src=32):
        G = self.gc.G; n = self.gc.n
        inf = np.where(states == 1)[0]
        if len(inf) == 0:
            inf = np.array([0])
        src = np.random.choice(inf, min(n_src, len(inf)), replace=False).tolist()
        dist = np.full(n, n, float)
        for v, d in nx.multi_source_dijkstra_path_length(G, set(int(s) for s in inf)).items():
            dist[v] = d
        bs = nx.betweenness_centrality_subset(G, sources=src, targets=list(range(n)), normalized=False)
        ebs = nx.edge_betweenness_centrality_subset(G, sources=src, targets=list(range(n)), normalized=False)
        bs = _norm([bs[i] for i in range(n)])
        ebs_arr = np.zeros(len(self.edges))
        for (u, v), val in ebs.items():
            k = self.e_index.get((min(u, v), max(u, v)))
            if k is not None: ebs_arr[k] += val
        ebs_arr = _norm(ebs_arr)
        isI = (states == 1).astype(float); isS = (states == 0).astype(float)
        u, v = self.edges[:, 0], self.edges[:, 1]
        def pair(a):
            return [a[u], a[v]]
        cols = (pair(self.deg) + pair(self.btw) + pair(bs) + pair(self.eig) + pair(self.clo)
                + pair(self.clu) + pair(_norm(dist)) + pair(isI) + pair(isS)
                + [ebs_arr, self.cross, np.abs(self.deg[u] - self.deg[v]),
                   isI[u] * isS[v] + isS[u] * isI[v]])        # edge links an infected and a susceptible user
        X = np.stack(cols, axis=1).astype(np.float32)
        return torch.from_numpy(X)


class EdgePolicy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 32),
                                 nn.Tanh(), nn.Linear(32, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)


def _pl_logprob(scores, order):
    """Plackett-Luce log-probability of selecting `order` (in sequence)."""
    sel = scores[order]
    mask = torch.ones_like(scores, dtype=torch.bool); mask[order] = False
    rest = torch.logsumexp(scores[mask], dim=0) if mask.any() else torch.tensor(-1e9)
    # logsumexp over {sel[i:], rest} for each i, computed from the back
    rev = torch.flip(sel, [0])
    cum = torch.logcumsumexp(torch.cat([rest.view(1), rev]), dim=0)[1:]
    denom = torch.flip(cum, [0])
    return (sel - denom).sum()


class DRLE:
    def __init__(self, gc, budget_frac=0.10, node_cap=0.40, lr=4e-4, clip=0.2,
                 epochs=4, ent_coef=0.01, temperature=1.0):
        self.gc = gc
        self.F = EdgeFeatures(gc)
        self.pi = EdgePolicy(self.F.dim)
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.k = max(1, int(budget_frac * len(self.F.edges)))
        self.cap = np.array([max(1, int(node_cap * len(nb))) for nb in gc.nbrs])
        self.clip, self.epochs, self.ent = clip, epochs, ent_coef
        self.baseline = None
        self.temp = temperature

    def _choose(self, scores, greedy):
        s = scores.detach().numpy() / self.temp
        if not greedy:
            s = s + np.random.gumbel(size=len(s))
        used = np.zeros(self.gc.n, dtype=np.int64)
        order = []
        for e in np.argsort(-s):
            u, v = self.F.edges[e]
            if used[u] < self.cap[u] and used[v] < self.cap[v]:
                order.append(int(e)); used[u] += 1; used[v] += 1
                if len(order) == self.k: break
        return np.array(order, dtype=np.int64)

    def cut(self, states, view, greedy=False, store=None):
        X = self.F.dynamic(states)
        with torch.no_grad():
            scores = self.pi(X)
        order = self._choose(scores, greedy)
        cut = set(map(tuple, self.F.edges[order].tolist()))
        nbrs = [[w for w in nb if (min(i, w), max(i, w)) not in cut]
                for i, nb in enumerate(view.nbrs)]
        if store is not None:
            store['X'] = X; store['order'] = torch.from_numpy(order)
            with torch.no_grad():
                store['old_lp'] = _pl_logprob(self.pi(X) / self.temp, store['order'])
        return _NbrView(self.gc, nbrs), np.zeros(0, dtype=np.int64)

    def update(self, store, G):
        # running mean / variance of returns -> normalised advantage (as in PPO)
        if self.baseline is None:
            self.baseline, self.var = G, 1e-4
        adv = (G - self.baseline) / (math.sqrt(self.var) + 1e-3)
        self.var = 0.9 * self.var + 0.1 * (G - self.baseline) ** 2
        self.baseline = 0.9 * self.baseline + 0.1 * G
        for _ in range(self.epochs):
            scores = self.pi(store['X']) / self.temp
            lp = _pl_logprob(scores, store['order'])
            ratio = torch.exp(lp - store['old_lp'])
            obj = torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv)
            p = torch.softmax(scores, 0)
            ent = -(p * torch.log(p + 1e-12)).sum()
            loss = -(obj + self.ent * ent)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            self.opt.step()


class LinkCutIS:
    """Reference rule (not learned): with DRLE's budget and per-node cap, cut edges
    that connect an infected and a susceptible user first. Used to show the
    ceiling of link removal in this contagion kernel."""
    def __init__(self, gc, budget_frac=0.10, node_cap=0.40):
        self.gc = gc
        self.edges = np.array([(u, v) if u < v else (v, u) for u, v in gc.G.edges()], dtype=np.int64)
        self.k = max(1, int(budget_frac * len(self.edges)))
        self.cap = np.array([max(1, int(node_cap * len(nb))) for nb in gc.nbrs])

    def cut(self, states, view):
        a, b = states[self.edges[:, 0]], states[self.edges[:, 1]]
        score = (((a == 1) & (b == 0)) | ((a == 0) & (b == 1))).astype(float) + 1e-3 * np.random.rand(len(self.edges))
        used = np.zeros(self.gc.n, dtype=np.int64); cut = set()
        for e in np.argsort(-score):
            u, v = self.edges[e]
            if used[u] < self.cap[u] and used[v] < self.cap[v]:
                cut.add((int(u), int(v))); used[u] += 1; used[v] += 1
                if len(cut) == self.k: break
        nbrs = [[w for w in nb if (min(i, w), max(i, w)) not in cut] for i, nb in enumerate(view.nbrs)]
        return _NbrView(self.gc, nbrs), np.zeros(0, dtype=np.int64)


def train_drle(gc, cfg, contain_t, intervene_t, background, ref_sustained):
    agent = DRLE(gc, budget_frac=cfg.get('drle_budget', 0.10))
    for ep in range(cfg['eps']):
        store = {}
        def iv(states, view):
            return agent.cut(states, view, greedy=False, store=store)
        h, G = paired_episode(gc, cfg['max_time'], contain_t, background, intervene_t, iv)
        agent.update(store, G)
        if (ep + 1) % 20 == 0:
            print(f'  [DRLE] ep {ep+1}/{cfg["eps"]}  sustained_I={np.mean(h["I"][contain_t:]):.0f}  return={G:.3f}')
    return agent


def eval_drle(agent, cfg, contain_t, intervene_t, background, n_evals):
    def iv(states, view):
        return agent.cut(states, view, greedy=True)
    return average_hists([run_episode(agent.gc, cfg['max_time'], contain_t, background,
                                      intervene_t, iv) for _ in range(n_evals)])


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_baselines(seed, cfg, gc=None):
    """Train + evaluate QDOM-adapted and DRLE-adapted on one graph/seed.

    cfg keys: gtype, nodes, eps, max_time, n_evals, contain_t (default CONTAIN_T),
              intervene_t ('contain' or an int), background ('growth_share'|'passive'),
              qdom_budget, drle_budget, methods (list).
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    contain_t = cfg.get('contain_t', mc.CONTAIN_T)
    it = cfg.get('intervene_t', 'contain')
    intervene_t = contain_t if it == 'contain' else int(it)
    background = cfg.get('background', 'type_prior')
    n_evals = cfg.get('n_evals', 5)
    if gc is None:
        gc = mc.GraphCache(mc.make_graph(cfg['gtype'], cfg['nodes'], seed))

    h_none = average_hists([run_episode(gc, cfg['max_time'], contain_t, background)
                            for _ in range(n_evals)])
    ref = float(np.mean(h_none['I'][contain_t:]))
    out = {'seed': seed, 'contain_t': contain_t, 'intervene_t': intervene_t,
           'background': background, 'no_intervention': summarise(h_none, contain_t),
           'curves': {'no_intervention': h_none}}

    for name in cfg.get('methods', ['qdom', 'drle']):
        print(f'[seed={seed}] training {name} (intervene_t={intervene_t}, bg={background})')
        if name == 'linkcut_is':
            rule = LinkCutIS(gc, budget_frac=cfg.get('drle_budget', 0.10))
            h = average_hists([run_episode(gc, cfg['max_time'], contain_t, background, intervene_t, rule.cut)
                               for _ in range(n_evals)])
        elif name == 'qdom':
            agent = train_qdom(gc, cfg, contain_t, intervene_t, background, ref)
            h = eval_qdom(agent, cfg, contain_t, intervene_t, background, n_evals)
            out['qdom_strategy_usage'] = agent.strategy_usage()
            out['qdom_m'] = agent.m
        else:
            agent = train_drle(gc, cfg, contain_t, intervene_t, background, ref)
            h = eval_drle(agent, cfg, contain_t, intervene_t, background, n_evals)
            out['drle_k'] = agent.k
        out[name] = summarise(h, contain_t)
        out['curves'][name] = h
    return out
