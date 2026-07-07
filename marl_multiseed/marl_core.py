"""
Core MARL module — extracted from the notebook so multiple seeds can be
run in parallel without notebook overhead. Mirrors the v7 implementation
(proper GAT replay via ego subgraphs).

Exports:
    run_full_experiment(seed, cfg) -> dict of metrics
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Module-level defaults (overridable via cfg dict)
# ─────────────────────────────────────────────────────────────────────────────
LR             = 1e-3
GAMMA_RL       = 0.99
BATCH_SIZE     = 128
BUFFER_CAP     = 50_000
EPS_START      = 0.90
EPS_END        = 0.05
EPS_DECAY      = 0.993
TARGET_UPDATE  = 5

PEAK_T         = 20
GROWTH_K       = 0.30
FATIGUE_LAMBDA = 0.05
I_FLOOR        = 0.25
PEER_ALPHA     = 0.70
BASE_SPREAD    = 0.04
CONTAIN_T      = 30
REFUTE_PROB_CONTAIN = 0.40
REFUTE_PROB_GROWTH  = 0.10
REFUTE_CASCADE      = 0.12
SPONTANEOUS_RECOV   = 0.03

NODE_TYPE_PROBS = [0.30, 0.50, 0.20]
EGT_BASE_PAYOFFS = {
    'Informed':    {'Ignore': -0.10, 'Share': -0.30, 'Refute':  0.40},
    'Uninformed':  {'Ignore':  0.00, 'Share':  0.20, 'Refute':  0.10},
    'Indifferent': {'Ignore':  0.15, 'Share':  0.05, 'Refute':  0.05},
}
INIT_INFECTED  = 20
N_COMMUNITIES  = 4
IN_FEATS       = 11
HIDDEN         = 64
N_ACTIONS      = 3

_TYPE_NAMES = ['Informed', 'Uninformed', 'Indifferent']
_TYPE_PAYOFFS = np.array([
    [EGT_BASE_PAYOFFS[t][a] for a in ['Ignore', 'Share', 'Refute']]
    for t in _TYPE_NAMES
], dtype=np.float32)

Transition = namedtuple('Transition', ['s', 'a', 'r', 'ns', 'd', 'node_idx'])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def rumour_intensity(t):
    rise    = 1.0 / (1.0 + np.exp(-GROWTH_K * (t - PEAK_T)))
    fatigue = np.exp(-FATIGUE_LAMBDA * max(0, t - PEAK_T))
    return float(I_FLOOR + (1.0 - I_FLOOR) * rise * fatigue)


def assign_node_types(n):
    return np.random.choice(3, size=n, p=NODE_TYPE_PROBS).astype(np.int64)


def make_graph(gtype, n, seed):
    if gtype == 'ba':  return nx.barabasi_albert_graph(n, 3, seed=seed)
    if gtype == 'er':  return nx.erdos_renyi_graph(n, 6/n, seed=seed)
    if gtype == 'ws':  return nx.watts_strogatz_graph(n, k=6, p=0.1, seed=seed)
    if gtype == 'sbm':
        q = n // 4; sizes = [q, q, q, n - 3*q]
        probs = [[0.02,.002,.002,.002],[.002,.02,.002,.002],
                 [.002,.002,.02,.002],[.002,.002,.002,.02]]
        return nx.stochastic_block_model(sizes, probs, seed=seed)
    raise ValueError(gtype)


def build_edge_index(G):
    e = list(G.edges()); r = [(b, a) for a, b in e]
    if not e:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(list(zip(*(e + r))), dtype=torch.long)


def egt_payoff(node_types, actions, peer_ratio, intensity):
    n = len(node_types)
    base = _TYPE_PAYOFFS[node_types, actions]
    payoffs = base.copy()
    for i in range(n):
        t = node_types[i]; a = actions[i]
        if t == 0:
            if a == 1:   payoffs[i] -= 0.3 * intensity
            elif a == 2: payoffs[i] += 0.2 * peer_ratio[i]
        elif t == 1:
            if a == 1:   payoffs[i] += 0.3 * peer_ratio[i]
            elif a == 2: payoffs[i] += 0.05
        elif t == 2:
            payoffs[i] *= (1.0 - 0.5 * peer_ratio[i])
    return payoffs


class GraphCache:
    def __init__(self, G):
        self.G    = G
        self.n    = G.number_of_nodes()
        self.ei   = build_edge_index(G)
        pr_dict   = nx.pagerank(G)
        self.pr   = np.array([pr_dict[i]        for i in range(self.n)], np.float32)
        self.deg  = np.array([G.degree(i)/self.n for i in range(self.n)], np.float32)
        self.nbrs = [list(G.neighbors(i))        for i in range(self.n)]
        self.node_types = assign_node_types(self.n)

        # Ego-subgraph cache for proper GAT replay
        self.ego_edges = []
        self.ego_nodes = []
        for i in range(self.n):
            nbrs_i = self.nbrs[i]
            ego_n  = [i] + list(nbrs_i)
            local  = {orig: k for k, orig in enumerate(ego_n)}
            if len(nbrs_i) > 0:
                ei = []
                for nb in nbrs_i:
                    ei.append([local[i],  local[nb]])
                    ei.append([local[nb], local[i]])
                ei_t = torch.tensor(ei, dtype=torch.long).T
            else:
                ei_t = torch.zeros(2, 0, dtype=torch.long)
            self.ego_edges.append(ei_t)
            self.ego_nodes.append(np.asarray(ego_n, dtype=np.int64))

    def features(self, states, t):
        n = self.n
        intensity = rumour_intensity(t)
        ir = np.array([sum(states[j]==1 for j in self.nbrs[i]) / (len(self.nbrs[i])+1)
                       for i in range(n)], np.float32)
        rr = np.array([sum(states[j]==2 for j in self.nbrs[i]) / (len(self.nbrs[i])+1)
                       for i in range(n)], np.float32)
        sr = np.array([sum(states[j]==0 for j in self.nbrs[i]) / (len(self.nbrs[i])+1)
                       for i in range(n)], np.float32)
        nm = np.array([float(np.mean([states[j] for j in self.nbrs[i]]))
                       if self.nbrs[i] else 0. for i in range(n)], np.float32)
        return torch.tensor(np.stack([
            states.astype(np.float32),
            self.deg, ir,
            np.full(n, intensity, np.float32),
            self.pr, nm, rr, sr, ir,
            np.full(n, intensity, np.float32),
            self.node_types.astype(np.float32) / 2.0,
        ], axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────
class AgentGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat    = GATConv(IN_FEATS, HIDDEN // 2, heads=2)
        self.gru    = nn.GRUCell(HIDDEN, HIDDEN)
        self.q_head = nn.Linear(HIDDEN, N_ACTIONS)

    def forward(self, x, edge_index, h):
        x_out  = F.elu(self.gat(x, edge_index))
        h_next = self.gru(x_out, h)
        return self.q_head(h_next), h_next


class ReplayBuffer:
    def __init__(self, cap=BUFFER_CAP):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, ns, d, node_idx):
        self.buf.append(Transition(
            np.asarray(s, np.float32), int(a), float(r),
            np.asarray(ns, np.float32), float(d), int(node_idx)
        ))

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


class MARLSystem:
    def __init__(self, gc):
        self.gc = gc
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(gc.G))
        self.node_community = np.zeros(gc.n, dtype=np.int64)
        for c_idx, community in enumerate(communities):
            for node in community:
                self.node_community[node] = min(c_idx, N_COMMUNITIES - 1)

        self.online  = [AgentGNN() for _ in range(N_COMMUNITIES)]
        self.target  = [AgentGNN() for _ in range(N_COMMUNITIES)]
        self.opts    = [torch.optim.Adam(net.parameters(), lr=LR) for net in self.online]
        self.buffers = [ReplayBuffer() for _ in range(N_COMMUNITIES)]
        for i in range(N_COMMUNITIES):
            self.target[i].load_state_dict(self.online[i].state_dict())
            self.target[i].eval()
        self.community_masks = [
            np.where(self.node_community == c)[0] for c in range(N_COMMUNITIES)
        ]

    def select_actions(self, x, h, eps):
        actions = np.zeros(self.gc.n, dtype=np.int64)
        h_new   = torch.zeros_like(h)
        with torch.no_grad():
            for c in range(N_COMMUNITIES):
                mask = self.community_masks[c]
                if len(mask) == 0: continue
                q_all, h_all = self.online[c](x, self.gc.ei, h)
                q_c = q_all[mask]
                greedy  = q_c.argmax(dim=1).numpy()
                explore = np.random.rand(len(mask)) < eps
                actions[mask] = np.where(explore,
                    np.random.randint(0, N_ACTIONS, len(mask)), greedy)
                h_new[mask] = h_all[mask]
        return actions, h_new

    def _build_ego_batch(self, batch_transitions, use_next_state=False):
        feats, edges, centers = [], [], []
        offset = 0
        for tr in batch_transitions:
            i        = tr.node_idx
            ego_n    = self.gc.ego_nodes[i]
            ei_local = self.gc.ego_edges[i]
            k        = len(ego_n)
            ego_x = np.zeros((k, IN_FEATS), dtype=np.float32)
            ego_x[0] = tr.ns if use_next_state else tr.s
            for k_local in range(1, k):
                orig = ego_n[k_local]
                ego_x[k_local, 1]  = self.gc.deg[orig]
                ego_x[k_local, 4]  = self.gc.pr[orig]
                ego_x[k_local, 10] = self.gc.node_types[orig] / 2.0
            feats.append(torch.from_numpy(ego_x))
            edges.append(ei_local + offset)
            centers.append(offset)
            offset += k
        x_batch  = torch.cat(feats,  dim=0)
        ei_batch = torch.cat(edges,  dim=1) if edges else torch.zeros(2, 0, dtype=torch.long)
        centers  = torch.tensor(centers, dtype=torch.long)
        return x_batch, ei_batch, centers

    def update(self, c):
        buf = self.buffers[c]
        if len(buf) < BATCH_SIZE: return 0.0
        batch = buf.sample(BATCH_SIZE)
        a_b = torch.tensor([b.a for b in batch], dtype=torch.long)
        r_b = torch.tensor([b.r for b in batch], dtype=torch.float32)
        d_b = torch.tensor([b.d for b in batch], dtype=torch.float32)
        net_c = self.online[c]; tgt_c = self.target[c]
        x_s,  ei_s,  cen_s  = self._build_ego_batch(batch, use_next_state=False)
        x_ns, ei_ns, cen_ns = self._build_ego_batch(batch, use_next_state=True)
        h_zero_s  = torch.zeros(x_s.size(0),  HIDDEN)
        h_zero_ns = torch.zeros(x_ns.size(0), HIDDEN)
        q_all_s, _ = net_c(x_s, ei_s, h_zero_s)
        q_pred = q_all_s[cen_s].gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_all_ns, _ = tgt_c(x_ns, ei_ns, h_zero_ns)
            q_next = q_all_ns[cen_ns].max(1)[0]
        q_target = r_b + GAMMA_RL * q_next * (1.0 - d_b)
        loss = F.smooth_l1_loss(q_pred, q_target)
        self.opts[c].zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online[c].parameters(), 1.0)
        self.opts[c].step()
        return loss.item()

    def sync_targets(self):
        for c in range(N_COMMUNITIES):
            self.target[c].load_state_dict(self.online[c].state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
def env_step(states, actions, gc, t, init_nodes=None):
    intensity = rumour_intensity(t)
    new = states.copy()
    node_types = gc.node_types
    in_contain = (t >= CONTAIN_T)
    if init_nodes is not None and t < PEAK_T:
        actions = actions.copy()
        for i in init_nodes:
            if states[i] == 1:
                actions[i] = 1
    TYPE_SUSCEPT = np.array([0.55, 1.00, 0.25], dtype=np.float32)
    refuting_R_nbrs = np.array([
        sum(1 for j in gc.nbrs[i] if states[j] == 2 and actions[j] == 2)
        for i in range(gc.n)
    ], dtype=np.float32)
    for i in range(gc.n):
        nbrs_i = gc.nbrs[i]; n_nbrs = len(nbrs_i) + 1
        suscep = TYPE_SUSCEPT[node_types[i]]
        if states[i] == 0:
            infected_ratio = sum(states[j] == 1 for j in nbrs_i) / n_nbrs
            p_peer   = PEER_ALPHA * infected_ratio * intensity * suscep
            p_spread = BASE_SPREAD * suscep + p_peer
            if actions[i] == 2 and in_contain:
                p_spread *= 0.35
            if actions[i] == 1 and np.random.rand() < p_spread:
                new[i] = 1
            elif actions[i] != 1 and np.random.rand() < p_peer * 0.4:
                new[i] = 1
        elif states[i] == 1:
            if in_contain:
                p_refute = REFUTE_PROB_CONTAIN if actions[i] == 2 else SPONTANEOUS_RECOV
                p_refute = min(0.95, p_refute + REFUTE_CASCADE * refuting_R_nbrs[i] / n_nbrs)
            else:
                p_refute = REFUTE_PROB_GROWTH if (actions[i] == 2 and t > 5) else SPONTANEOUS_RECOV
            if np.random.rand() < p_refute:
                new[i] = 2
    for i in range(gc.n):
        if states[i] == 0 and new[i] == 0:
            suscep = TYPE_SUSCEPT[node_types[i]]
            resist = 0.35 if (actions[i] == 2 and in_contain) else 1.0
            for nb in gc.nbrs[i]:
                if states[nb] == 1 and np.random.rand() < BASE_SPREAD * intensity * suscep * resist:
                    new[i] = 1; break
    return new


def reward_fn(states, actions, gc, t, flat=False):
    intensity = rumour_intensity(t)
    r = np.zeros(gc.n, np.float32)
    node_types = gc.node_types
    in_contain = (t >= CONTAIN_T)
    if flat:
        for i in range(gc.n):
            if   states[i]==1 and actions[i]==2: r[i] =  1.0
            elif states[i]==0 and actions[i]==1: r[i] = -1.0
        return r
    pr = np.array([sum(states[j]==1 for j in gc.nbrs[i]) / (len(gc.nbrs[i])+1)
                   for i in range(gc.n)], np.float32)
    egt = egt_payoff(node_types, actions, pr, intensity)
    for i in range(gc.n):
        a = actions[i]
        if states[i] == 0:
            if not in_contain: r[i] = 0.15 if a == 1 else 0.0
            else:
                if   a == 1: r[i] = -2.0 * intensity
                elif a == 2: r[i] =  0.5 + 0.4 * (1.0 - pr[i])
                else:        r[i] =  0.1
        elif states[i] == 1:
            if not in_contain:
                r[i] = 0.4 * intensity if a == 1 else (-0.1 if a == 2 else 0.0)
            else:
                if   a == 2: r[i] =  4.0
                elif a == 1: r[i] = -2.0 * intensity * pr[i]
                else:        r[i] = -0.3
        else:
            r[i] = 0.3 if (a == 2 and in_contain) else 0.05
        r[i] *= 1.0 + 0.5 * egt[i]
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train_marl(gc, n_eps, max_time, init_infected=INIT_INFECTED):
    marl = MARLSystem(gc)
    eps = EPS_START
    SUBSAMPLE = max(1, gc.n // 50)
    for ep in range(n_eps):
        states = np.zeros(gc.n, dtype=np.int64)
        init_nodes = np.random.choice(gc.n, init_infected, replace=False)
        states[init_nodes] = 1
        h = torch.zeros(gc.n, HIDDEN)
        for t in range(max_time):
            x = gc.features(states, t)
            actions, h_new = marl.select_actions(x, h, eps)
            new_states = env_step(states, actions, gc, t, init_nodes=init_nodes)
            rewards    = reward_fn(states, actions, gc, t)
            x_next     = gc.features(new_states, t + 1)
            done = float(t == max_time - 1)
            for i in range(0, gc.n, SUBSAMPLE):
                c = marl.node_community[i]
                marl.buffers[c].push(x[i].numpy(), actions[i], rewards[i],
                                     x_next[i].numpy(), done, node_idx=i)
            for c in range(N_COMMUNITIES):
                marl.update(c)
            states = new_states; h = h_new.detach()
        if (ep + 1) % TARGET_UPDATE == 0:
            marl.sync_targets()
        eps = max(EPS_END, eps * EPS_DECAY)
        if (ep + 1) % 10 == 0:
            print(f'  ep {ep+1}/{n_eps}  final_I={int(np.sum(states==1))}  eps={eps:.3f}')
    return marl


def eval_marl_avg(marl, max_time, n_evals=5, init_infected=INIT_INFECTED):
    """
    Average MARL evaluation over n_evals rollouts with different init-node sets.
    Reduces evaluation variance so the per-seed numbers reflect the trained
    policy's quality rather than which 20 nodes happened to be the seed set.

    Returns:
        dict with keys S, I, R (each a list of mean values per timestep)
        plus 'std_I', 'std_R' for the per-timestep dispersion across rollouts.
    """
    rollouts = [eval_marl(marl, max_time, init_infected) for _ in range(n_evals)]
    I_stack = np.array([r['I'] for r in rollouts])   # [n_evals, T]
    R_stack = np.array([r['R'] for r in rollouts])
    S_stack = np.array([r['S'] for r in rollouts])
    return {
        'S':     S_stack.mean(axis=0).tolist(),
        'I':     I_stack.mean(axis=0).tolist(),
        'R':     R_stack.mean(axis=0).tolist(),
        'std_I': I_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'std_R': R_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'n_evals': n_evals,
    }


def run_sir_avg(gc, max_time, n_evals=5, beta=0.2, gamma=0.02, init_infected=INIT_INFECTED):
    """SIR baseline averaged over n_evals rollouts for a fair comparison."""
    rollouts = [run_sir(gc, max_time, beta, gamma, init_infected) for _ in range(n_evals)]
    I_stack = np.array([r['I'] for r in rollouts])
    R_stack = np.array([r['R'] for r in rollouts])
    S_stack = np.array([r['S'] for r in rollouts])
    return {
        'S':     S_stack.mean(axis=0).tolist(),
        'I':     I_stack.mean(axis=0).tolist(),
        'R':     R_stack.mean(axis=0).tolist(),
        'std_I': I_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'std_R': R_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'n_evals': n_evals,
    }


def eval_marl(marl, max_time, init_infected=INIT_INFECTED):
    gc = marl.gc
    states = np.zeros(gc.n, dtype=np.int64)
    init_nodes = np.random.choice(gc.n, init_infected, replace=False)
    states[init_nodes] = 1
    h = torch.zeros(gc.n, HIDDEN)
    hist = {'S': [], 'I': [], 'R': []}
    for net in marl.online: net.eval()
    with torch.no_grad():
        for t in range(max_time):
            x = gc.features(states, t)
            actions, h = marl.select_actions(x, h, eps=0.0)
            states = env_step(states, actions, gc, t, init_nodes=init_nodes)
            hist['S'].append(int(np.sum(states==0)))
            hist['I'].append(int(np.sum(states==1)))
            hist['R'].append(int(np.sum(states==2)))
    return hist


def run_sir(gc, max_time, beta=0.2, gamma=0.02, init_infected=INIT_INFECTED):
    n = gc.n
    states = np.zeros(n, dtype=np.int64)
    states[np.random.choice(n, init_infected, replace=False)] = 1
    hist = {'S':[], 'I':[], 'R':[]}
    for t in range(max_time):
        intensity = rumour_intensity(t)
        new = states.copy()
        for u in np.where(states==1)[0]:
            for v in gc.nbrs[u]:
                if states[v]==0 and np.random.rand() < beta * intensity: new[v]=1
            if np.random.rand() < gamma: new[u]=2
        states = new
        hist['S'].append(int(np.sum(states==0)))
        hist['I'].append(int(np.sum(states==1)))
        hist['R'].append(int(np.sum(states==2)))
    return hist



def eval_greedy_policy(gc, max_time, init_infected=INIT_INFECTED):
    """
    Hand-crafted rule-based baseline. No learning, no Q-networks.

    Decision rules:
    - Growth phase (t < CONTAIN_T):
        * Susceptible + top-20% PageRank  -> Share
        * Infected + t > 5               -> Refute (early recovery attempt)
        * Otherwise                       -> Ignore
    - Containment phase (t >= CONTAIN_T):
        * Infected                        -> Refute
        * Susceptible + any infected nbr  -> Refute (social inoculation)
        * Otherwise                       -> Ignore

    Returns: dict with S/I/R curves, same format as eval_marl().
    """
    states = np.zeros(gc.n, dtype=np.int64)
    init_nodes = np.random.choice(gc.n, init_infected, replace=False)
    states[init_nodes] = 1

    pr_threshold = np.sort(gc.pr)[int(gc.n * 0.80)]  # top-20% PR cutoff
    hist = {'S': [], 'I': [], 'R': []}

    for t in range(max_time):
        in_contain = (t >= CONTAIN_T)
        actions = np.zeros(gc.n, dtype=np.int64)

        for i in range(gc.n):
            if in_contain:
                if states[i] == 1:
                    actions[i] = 2
                elif states[i] == 0 and any(states[j] == 1 for j in gc.nbrs[i]):
                    actions[i] = 2
            else:
                if states[i] == 0 and gc.pr[i] >= pr_threshold:
                    actions[i] = 1
                elif states[i] == 1 and t > 5:
                    actions[i] = 2

        states = env_step(states, actions, gc, t, init_nodes=init_nodes)
        hist['S'].append(int(np.sum(states == 0)))
        hist['I'].append(int(np.sum(states == 1)))
        hist['R'].append(int(np.sum(states == 2)))

    return hist


def run_greedy_avg(gc, max_time, n_evals=5, init_infected=INIT_INFECTED):
    """Greedy heuristic averaged over n_evals independent rollouts."""
    rollouts = [eval_greedy_policy(gc, max_time, init_infected) for _ in range(n_evals)]
    I_stack = np.array([r['I'] for r in rollouts])
    R_stack = np.array([r['R'] for r in rollouts])
    S_stack = np.array([r['S'] for r in rollouts])
    return {
        'S':     S_stack.mean(axis=0).tolist(),
        'I':     I_stack.mean(axis=0).tolist(),
        'R':     R_stack.mean(axis=0).tolist(),
        'std_I': I_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'std_R': R_stack.std(axis=0, ddof=1).tolist() if n_evals > 1 else [0.0]*max_time,
        'n_evals': n_evals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Centralized DQN baseline
# ─────────────────────────────────────────────────────────────────────────────
class CentralizedMARL:
    """
    Single-network DQN baseline.  One Q-network shared across all nodes,
    no community partitioning.  Same AgentGNN (GAT+GRU+Q-head) architecture
    and reward function as TEMPEST; only the policy structure differs.
    """
    def __init__(self, gc):
        self.gc     = gc
        self.online = AgentGNN()
        self.target = AgentGNN()
        self.opt    = torch.optim.Adam(self.online.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def select_actions(self, x, h, eps):
        with torch.no_grad():
            q_all, h_new = self.online(x, self.gc.ei, h)
        greedy  = q_all.argmax(dim=1).numpy()
        explore = np.random.rand(self.gc.n) < eps
        actions = np.where(explore,
            np.random.randint(0, N_ACTIONS, self.gc.n), greedy)
        return actions, h_new

    def _build_ego_batch(self, batch_transitions, use_next_state=False):
        """Identical to MARLSystem._build_ego_batch — ego subgraph replay."""
        feats, edges, centers = [], [], []
        offset = 0
        for tr in batch_transitions:
            i        = tr.node_idx
            ego_n    = self.gc.ego_nodes[i]
            ei_local = self.gc.ego_edges[i]
            k        = len(ego_n)
            ego_x    = np.zeros((k, IN_FEATS), dtype=np.float32)
            ego_x[0] = tr.ns if use_next_state else tr.s
            for k_local in range(1, k):
                orig = ego_n[k_local]
                ego_x[k_local, 1]  = self.gc.deg[orig]
                ego_x[k_local, 4]  = self.gc.pr[orig]
                ego_x[k_local, 10] = self.gc.node_types[orig] / 2.0
            feats.append(torch.from_numpy(ego_x))
            edges.append(ei_local + offset)
            centers.append(offset)
            offset += k
        x_batch  = torch.cat(feats, dim=0)
        ei_batch = (torch.cat(edges, dim=1) if edges
                    else torch.zeros(2, 0, dtype=torch.long))
        centers  = torch.tensor(centers, dtype=torch.long)
        return x_batch, ei_batch, centers

    def update(self):
        if len(self.buffer) < BATCH_SIZE: return 0.0
        batch = self.buffer.sample(BATCH_SIZE)
        a_b = torch.tensor([b.a for b in batch], dtype=torch.long)
        r_b = torch.tensor([b.r for b in batch], dtype=torch.float32)
        d_b = torch.tensor([b.d for b in batch], dtype=torch.float32)
        x_s,  ei_s,  cen_s  = self._build_ego_batch(batch, use_next_state=False)
        x_ns, ei_ns, cen_ns = self._build_ego_batch(batch, use_next_state=True)
        h_zero_s  = torch.zeros(x_s.size(0),  HIDDEN)
        h_zero_ns = torch.zeros(x_ns.size(0), HIDDEN)
        q_all_s, _ = self.online(x_s, ei_s, h_zero_s)
        q_pred = q_all_s[cen_s].gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_all_ns, _ = self.target(x_ns, ei_ns, h_zero_ns)
            q_next = q_all_ns[cen_ns].max(1)[0]
        q_target = r_b + GAMMA_RL * q_next * (1.0 - d_b)
        loss = F.smooth_l1_loss(q_pred, q_target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def sync_targets(self):
        self.target.load_state_dict(self.online.state_dict())


def train_centralized(gc, n_eps, max_time, n_updates_per_step=4,
                      init_infected=INIT_INFECTED):
    """
    Train centralized DQN with fair-compute parity to TEMPEST.

    TEMPEST has K=4 networks each updated once per step → 4 gradient steps
    per env step.  We replicate that total by updating the single network
    n_updates_per_step=4 times per step so the comparison is purely
    architectural (1 network vs K community-partitioned networks).
    """
    cmarl = CentralizedMARL(gc)
    eps = EPS_START
    SUBSAMPLE = max(1, gc.n // 50)
    for ep in range(n_eps):
        states = np.zeros(gc.n, dtype=np.int64)
        init_nodes = np.random.choice(gc.n, init_infected, replace=False)
        states[init_nodes] = 1
        h = torch.zeros(gc.n, HIDDEN)
        for t in range(max_time):
            x = gc.features(states, t)
            actions, h_new = cmarl.select_actions(x, h, eps)
            new_states = env_step(states, actions, gc, t, init_nodes=init_nodes)
            rewards    = reward_fn(states, actions, gc, t)
            x_next     = gc.features(new_states, t + 1)
            done = float(t == max_time - 1)
            for i in range(0, gc.n, SUBSAMPLE):
                cmarl.buffer.push(x[i].numpy(), actions[i], rewards[i],
                                  x_next[i].numpy(), done, node_idx=i)
            for _ in range(n_updates_per_step):
                cmarl.update()
            states = new_states; h = h_new.detach()
        if (ep + 1) % TARGET_UPDATE == 0:
            cmarl.sync_targets()
        eps = max(EPS_END, eps * EPS_DECAY)
        if (ep + 1) % 10 == 0:
            print(f'  [centralized] ep {ep+1}/{n_eps}'
                  f'  final_I={int(np.sum(states==1))}  eps={eps:.3f}')
    return cmarl


def eval_centralized(cmarl, max_time, init_infected=INIT_INFECTED):
    gc = cmarl.gc
    states = np.zeros(gc.n, dtype=np.int64)
    init_nodes = np.random.choice(gc.n, init_infected, replace=False)
    states[init_nodes] = 1
    h = torch.zeros(gc.n, HIDDEN)
    hist = {'S': [], 'I': [], 'R': []}
    cmarl.online.eval()
    with torch.no_grad():
        for t in range(max_time):
            x = gc.features(states, t)
            actions, h = cmarl.select_actions(x, h, eps=0.0)
            states = env_step(states, actions, gc, t, init_nodes=init_nodes)
            hist['S'].append(int(np.sum(states == 0)))
            hist['I'].append(int(np.sum(states == 1)))
            hist['R'].append(int(np.sum(states == 2)))
    return hist


def eval_centralized_avg(cmarl, max_time, n_evals=5, init_infected=INIT_INFECTED):
    rollouts = [eval_centralized(cmarl, max_time, init_infected)
                for _ in range(n_evals)]
    I_stack = np.array([r['I'] for r in rollouts])
    R_stack = np.array([r['R'] for r in rollouts])
    S_stack = np.array([r['S'] for r in rollouts])
    return {
        'S':     S_stack.mean(axis=0).tolist(),
        'I':     I_stack.mean(axis=0).tolist(),
        'R':     R_stack.mean(axis=0).tolist(),
        'std_I': (I_stack.std(axis=0, ddof=1).tolist() if n_evals > 1
                  else [0.0] * max_time),
        'std_R': (R_stack.std(axis=0, ddof=1).tolist() if n_evals > 1
                  else [0.0] * max_time),
        'n_evals': n_evals,
    }


def simulate_multi_rumour(gc, marl_system, n_rumours, max_time,
                          detection_threshold=0.05):
    n = gc.n
    out = {'n_generated': n_rumours, 'n_detected': 0,
           'detection_times': [], 'peak_I_list': [], 'final_R_list': []}
    for _ in range(n_rumours):
        states = np.zeros(n, dtype=np.int64)
        init_nodes = np.random.choice(n, INIT_INFECTED, replace=False)
        states[init_nodes] = 1
        h = torch.zeros(n, HIDDEN)
        if marl_system is not None:
            for net in marl_system.online: net.eval()
        I_curve = []; detected = False; detect_t = None
        with torch.no_grad():
            for t in range(max_time):
                if marl_system is not None:
                    x = gc.features(states, t)
                    actions, h = marl_system.select_actions(x, h, eps=0.0)
                    states = env_step(states, actions, gc, t, init_nodes=init_nodes)
                else:
                    intensity = rumour_intensity(t)
                    new = states.copy()
                    for u in np.where(states==1)[0]:
                        for v in gc.nbrs[u]:
                            if states[v]==0 and np.random.rand() < 0.2 * intensity:
                                new[v] = 1
                        if np.random.rand() < 0.05:
                            new[u] = 2
                    states = new
                n_I = int(np.sum(states == 1)); n_R = int(np.sum(states == 2))
                I_curve.append(n_I)
                if not detected and (n_R / n) >= detection_threshold:
                    detected = True; detect_t = t
                    out['n_detected'] += 1
        out['detection_times'].append(detect_t)
        out['peak_I_list'].append(int(max(I_curve)))
        out['final_R_list'].append(int(np.sum(states == 2)))
    out['detection_rate'] = out['n_detected'] / n_rumours
    out['mean_detect_t']  = float(np.mean(
        [t for t in out['detection_times'] if t is not None] or [max_time]))
    out['mean_peak_I']    = float(np.mean(out['peak_I_list']))
    out['mean_final_R']   = float(np.mean(out['final_R_list']))
    return out


def influential_efficiency(gc, hist_marl, hist_sir, top_k_frac=0.10):
    n = gc.n
    top_k = max(1, int(n * top_k_frac))
    pr_sorted     = np.sort(gc.pr)[::-1]
    pr_top_weight = pr_sorted[:top_k].sum() / (gc.pr.sum() + 1e-9)
    I_m = np.array(hist_marl['I']); R_m = np.array(hist_marl['R'])
    I_s = np.array(hist_sir['I']);  R_s = np.array(hist_sir['R'])
    if len(R_m) <= CONTAIN_T:
        # Edge case: simulation didn't reach containment phase — IE undefined
        return 0.0, 0.0, 0.0
    marl_rate = np.mean((R_m[CONTAIN_T:] / (R_m[CONTAIN_T:] + I_m[CONTAIN_T:] + 1e-9)) * pr_top_weight)
    sir_rate  = np.mean((R_s[CONTAIN_T:] / (R_s[CONTAIN_T:] + I_s[CONTAIN_T:] + 1e-9)) * pr_top_weight)
    IE = (marl_rate - sir_rate) / (sir_rate + 1e-9)
    return float(IE), float(marl_rate), float(sir_rate)


def run_full_experiment(seed, cfg):
    """
    Entry point called per-seed by the parallel runner.
    Returns a dict of metrics ready to be JSON-dumped.
    """
    print(f'[seed={seed}] making graph: gtype={cfg["gtype"]} n={cfg["nodes"]}')
    G  = make_graph(cfg['gtype'], cfg['nodes'], seed)
    gc = GraphCache(G)

    print(f'[seed={seed}] training MARL: {cfg["eps"]} eps × {cfg["max_time"]} steps')
    marl = train_marl(gc, n_eps=cfg['eps'], max_time=cfg['max_time'])

    n_evals = cfg.get('n_evals', 5)
    print(f'[seed={seed}] evaluating MARL + SIR baseline ({n_evals} rollouts each, averaged)')
    h_marl = eval_marl_avg(marl, max_time=cfg['max_time'], n_evals=n_evals)
    h_sir  = run_sir_avg(gc,  max_time=cfg['max_time'], n_evals=n_evals)

    print(f'[seed={seed}] running {cfg["n_rumours"]} rumour outbreaks (SIR & MARL)')
    res_sir  = simulate_multi_rumour(gc, None, cfg['n_rumours'], cfg['max_time'])
    res_marl = simulate_multi_rumour(gc, marl, cfg['n_rumours'], cfg['max_time'])

    # Scalar metrics
    I_s = np.array(h_sir['I']);  R_s = np.array(h_sir['R'])
    I_m = np.array(h_marl['I']); R_m = np.array(h_marl['R'])
    t_cut = min(CONTAIN_T, cfg['max_time'])

    IE, marl_rate, sir_rate = influential_efficiency(gc, h_marl, h_sir)

    print(f'[seed={seed}] evaluating Greedy heuristic baseline')
    np.random.seed(seed * 1000 + 7)
    h_greedy = run_greedy_avg(gc, max_time=cfg['max_time'], n_evals=n_evals)
    I_g = np.array(h_greedy['I']); R_g = np.array(h_greedy['R'])

    print(f'[seed={seed}] training Centralized DQN ({cfg["eps"]} eps, 4× updates/step)')
    np.random.seed(seed * 1000 + 17)
    random.seed(seed * 1000 + 17)
    torch.manual_seed(seed * 1000 + 17)
    cmarl = train_centralized(gc, n_eps=cfg['eps'], max_time=cfg['max_time'])

    print(f'[seed={seed}] evaluating Centralized DQN ({n_evals} rollouts)')
    h_cen = eval_centralized_avg(cmarl, max_time=cfg['max_time'], n_evals=n_evals)
    I_c = np.array(h_cen['I']); R_c = np.array(h_cen['R'])

    return {
        'spread_reduction'       : float((I_s.max() - I_m.max()) / (I_s.max() + 1e-9)),
        'peak_I_sir'             : int(I_s.max()),
        'peak_I_marl'            : int(I_m.max()),
        'truth_amp_sir'          : float(R_s[:t_cut].mean() / (I_s[:t_cut].mean() + 1e-9)),
        'truth_amp_marl'         : float(R_m[:t_cut].mean() / (I_m[:t_cut].mean() + 1e-9)),
        'polarization_sir'       : float(np.mean(np.abs(I_s-R_s)/(I_s+R_s+1e-9))),
        'polarization_marl'      : float(np.mean(np.abs(I_m-R_m)/(I_m+R_m+1e-9))),
        'influential_efficiency' : IE,
        'hub_recov_sir'          : sir_rate,
        'hub_recov_marl'         : marl_rate,
        'rumours_generated'           : res_sir['n_generated'],
        'rumours_detected_sir'        : res_sir['n_detected'],
        'rumours_detected_marl'       : res_marl['n_detected'],
        'detection_rate_sir'          : res_sir['detection_rate'],
        'detection_rate_marl'         : res_marl['detection_rate'],
        'mean_detect_t_sir'           : res_sir['mean_detect_t'],
        'mean_detect_t_marl'          : res_marl['mean_detect_t'],
        'mean_peak_I_per_rumour_sir'  : res_sir['mean_peak_I'],
        'mean_peak_I_per_rumour_marl' : res_marl['mean_peak_I'],
        'final_R_sir'  : int(R_s[-1]),
        'final_R_marl' : int(R_m[-1]),
        'peak_I_greedy'      : int(I_g.max()),
        'final_R_greedy'     : int(R_g[-1]),
        'truth_amp_greedy'   : float(R_g[:t_cut].mean() / (I_g[:t_cut].mean() + 1e-9)),
        'polarization_greedy': float(np.mean(np.abs(I_g - R_g) / (I_g + R_g + 1e-9))),
        'spread_red_greedy'  : float((I_s.max() - I_g.max()) / (I_s.max() + 1e-9)),
        'I_curve_greedy'     : list(map(int, I_g.tolist())),
        'R_curve_greedy'     : list(map(int, R_g.tolist())),
        'peak_I_centralized'      : int(I_c.max()),
        'final_R_centralized'     : int(R_c[-1]),
        'truth_amp_centralized'   : float(R_c[:t_cut].mean() / (I_c[:t_cut].mean() + 1e-9)),
        'polarization_centralized': float(np.mean(np.abs(I_c - R_c) / (I_c + R_c + 1e-9))),
        'spread_red_centralized'  : float((I_s.max() - I_c.max()) / (I_s.max() + 1e-9)),
        'I_curve_centralized'     : list(map(int, I_c.tolist())),
        'R_curve_centralized'     : list(map(int, R_c.tolist())),
        'I_curve_sir'  : list(map(int, I_s.tolist())),
        'I_curve_marl' : list(map(int, I_m.tolist())),
        'R_curve_sir'  : list(map(int, R_s.tolist())),
        'R_curve_marl' : list(map(int, R_m.tolist())),
    }
