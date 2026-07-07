"""
Standalone classical SIR baseline on the Stanford SNAP Twitter ego-network.
Outputs:
    outputs/twitter_sir_curves.json
    outputs/03_twitter_sir.png

Does NOT retrain MARL. Does NOT modify any existing files.
"""

import os, sys, json, time
import numpy as np
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pull only pure-Python constants and the rumour_intensity function from marl_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marl_core import rumour_intensity, INIT_INFECTED, CONTAIN_T

# ── Constants (notebook cell 4 values) ──────────────────────────────────────
MAX_TIME   = 60
BETA       = 0.2
GAMMA      = 0.05   # user-specified for this baseline
GRAPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "twitter_combined.txt")

t_wall = time.time()

# ── Load Twitter graph (identical to Cell 24) ────────────────────────────────
print("Loading Twitter graph …", flush=True)
G_tw = nx.convert_node_labels_to_integers(
    nx.read_edgelist(GRAPH_PATH, create_using=nx.Graph()))
n = G_tw.number_of_nodes()
m = G_tw.number_of_edges()
print(f"  {n:,} nodes  |  {m:,} edges", flush=True)

# Pre-compute neighbor arrays (numpy int32 for fast boolean indexing)
print("Pre-computing neighbor lists …", flush=True)
nbrs = [np.array(list(G_tw.neighbors(i)), dtype=np.int32) for i in range(n)]
del G_tw   # free networkx object — only nbrs needed from here on

# ── Seed RNG and sample initial infected ─────────────────────────────────────
np.random.seed(0)
n_init = min(INIT_INFECTED, n // 10)
init_nodes_tw = np.random.choice(n, n_init, replace=False)
print(f"Initial infected: {n_init} nodes  (seed=0)", flush=True)

# ── Classical SIR with rumour_intensity modulation ───────────────────────────
# State encoding: 0=Susceptible, 1=Infected, 2=Recovered
states = np.zeros(n, dtype=np.int8)
states[init_nodes_tw] = 1

hist = {'S': [], 'I': [], 'R': []}

print(f"Running SIR for {MAX_TIME} timesteps …", flush=True)
for t in range(MAX_TIME):
    intensity  = rumour_intensity(t)
    eff_beta   = BETA * intensity
    new_states = states.copy()

    inf_nodes = np.where(states == 1)[0]

    for u in inf_nodes:
        nb = nbrs[u]
        if nb.size:
            # Vectorised: draw one random per neighbour, infect susceptible ones
            susc = nb[states[nb] == 0]
            if susc.size:
                hits = np.random.rand(susc.size) < eff_beta
                new_states[susc[hits]] = 1
        # Recovery
        if np.random.rand() < GAMMA:
            new_states[u] = 2

    states = new_states
    hist['S'].append(int(np.sum(states == 0)))
    hist['I'].append(int(np.sum(states == 1)))
    hist['R'].append(int(np.sum(states == 2)))

    if (t + 1) % 10 == 0:
        print(f"  t={t+1:3d}  I={hist['I'][-1]:>7,}  R={hist['R'][-1]:>7,}", flush=True)

t_total = time.time() - t_wall

# ── Metrics ──────────────────────────────────────────────────────────────────
I_arr = np.array(hist['I'])
R_arr = np.array(hist['R'])

peak_I       = int(I_arr.max())
peak_I_t     = int(I_arr.argmax())
final_R      = int(R_arr[-1])
sustained_I  = float(I_arr[CONTAIN_T:].mean())

print()
print("=== Twitter SIR Baseline Metrics ===")
print(f"  Peak infection (I):                    {peak_I:>8,}  nodes  (at t={peak_I_t})")
print(f"  Final recovered (R):                   {final_R:>8,}  nodes")
print(f"  Mean infection t>={CONTAIN_T} (sustained):   {sustained_I:>10.2f}  nodes")
print(f"  Total runtime:                                  {t_total:.1f}s")

# ── Save JSON ────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
curves_path = "outputs/twitter_sir_curves.json"
with open(curves_path, 'w') as f:
    json.dump({
        'n_nodes':   n,
        'n_edges':   m,
        'n_init':    n_init,
        'beta':      BETA,
        'gamma':     GAMMA,
        'max_time':  MAX_TIME,
        'contain_t': CONTAIN_T,
        'S': hist['S'],
        'I': hist['I'],
        'R': hist['R'],
        'metrics': {
            'peak_I':          peak_I,
            'peak_I_t':        peak_I_t,
            'final_R':         final_R,
            'sustained_I_mean': round(sustained_I, 4),
            'runtime_s':       round(t_total, 2),
        }
    }, f, indent=2)
print(f"\nSaved: {curves_path}")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ts = list(range(MAX_TIME))
ax.plot(ts, hist['S'], 'b-',  linewidth=2,   label='Susceptible')
ax.plot(ts, hist['I'], 'r-',  linewidth=2,   label='Infected (spreading)')
ax.plot(ts, hist['R'], 'g-',  linewidth=2,   label='Recovered (refuted)')
ax.axvline(CONTAIN_T, color='black', linestyle='--', linewidth=1.5,
           label=f'Containment phase (t={CONTAIN_T})')
ax.set_title(
    "Classical SIR Baseline — Twitter Ego-Network (Stanford SNAP)\n"
    rf"$\beta$={BETA},  $\gamma$={GAMMA},  seed=0,  $n_{{init}}$={n_init}"
)
ax.set_xlabel("Time")
ax.set_ylabel("Nodes")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

plot_path = "outputs/03_twitter_sir.png"
fig.savefig(plot_path, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {plot_path}")
