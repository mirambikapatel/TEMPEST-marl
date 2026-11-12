"""
Zero-shot transfer of a TEMPEST policy to the Stanford SNAP Twitter ego-network.

Fix: community detection is run on G_tw itself (not the BA training graph),
so all 81,306 nodes receive proper Q-network assignments.

Outputs go to outputs/twitter_transfer_fixed/
"""

import os, sys, json, time, random, threading
import numpy as np
import torch
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marl_core import (
    make_graph, GraphCache, train_marl, eval_marl_avg,
    env_step, N_COMMUNITIES, CONTAIN_T, INIT_INFECTED, HIDDEN,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "twitter_transfer_fixed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Logging helper ────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Seed everything
# ─────────────────────────────────────────────────────────────────────────────
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
log("[seed] random / numpy / torch seeded with 0")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train on BA graph
# ─────────────────────────────────────────────────────────────────────────────
log("\n[train] Building BA graph (n=5000, seed=0) …")
t0_train = time.time()
G_ba = make_graph('ba', 5000, 0)
gc_ba = GraphCache(G_ba)
log(f"[train] BA graph: {gc_ba.n} nodes, {G_ba.number_of_edges()} edges")

log("[train] Training MARL (80 eps × 60 steps) …")
marl = train_marl(gc_ba, n_eps=80, max_time=60)
t_train = time.time() - t0_train
log(f"[train] Done in {t_train:.1f}s")

policy_path = os.path.join(OUT_DIR, "policy_ba_seed0.pt")
torch.save([net.state_dict() for net in marl.online], policy_path)
log(f"[train] Saved policy to {policy_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Load Twitter graph
# ─────────────────────────────────────────────────────────────────────────────
tw_candidates = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitter_combined.txt"),
    os.path.expanduser("~/Downloads/twitter_combined.txt"),
]
tw_path = next((p for p in tw_candidates if os.path.exists(p)), None)
if tw_path is None:
    raise FileNotFoundError("twitter_combined.txt not found in project root or ~/Downloads/")

log(f"\n[twitter] Loading graph from {tw_path} …")
t0_tw = time.time()
G_tw = nx.convert_node_labels_to_integers(
    nx.read_edgelist(tw_path, create_using=nx.Graph()))
log(f"[twitter] {G_tw.number_of_nodes():,} nodes  |  {G_tw.number_of_edges():,} edges")

log("[twitter] Building GraphCache (PageRank on 81k nodes – takes ~1 min) …")
gc_tw = GraphCache(G_tw)
log(f"[twitter] GraphCache done in {time.time()-t0_tw:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Community detection on G_tw
# ─────────────────────────────────────────────────────────────────────────────
log("\n[community] Attempting greedy_modularity_communities (60-min limit) …")
t0_comm = time.time()
_greedy_result = [None]
_greedy_exc    = [None]

def _run_greedy_thread():
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        _greedy_result[0] = list(greedy_modularity_communities(G_tw))
    except Exception as e:
        _greedy_exc[0] = e

_gt = threading.Thread(target=_run_greedy_thread, daemon=True)
_gt.start()
_gt.join(timeout=3600)   # 60-minute wall-clock limit

if _gt.is_alive():
    comm_method = "louvain"
    log("[community] greedy timed out — falling back to louvain_communities …")
    raw_comms = nx.community.louvain_communities(G_tw, seed=0)
elif _greedy_result[0] is not None:
    raw_comms = _greedy_result[0]
    comm_method = "greedy_modularity"
    log("[community] greedy_modularity_communities succeeded")
else:
    comm_method = "louvain"
    log(f"[community] greedy raised {_greedy_exc[0]!r} — falling back to louvain_communities …")
    raw_comms = nx.community.louvain_communities(G_tw, seed=0)

t_comm = time.time() - t0_comm
log(f"[community] {comm_method} found {len(raw_comms)} communities in {t_comm:.1f}s")

# Sort largest-first (same rule as MARLSystem.__init__)
raw_comms = sorted(raw_comms, key=len, reverse=True)

node_community = np.zeros(gc_tw.n, dtype=np.int64)
for c_idx, community in enumerate(raw_comms):
    for node in community:
        node_community[node] = min(c_idx, N_COMMUNITIES - 1)

nodes_per_network = [int(np.sum(node_community == c)) for c in range(N_COMMUNITIES)]
log(f"[community] nodes per network: {nodes_per_network}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Point the trained system at Twitter
# ─────────────────────────────────────────────────────────────────────────────
marl.gc             = gc_tw
marl.node_community = node_community
marl.community_masks = [np.where(node_community == c)[0] for c in range(N_COMMUNITIES)]

# Sanity-check: masks must cover every node exactly once
total_in_masks = sum(len(m) for m in marl.community_masks)
all_nodes = np.concatenate(marl.community_masks)
assert total_in_masks == gc_tw.n, \
    f"Mask sizes sum to {total_in_masks}, expected {gc_tw.n}"
assert len(np.unique(all_nodes)) == gc_tw.n, \
    "Masks contain duplicate nodes"
log(f"[assert] All {gc_tw.n} nodes covered exactly once — OK")

for net in marl.online:
    net.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Evaluate with eval_marl_avg + one instrumented rollout
# ─────────────────────────────────────────────────────────────────────────────
log("\n[eval] Running eval_marl_avg (5 rollouts, max_time=60) …")
t0_inf = time.time()
marl_avg = eval_marl_avg(marl, max_time=60, n_evals=5)
t_inf = time.time() - t0_inf
log(f"[eval] Done in {t_inf:.1f}s")

# Extra instrumented rollout (action counts per step)
log("[eval] Running instrumented rollout (action counts) …")
action_counts = {'Ignore': [], 'Share': [], 'Refute': []}

states = np.zeros(gc_tw.n, dtype=np.int64)
init_nodes = np.random.choice(gc_tw.n, INIT_INFECTED, replace=False)
states[init_nodes] = 1
h = torch.zeros(gc_tw.n, HIDDEN)

with torch.no_grad():
    for t in range(60):
        x = gc_tw.features(states, t)
        actions, h = marl.select_actions(x, h, eps=0.0)
        states = env_step(states, actions, gc_tw, t, init_nodes=init_nodes)
        action_counts['Ignore'].append(int(np.sum(actions == 0)))
        action_counts['Share'].append(int(np.sum(actions == 1)))
        action_counts['Refute'].append(int(np.sum(actions == 2)))

log("[eval] Instrumented rollout done")

# Individual rollouts (store all 5)
log("[eval] Collecting individual rollout curves …")
from marl_core import eval_marl
all_rollouts = []
for _ in range(5):
    r = eval_marl(marl, max_time=60)
    all_rollouts.append(r)

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Load existing SIR baseline
# ─────────────────────────────────────────────────────────────────────────────
sir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "twitter_sir_curves.json")
with open(sir_path) as f:
    sir_data = json.load(f)
log(f"\n[sir] Loaded baseline: beta={sir_data['beta']} gamma={sir_data['gamma']}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Compute metrics and save summary.json + curves.json
# ─────────────────────────────────────────────────────────────────────────────
I_marl = np.array(marl_avg['I'])
R_marl = np.array(marl_avg['R'])
S_marl = np.array(marl_avg['S'])

I_sir = np.array(sir_data['I'])
R_sir = np.array(sir_data['R'])
S_sir = np.array(sir_data['S'])

peak_I_marl     = float(I_marl.max())
peak_I_marl_t   = int(I_marl.argmax())
peak_I_sir      = float(I_sir.max())
peak_I_sir_t    = int(I_sir.argmax())

mean_I_marl_post = float(I_marl[CONTAIN_T:].mean())
mean_I_sir_post  = float(I_sir[CONTAIN_T:].mean())

sustained_reduction = float(1.0 - mean_I_marl_post / (mean_I_sir_post + 1e-9))
peak_reduction      = float(1.0 - peak_I_marl / (peak_I_sir + 1e-9))

final_R_marl = float(R_marl[-1])
final_R_sir  = float(R_sir[-1])

# Signed polarization: mean of (R - I) / (R + I + 1e-9) for two phases
def signed_pol(I_arr, R_arr, start, end):
    I_s = I_arr[start:end]; R_s = R_arr[start:end]
    return float(np.mean((R_s - I_s) / (R_s + I_s + 1e-9)))

pol_marl_early = signed_pol(I_marl, R_marl, 0, CONTAIN_T)
pol_marl_late  = signed_pol(I_marl, R_marl, CONTAIN_T, 60)
pol_sir_early  = signed_pol(I_sir,  R_sir,  0, CONTAIN_T)
pol_sir_late   = signed_pol(I_sir,  R_sir,  CONTAIN_T, 60)

summary = {
    "community_method":        comm_method,
    "community_detection_time_s": round(t_comm, 2),
    "nodes_per_network":       nodes_per_network,
    "total_communities_found": len(raw_comms),
    "sir_params": {
        "beta":    sir_data['beta'],
        "gamma":   sir_data['gamma'],
        "n_nodes": sir_data['n_nodes'],
        "n_init":  sir_data['n_init'],
    },
    "peak_I_marl":          round(peak_I_marl, 2),
    "peak_I_marl_t":        peak_I_marl_t,
    "peak_I_sir":           round(peak_I_sir, 2),
    "peak_I_sir_t":         peak_I_sir_t,
    "mean_I_post30_marl":   round(mean_I_marl_post, 4),
    "mean_I_post30_sir":    round(mean_I_sir_post, 4),
    "sustained_infection_reduction": round(sustained_reduction, 6),
    "peak_reduction":        round(peak_reduction, 6),
    "final_R_marl":         round(final_R_marl, 2),
    "final_R_sir":          round(final_R_sir, 2),
    "signed_polarization": {
        "marl_t_lt_30":  round(pol_marl_early, 6),
        "marl_t_ge_30":  round(pol_marl_late, 6),
        "sir_t_lt_30":   round(pol_sir_early, 6),
        "sir_t_ge_30":   round(pol_sir_late, 6),
    },
    "wall_clock_times_s": {
        "training":           round(t_train, 2),
        "community_detection": round(t_comm, 2),
        "twitter_inference":  round(t_inf, 2),
    },
    "action_counts_per_step": action_counts,
}

summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"[save] summary.json written")

curves = {
    "mean": {
        "S": marl_avg['S'],
        "I": marl_avg['I'],
        "R": marl_avg['R'],
        "std_I": marl_avg['std_I'],
        "std_R": marl_avg['std_R'],
    },
    "rollouts": all_rollouts,
}
curves_path = os.path.join(OUT_DIR, "curves.json")
with open(curves_path, 'w') as f:
    json.dump(curves, f, indent=2)
log(f"[save] curves.json written")

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ts = list(range(60))

ax.plot(ts, marl_avg['S'], 'b-',  linewidth=2, label='Susceptible')
ax.plot(ts, marl_avg['I'], 'r-',  linewidth=2, label='Infected (spreading)')
ax.plot(ts, marl_avg['R'], 'g-',  linewidth=2, label='Recovered (refuted)')
ax.axvline(CONTAIN_T, color='black', linestyle='--', linewidth=1.5,
           label='Containment phase starts (t=30)')
ax.set_title('TEMPEST applied to the Twitter ego-network (zero-shot)')
ax.set_xlabel('Time')
ax.set_ylabel('Nodes')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "03_twitter_marl_fixed.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
log(f"[plot] Saved {plot_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 12: Print summary to stdout
# ─────────────────────────────────────────────────────────────────────────────
log("\n=== FINAL SUMMARY ===")
log(f"  Community method:               {comm_method} ({t_comm:.1f}s)")
log(f"  Nodes per network:              {nodes_per_network}")
log(f"  Peak I (TEMPEST):               {peak_I_marl:.0f}  at t={peak_I_marl_t}")
log(f"  Peak I (SIR):                   {peak_I_sir:.0f}  at t={peak_I_sir_t}")
log(f"  Mean I t>=30 TEMPEST:           {mean_I_marl_post:.2f}")
log(f"  Mean I t>=30 SIR:               {mean_I_sir_post:.2f}")
log(f"  Sustained infection reduction:  {sustained_reduction:.4f}")
log(f"  Peak reduction:                 {peak_reduction:.4f}")
log(f"  Final R (TEMPEST):              {final_R_marl:.0f}")
log(f"  Final R (SIR):                  {final_R_sir:.0f}")
log(f"  Training time:                  {t_train:.1f}s")
log(f"  Inference time:                 {t_inf:.1f}s")
log("DONE")
