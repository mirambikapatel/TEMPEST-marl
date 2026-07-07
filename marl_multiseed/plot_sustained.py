"""
Sustained infection analysis across 20 seeds.

For each seed, computes:
    sustained_I_sir  = mean(I_curve_sir[30:])
    sustained_I_marl = mean(I_curve_marl[30:])
    reduction_pct    = (sustained_I_sir - sustained_I_marl) / sustained_I_sir * 100

Reports mean ± std ± SEM of reduction_pct and saves figures/04_sustained_infection.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

INDIR  = Path('multiseed_results')
OUTDIR = Path('figures')
OUTDIR.mkdir(exist_ok=True)

# ── Load per-seed curves ──────────────────────────────────────────────────────
seed_files = sorted(INDIR.glob('seed_*.json'))
assert seed_files, f"No seed_*.json found in {INDIR}"

sir_curves, marl_curves, reductions = [], [], []

for sf in seed_files:
    with open(sf) as f:
        r = json.load(f)
    sir  = np.array(r['I_curve_sir'])
    marl = np.array(r['I_curve_marl'])
    sir_curves.append(sir)
    marl_curves.append(marl)

    s_sir  = sir[30:].mean()
    s_marl = marl[30:].mean()
    pct    = (s_sir - s_marl) / s_sir * 100
    reductions.append(pct)

reductions = np.array(reductions)
mean_r = reductions.mean()
std_r  = reductions.std(ddof=1)
sem_r  = std_r / np.sqrt(len(reductions))

print(f"\nSustained infection reduction (t≥30):")
print(f"  mean : {mean_r:.1f}%")
print(f"  std  : {std_r:.1f}%")
print(f"  SEM  : {sem_r:.1f}%")
print(f"  min  : {reductions.min():.1f}%  (seed {reductions.argmin()})")
print(f"  max  : {reductions.max():.1f}%  (seed {reductions.argmax()})")

# ── Pad curves to same length ─────────────────────────────────────────────────
max_t = max(len(c) for c in sir_curves + marl_curves)
def pad(curves, length):
    return np.array([np.pad(c, (0, length - len(c)), mode='edge') for c in curves])

sir_arr  = pad(sir_curves,  max_t)
marl_arr = pad(marl_curves, max_t)
T = np.arange(max_t)

sir_mean  = sir_arr.mean(axis=0)
sir_std   = sir_arr.std(axis=0, ddof=1)
marl_mean = marl_arr.mean(axis=0)
marl_std  = marl_arr.std(axis=0, ddof=1)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# SIR band
ax.fill_between(T, sir_mean - sir_std, sir_mean + sir_std,
                alpha=0.15, color='#d62728', label='_nolegend_')
ax.plot(T, sir_mean, color='#d62728', lw=2, label='Classical SIR')

# MARL band
ax.fill_between(T, marl_mean - marl_std, marl_mean + marl_std,
                alpha=0.15, color='#1f77b4', label='_nolegend_')
ax.plot(T, marl_mean, color='#1f77b4', lw=2, label='MARL-DQN (ours)')

# Vertical dashed line at t=30
ax.axvline(30, color='black', lw=1.2, linestyle='--', alpha=0.7, label='Containment onset (t=30)')

# Green shaded wedge between the curves from t=30 onward
T_post = T[30:]
ax.fill_between(T_post, marl_mean[30:], sir_mean[30:],
                where=(sir_mean[30:] > marl_mean[30:]),
                alpha=0.25, color='#2ca02c', label='Sustained reduction')

# Annotation
ax.text(0.97, 0.96,
        f'Sustained infection reduction\n'
        f'{mean_r:.1f}% ± {std_r:.1f}%  (SEM {sem_r:.1f}%)',
        transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#aaaaaa', alpha=0.9))

ax.set_xlabel('Simulation step', fontsize=12)
ax.set_ylabel('Infected nodes', fontsize=12)
ax.set_title('Mean Infected Curve: MARL-DQN vs Classical SIR\n'
             f'(20 seeds, ±1σ bands, 5000-node BA graph, 80 training episodes)', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, max_t - 1)
ax.set_ylim(bottom=0)
plt.tight_layout()

out = OUTDIR / '04_sustained_infection.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.close()
