"""Figures added in the revision of the TEMPEST paper.

Writes to figures/:
  fig_intensity_T.png         rumour intensity and mean infected curve (Fig. 2 in the revised paper)
  fig_ablation_20seeds.png    component ablations over 20 seeds
  fig_learned_interventions.png  TEMPEST vs. QDOM-adapted and DRLE-adapted

Inputs (produced by the other scripts in this folder):
  multiseed_results/seed_*.json        run_seeds.py (main 20-seed BA run)
  results_v2/ablation/ba_k4_T30/       revision_runner.py --only ablation
  results_v2/baselines/ba_k4_T30/      revision_runner.py --only baselines

Usage:  python make_paper_figures.py
"""
import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results_v2')
OLD = os.path.join(HERE, 'multiseed_results')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
T, N = 30, 5000
seeds = list(range(20))

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10.5, 'axes.grid': True, 'grid.alpha': 0.3,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
CS, CI, CR = 'blue', 'red', 'green'


def load(exp, var):
    out = {}
    for f in sorted(glob.glob(f'{R}/{exp}/{var}/seed_*.json')):
        d = json.load(open(f)); out[d['seed']] = d
    return out


old = {}
for f in sorted(glob.glob(OLD + '/seed_*.json')):
    d = json.load(open(f)); old[d['seed']] = d


def oc(s, k):
    I = np.array(old[s][f'I_curve_{k}'], float); Rr = np.array(old[s][f'R_curve_{k}'], float)
    return {'S': N - I - Rr, 'I': I, 'R': Rr}


def mean_sd(curves):
    A = np.array(curves); return A.mean(0), A.std(0, ddof=1)


def contain_line(ax, label=True):
    ax.axvline(T, color='black', ls='--', lw=1.4,
               label=f'Containment phase starts (t={T})' if label else None)


t = np.arange(60)

# ------------------------------------------------------------------ R1.1
PEAK_T, K, LAM, FL = 20, 0.30, 0.05, 0.25
tt = np.linspace(0, 59, 600)
it = FL + (1 - FL) / (1 + np.exp(-K * (tt - PEAK_T))) * np.exp(-LAM * np.maximum(0, tt - PEAK_T))
tstar = tt[it.argmax()]
tem = [oc(s, 'marl') for s in seeds]; sir = [oc(s, 'sir') for s in seeds]
mI, sI = mean_sd([c['I'] for c in tem]); mS, _ = mean_sd([c['I'] for c in sir])

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
a = ax[0]
a.plot(tt, it, color='tab:blue', lw=2.5, label=r'Rumour intensity $\iota_t$')
a.axvline(PEAK_T, color='gray', ls=':', lw=1.4, label=r'$t_{peak}=20$ (fatigue onset)')
a.axvline(tstar, color='tab:orange', ls='-.', lw=1.6, label=rf'Intensity maximum $t^*\approx{tstar:.1f}$')
contain_line(a)
a.set_title(r'(a) Rumour intensity $\iota_t$')
a.set_xlabel('Time'); a.set_ylabel(r'$\iota_t$'); a.legend(loc='upper left')

a = ax[1]
a.plot(t, mS, color=CI, ls='--', lw=2, label=r'Classical SIR (fixed $\beta$=0.2)')
a.plot(t, mI, color=CI, lw=2.5, label='TEMPEST (learned policy)')
a.fill_between(t, mI - sI, mI + sI, color=CI, alpha=0.12, lw=0, label='TEMPEST ±1 SD')
a.axvline(int(mI.argmax()), color='gray', ls=':', lw=1.4, label=f'TEMPEST infection peak (t={int(mI.argmax())})')
contain_line(a)
a.set_title('(b) Infected over time (20 seeds)')
a.set_xlabel('Time'); a.set_ylabel('Infected nodes'); a.set_ylim(-250, 5900); a.legend(loc='upper right', ncol=2, fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT + '/fig_intensity_T.pdf'); fig.savefig(OUT + '/fig_intensity_T.png'); plt.close(fig)
print('t* =', round(tstar, 2), ' TEMPEST peak t =', int(mI.argmax()))

# ------------------------------------------------------------------ R1.2 ablation (S/I/R panels, as Fig. 6)
A = load('ablation', 'ba_k4_T30')
panels = [('(a) Full TEMPEST', None), ('(b) No Social Contagion', 'abl_no_contagion'),
          ('(c) No MARL (random policy)', 'abl_random_policy'), ('(d) No EGT Utility Shaping', 'abl_no_egt'),
          ('(e) No Ignition Window', 'abl_no_ignition')]
fig, axs = plt.subplots(2, 3, figsize=(16, 9.2))
axs = axs.ravel()
for i, (title, k) in enumerate(panels):
    a = axs[i]
    cur = tem if k is None else [{kk: np.array(A[s]['curves'][k][kk], float) for kk in 'SIR'} for s in seeds]
    for kk, col in (('S', CS), ('I', CI), ('R', CR)):
        m, sd = mean_sd([c[kk] for c in cur])
        a.plot(t, m, color=col, lw=2, label=kk)
        a.fill_between(t, m - sd, m + sd, color=col, alpha=0.12, lw=0)
    contain_line(a, label=(i == 0))
    a.set_title(title); a.set_xlabel('Time'); a.set_ylabel('Nodes'); a.set_ylim(-150, 5150)
# panel (f): infected under every variant
a = axs[5]
sty = [('Full TEMPEST', None, 'red', '-'), ('No social contagion', 'abl_no_contagion', 'tab:orange', '--'),
       ('No MARL (random)', 'abl_random_policy', 'tab:purple', '-.'), ('No EGT shaping', 'abl_no_egt', 'tab:brown', (0, (3, 1, 1, 1))),
       ('No ignition window', 'abl_no_ignition', 'tab:pink', ':')]
for name, k, col, ls in sty:
    cur = tem if k is None else [{kk: np.array(A[s]['curves'][k][kk], float) for kk in 'IR'} for s in seeds]
    m, _ = mean_sd([c['I'] for c in cur]); a.plot(t, m, color=col, ls=ls, lw=2, label=name)
contain_line(a, label=False)
a.set_title('(f) Infected under each ablation'); a.set_xlabel('Time'); a.set_ylabel('Infected nodes')
a.legend(loc='upper right')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035), fontsize=10)
fig.tight_layout()
fig.savefig(OUT + '/fig_ablation_20seeds.pdf'); fig.savefig(OUT + '/fig_ablation_20seeds.png'); plt.close(fig)

# ------------------------------------------------------------------ R1.3 learned-intervention baselines (as Fig. 2)
B = load('baselines', 'ba_k4_T30')
def bc(k): return [{kk: np.array(B[s]['curves'][k][kk], float) for kk in 'IR'} for s in seeds]
meth = [('TEMPEST (learned user behaviour)', tem, 'red', '-', 2.5),
        ('QDOM-adapted (seed selection)', bc('qdom'), 'tab:blue', '--', 2),
        ('DRLE-adapted (link removal)', bc('drle'), 'tab:green', '-.', 2),
        ('No intervention', bc('no_intervention_type_prior'), 'gray', ':', 2)]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
means = {}
for name, cur, col, ls, lw in meth:
    for j, kk in enumerate('IR'):
        m, sd = mean_sd([c[kk] for c in cur]); means[(name, kk)] = m
        ax[j].plot(t, m, color=col, ls=ls, lw=lw, label=name)
        if cur is tem:
            ax[j].fill_between(t, m - sd, m + sd, color=col, alpha=0.10, lw=0)
best = means[(meth[1][0], 'I')]; mt = means[(meth[0][0], 'I')]
ax[0].fill_between(t[T:], mt[T:], best[T:], where=best[T:] > mt[T:], color='green', alpha=0.2,
                   label='TEMPEST containment advantage')
for j, a in enumerate(ax):
    contain_line(a); a.set_xlabel('Time')
ax[0].set_title('(a) Infected over time'); ax[0].set_ylabel('Infected nodes'); ax[0].legend(loc='upper right')
ax[1].set_title('(b) Recovered over time'); ax[1].set_ylabel('Recovered nodes'); ax[1].legend(loc='lower right')
fig.tight_layout()
fig.savefig(OUT + '/fig_learned_interventions.pdf'); fig.savefig(OUT + '/fig_learned_interventions.png'); plt.close(fig)
print('done')
