"""Generate figures/paper_table.tex from aggregate.json + sustained reduction stats."""
import json
import numpy as np
from pathlib import Path

# ── Load aggregate ────────────────────────────────────────────────────────────
with open('multiseed_results/aggregate.json') as f:
    agg = json.load(f)

def g(key):
    return agg[key]

# ── Sustained reduction (recomputed inline) ───────────────────────────────────
seed_files = sorted(Path('multiseed_results').glob('seed_*.json'))
reductions = []
for sf in seed_files:
    with open(sf) as f:
        r = json.load(f)
    s_sir  = np.array(r['I_curve_sir'])[30:].mean()
    s_marl = np.array(r['I_curve_marl'])[30:].mean()
    reductions.append((s_sir - s_marl) / s_sir * 100)
reductions = np.array(reductions)
sust_mean = reductions.mean()
sust_sem  = reductions.std(ddof=1) / np.sqrt(len(reductions))

def fmt(mean, sem, decimals=1):
    """Format as mean ± SEM with sem to 2 sig figs."""
    # Round sem to 2 significant figures
    if sem == 0:
        return f'{mean:.{decimals}f} $\\pm$ 0'
    from math import floor, log10
    mag = floor(log10(abs(sem)))
    sem_r = round(sem, -mag + 1)
    # Match decimals of mean to sem precision
    dec = max(decimals, max(0, -mag + 1))
    return f'{mean:.{dec}f} $\\pm$ {sem_r:.{dec}f}'

def pct_improve(sir_mean, marl_mean, direction='lower_is_better'):
    """Compute % improvement of MARL vs SIR."""
    if direction == 'lower_is_better':
        return (sir_mean - marl_mean) / sir_mean * 100
    else:
        return (marl_mean - sir_mean) / sir_mean * 100

peak_imp   = pct_improve(g('peak_I_sir')['mean'],   g('peak_I_marl')['mean'],   'lower_is_better')
detect_imp = g('mean_detect_t_sir')['mean'] - g('mean_detect_t_marl')['mean']
truth_ratio= g('truth_amp_marl')['mean'] / g('truth_amp_sir')['mean']

# ── Build LaTeX ───────────────────────────────────────────────────────────────
tex = r"""\begin{table}[t]
\centering
\caption{Aggregate results over 20 independent seeds (mean $\pm$ SEM).
         All experiments: Barab\'asi--Albert graph, $n=5000$ nodes, 80 training episodes.}
\label{tab:results}
\begin{tabular}{lccr}
\toprule
\textbf{Metric} & \textbf{Classical SIR} & \textbf{MARL-DQN (ours)} & \textbf{Improvement} \\
\midrule
"""

rows = [
    (
        'Peak infection (nodes)',
        fmt(g('peak_I_sir')['mean'],   g('peak_I_sir')['sem'],   0),
        fmt(g('peak_I_marl')['mean'],  g('peak_I_marl')['sem'],  0),
        f'$-{peak_imp:.1f}\\%$'
    ),
    (
        'Final recovered (nodes)',
        fmt(g('final_R_sir')['mean'],  g('final_R_sir')['sem'],  0),
        fmt(g('final_R_marl')['mean'], g('final_R_marl')['sem'], 0),
        'higher for MARL'
    ),
    (
        'Truth amplification',
        fmt(g('truth_amp_sir')['mean'],  g('truth_amp_sir')['sem'],  3),
        fmt(g('truth_amp_marl')['mean'], g('truth_amp_marl')['sem'], 3),
        f'${truth_ratio:.2f}\\times$'
    ),
    (
        'Belief polarization',
        fmt(g('polarization_sir')['mean'],  g('polarization_sir')['sem'],  3),
        fmt(g('polarization_marl')['mean'], g('polarization_marl')['sem'], 3),
        r'---'
    ),
    (
        'Mean detection time (steps)',
        fmt(g('mean_detect_t_sir')['mean'],  g('mean_detect_t_sir')['sem'],  1),
        fmt(g('mean_detect_t_marl')['mean'], g('mean_detect_t_marl')['sem'], 1),
        f'${detect_imp:.1f}$ steps faster'
    ),
    (
        'Influential node efficiency',
        r'---',
        fmt(g('influential_efficiency')['mean'], g('influential_efficiency')['sem'], 3),
        r'---'
    ),
    (
        r'Spread reduction',
        r'---',
        fmt(g('spread_reduction')['mean'] * 100, g('spread_reduction')['sem'] * 100, 1) + r'\%',
        r'---'
    ),
    (
        r'Sustained $I$ reduction ($t \geq 30$)',
        r'---',
        fmt(sust_mean, sust_sem, 1) + r'\%',
        r'---'
    ),
]

for metric, sir_val, marl_val, imp in rows:
    tex += f'  {metric} & {sir_val} & {marl_val} & {imp} \\\\\n'

tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

out = Path('figures/paper_table.tex')
out.write_text(tex)
print(f'Saved → {out}')
print()
print(tex)
