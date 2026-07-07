"""
Plot multi-seed aggregate results — mean ± std bands for the paper.

Run after run_seeds.py has finished:
    python plot_aggregate.py --indir multiseed_results --outdir figures

Produces:
    figures/01_I_curve_marl_vs_sir.png   — I-curve with ±1σ band
    figures/02_R_curve_marl_vs_sir.png   — R-curve with ±1σ band
    figures/03_metrics_barchart.png      — bar comparison with error bars
    figures/04_rumour_detection.png      — detection rate + per-rumour metrics
    figures/paper_metrics_table.csv      — paper-ready CSV
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_aggregate(indir: Path):
    with open(indir / 'aggregate.json') as f:
        return json.load(f)


def plot_curve_with_band(agg, curve_marl_key, curve_sir_key, ylabel, title, out_path,
                         curve_greedy_key=None, curve_centralized_key=None):
    """Plot mean curve with ±1σ shaded band for SIR / Greedy / Centralized / MARL."""
    fig, ax = plt.subplots(figsize=(10, 5))
    curves = [
        (curve_sir_key,  'Classical SIR',      '#1976D2'),
        (curve_marl_key, 'TEMPEST (MARL-DQN)', '#D32F2F'),
    ]
    if curve_greedy_key and curve_greedy_key in agg:
        curves.insert(1, (curve_greedy_key, 'Greedy Heuristic', '#E65100'))
    if curve_centralized_key and curve_centralized_key in agg:
        curves.insert(-1, (curve_centralized_key, 'Centralized DQN', '#7B1FA2'))
    for key, label, color in curves:
        if key not in agg: continue
        mean = np.array(agg[key]['mean'])
        std  = np.array(agg[key]['std'])
        t    = np.arange(len(mean))
        ax.plot(t, mean, color=color, linewidth=2.2, label=f'{label} (mean)')
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18,
                        label=f'{label} (±1σ)')
    ax.set_title(f'{title}\n({agg["n_seeds"]} seeds)')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved {out_path}')


def plot_metrics_bars(agg, out_path):
    """Bar chart: SIR / Greedy / Centralized DQN / TEMPEST with SEM error bars."""
    # (title, sir_key, greedy_key, centralized_key, marl_key)
    metrics = [
        ('Spread Reduction',       'spread_reduction',      'spread_red_greedy',
         'spread_red_centralized',  None),
        ('Peak I',                 'peak_I_sir',            'peak_I_greedy',
         'peak_I_centralized',      'peak_I_marl'),
        ('Truth Amplification',    'truth_amp_sir',         'truth_amp_greedy',
         'truth_amp_centralized',   'truth_amp_marl'),
        ('Polarization',           'polarization_sir',      'polarization_greedy',
         'polarization_centralized','polarization_marl'),
        ('Detection Rate',         'detection_rate_sir',    None,
         None,                      'detection_rate_marl'),
        ('Mean Detection t',       'mean_detect_t_sir',     None,
         None,                      'mean_detect_t_marl'),
        ('Influential Efficiency', 'influential_efficiency',None,
         None,                      None),
    ]
    colors = ['#1976D2', '#E65100', '#7B1FA2', '#D32F2F']
    lbls   = ['SIR', 'Greedy', 'Centralized', 'TEMPEST']

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    for ax, (title, sir_k, grdy_k, cen_k, marl_k) in zip(axes.flatten(), metrics):
        bar_labels, vals, errs, cols = [], [], [], []
        for k, lbl, col in zip(
                [sir_k, grdy_k, cen_k, marl_k], lbls, colors):
            if k and k in agg:
                bar_labels.append(lbl)
                vals.append(agg[k]['mean'])
                errs.append(agg[k]['sem'])
                cols.append(col)
        if not vals:
            ax.set_visible(False); continue
        x = np.arange(len(bar_labels))
        ax.bar(x, vals, yerr=errs, color=cols, edgecolor='black',
               alpha=0.85, capsize=5, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        for xi, (v, e) in enumerate(zip(vals, errs)):
            ax.text(xi, v + e + 0.01 * abs(v + 1),
                    f'{v:.2f}\n±{e:.2f}', ha='center', fontsize=7)
        ax.set_title(title); ax.grid(axis='y', alpha=0.3)

    for ax in axes.flatten()[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(
        f'Metrics Comparison: SIR / Greedy / Centralized DQN / TEMPEST'
        f' — {agg["n_seeds"]} seeds (SEM bars)',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved {out_path}')


def plot_baselines_comparison(agg, out_path):
    """Four-baseline paper figure: Peak I, Final R, Spread Reduction."""
    # (title, sir_key, greedy_key, centralized_key, marl_key)
    metrics = [
        ('Peak Infection (nodes)',  'peak_I_sir',       'peak_I_greedy',
         'peak_I_centralized',      'peak_I_marl'),
        ('Final Recovered (nodes)', 'final_R_sir',      'final_R_greedy',
         'final_R_centralized',     'final_R_marl'),
        ('Spread Reduction (%)',    'spread_reduction', 'spread_red_greedy',
         'spread_red_centralized',  None),
    ]
    all_labels = ['Classical SIR', 'Greedy', 'Centralized DQN', 'TEMPEST']
    colors     = ['#1976D2', '#E65100', '#7B1FA2', '#D32F2F']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, sir_k, grdy_k, cen_k, marl_k) in zip(axes, metrics):
        bar_labels, vals, errs, cols = [], [], [], []
        for k, lbl, col in zip(
                [sir_k, grdy_k, cen_k, marl_k], all_labels, colors):
            if k and k in agg:
                s = agg[k]
                if isinstance(s.get('mean'), (int, float)):
                    bar_labels.append(lbl); vals.append(s['mean'])
                    errs.append(s['sem']); cols.append(col)
        if not vals:
            ax.set_visible(False); continue
        x = np.arange(len(bar_labels))
        ax.bar(x, vals, yerr=errs, color=cols, edgecolor='black',
               alpha=0.88, capsize=7, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9, rotation=15, ha='right')
        for xi, (v, e) in enumerate(zip(vals, errs)):
            ax.text(xi, v + e + 0.01 * abs(v + 1),
                    f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'Four-Baseline Comparison: SIR / Greedy / Centralized DQN / TEMPEST\n'
        f'({agg["n_seeds"]} seeds, error bars = SEM)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved {out_path}')


def export_paper_csv(agg, out_path):
    """Write a CSV in the format the paper will quote: metric, mean, std, sem, n."""
    rows = []
    for k, v in agg.items():
        if isinstance(v, dict) and 'mean' in v and 'std' in v and not isinstance(v['mean'], list):
            rows.append([k, v['mean'], v['std'], v['sem'], v['n']])
    rows.sort()
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mean', 'std', 'sem', 'n_seeds'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.6f}', f'{r[2]:.6f}', f'{r[3]:.6f}', r[4]])
    print(f'  saved {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir',  type=str, default='multiseed_results')
    ap.add_argument('--outdir', type=str, default='figures')
    args = ap.parse_args()

    indir  = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    agg = load_aggregate(indir)
    print(f'Loaded aggregate: n_seeds={agg["n_seeds"]}')

    plot_curve_with_band(agg, 'I_curve_marl', 'I_curve_sir',
                         'Infected nodes',
                         'Infected Curve — SIR / Greedy / Centralized / TEMPEST',
                         outdir / '01_I_curve_marl_vs_sir.png',
                         curve_greedy_key='I_curve_greedy',
                         curve_centralized_key='I_curve_centralized')
    plot_curve_with_band(agg, 'R_curve_marl', 'R_curve_sir',
                         'Recovered nodes',
                         'Recovered Curve — SIR / Greedy / Centralized / TEMPEST',
                         outdir / '02_R_curve_marl_vs_sir.png',
                         curve_greedy_key='R_curve_greedy',
                         curve_centralized_key='R_curve_centralized')
    plot_metrics_bars(agg, outdir / '03_metrics_barchart.png')
    plot_baselines_comparison(agg, outdir / '06_baselines_comparison.png')
    export_paper_csv(agg, outdir / 'paper_metrics_table.csv')

    print('\nAll plots/tables ready for the paper.')


if __name__ == '__main__':
    main()
