# TEMPEST: MARL-DQN for Rumour Containment

Multi-agent reinforcement learning (MARL-DQN) system for containing rumour
spreading on social networks. Agents use Graph Attention Networks (GAT) with
GRU memory, partitioned into communities, and are compared against three
baselines: Classical SIR, Greedy Heuristic, and Centralized DQN.

Results are aggregated over **20 independent seeds** for statistical validity.

---

## Repository structure

| File | Purpose |
|---|---|
| `marl_core.py` | Core model, environment, train/eval pipeline. Exports `run_full_experiment(seed, cfg)`. |
| `run_seeds.py` | Parallel multi-seed runner (one process per seed). Produces per-seed JSONs + aggregate. |
| `run_twitter_sir.py` | Classical SIR baseline on the Stanford SNAP Twitter ego-network. |
| `plot_aggregate.py` | Loads `aggregate.json`; produces paper figures with mean ± std bands. |
| `plot_sustained.py` | Sustained infection analysis (post-containment phase) across seeds. |
| `significance.py` | Paired t-tests and Wilcoxon signed-rank tests across all four baselines. |
| `make_paper_table.py` | Generates `figures/paper_table.tex` (LaTeX table for the paper). |
| `add_greedy_to_existing.py` | Backfills Greedy Heuristic metrics into existing seed JSONs without retraining. |
| `add_centralized_to_existing.py` | Backfills Centralized DQN metrics into existing seed JSONs without retraining. |
| `launch_full_run.sh` | End-to-end shell script: smoke test → full run → figures. |
| `twitter_combined.txt` | Stanford SNAP Twitter ego-network edge list (42 MB, ~81K nodes). |
| `finaloutbreak.ipynb` | Original development notebook (v7 implementation). |
| `multiseed_results/` | Per-seed JSONs, aggregate JSON/TXT, and per-seed logs. |
| `figures/` | Paper figures (PNG), metrics CSV, and LaTeX table. |
| `outputs/` | Additional figures from the notebook and Twitter SIR baseline. |

---

## Setup

```bash
pip install -r requirements.txt
```

`torch-geometric` can be finicky about wheel availability. If the above fails,
follow the official install guide at https://pytorch-geometric.readthedocs.io/
for your exact torch + CUDA/CPU combination.

---

## Running the full pipeline

### 1. Smoke test (optional, ~2 min on a laptop)

```bash
python run_seeds.py --seeds 2 --workers 2 --eps 5 --nodes 500 \
    --max-time 30 --n-rumours 3 --outdir smoke_test
```

Check that `smoke_test/aggregate.json` is produced before committing to the full run.

### 2. Full 20-seed paper run

```bash
python run_seeds.py \
    --seeds 20 --workers 4 --eps 80 --nodes 5000 \
    --max-time 60 --n-rumours 10 --n-evals 5 \
    --outdir multiseed_results
```

Or use the convenience script (runs smoke test first, then full run, then plots):

```bash
bash launch_full_run.sh
```

**Hardware note:** each worker needs ~4–6 GB RAM at 5000 nodes. With 4 workers,
expect 2–4 hours total on CPU. Use `--workers $(( $(nproc) / 2 ))` to scale.

### 3. Twitter SIR baseline

```bash
python run_twitter_sir.py
```

Reads `twitter_combined.txt`. Writes `outputs/twitter_sir_curves.json` and
`outputs/03_twitter_sir.png`. Fixed random seed (`np.random.seed(0)`).

### 4. Generate paper figures

```bash
python plot_aggregate.py --indir multiseed_results --outdir figures
python plot_sustained.py
```

Produces:
- `figures/01_I_curve_marl_vs_sir.png` — I-curve with ±1σ band (4 baselines)
- `figures/02_R_curve_marl_vs_sir.png` — R-curve with ±1σ band (4 baselines)
- `figures/03_metrics_barchart.png` — bar chart of all metrics with SEM error bars
- `figures/06_baselines_comparison.png` — peak I / final R / spread reduction comparison
- `figures/04_sustained_infection.png` — post-containment infection reduction
- `figures/paper_metrics_table.csv` — paper-ready CSV

### 5. LaTeX table

```bash
python make_paper_table.py
```

Writes `figures/paper_table.tex`.

### 6. Statistical significance tests

```bash
python significance.py
```

Runs paired t-tests and Wilcoxon signed-rank tests for all baseline pairs.
Output is printed and saved to `figures/significance.txt`.

---

## Reproducibility

- Fixed random seed (`np.random.seed(0)`) used for the Twitter SIR baseline.
- Full multi-seed runs: seeds 0–19, each seeding `random`, `numpy`, and `torch`
  independently so graph topology, node-type assignment, and initial infected
  nodes all vary across seeds.
- Per-seed result files are in `multiseed_results/seed_NNN.json`.
- Aggregate statistics (mean ± std ± SEM) are in `multiseed_results/aggregate.json`.
- 20 seeds × 5 evaluation rollouts per seed → 100 total evaluation episodes per
  condition reported in the paper.

---

## Re-aggregating without re-running

```bash
python run_seeds.py --aggregate-only --outdir multiseed_results
python plot_aggregate.py --indir multiseed_results --outdir figures
```

---

## Monitoring a run in progress

```bash
tail -f multiseed_results/logs/seed_000.log
```

Each seed writes its own log. The main process prints progress to stdout.
