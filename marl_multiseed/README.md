# TEMPEST: Multi-Agent RL for Rumour Propagation on Social Networks

Code for the paper "TEMPEST: Modeling Peer Pressure as Social Contagion 
Using Multi-Agent Reinforcement Learning on Social Networks."

## Overview

TEMPEST is a multi-agent reinforcement learning framework for modelling 
rumour propagation and containment on social networks. Each node is an 
autonomous agent that chooses to Share, Refute, or Ignore information. 
The framework combines a peer-pressure contagion kernel, a warm-floor 
growth-then-fatigue intensity curve, a three-class heterogeneous 
node-type system with Evolutionary Game Theory payoffs, and a 
community-partitioned CTDE architecture with per-community GAT-GRU 
Q-networks.

## Code Structure

The repository contains only source code. Results (per-seed JSONs, 
aggregate metrics, figures) are reported in the paper, not stored 
in the repository. Running the code regenerates all outputs.

**Core code:**
- `marl_core.py` — core functions: environment, MARL system, SIR baseline, GraphCache
- `finaloutbreak.ipynb` — main experimental notebook

**Runners and scripts:**
- `run_seeds.py` — multi-seed BA experiment runner
- `run_twitter_sir.py` — classical SIR baseline on Twitter graph
- `launch_full_run.sh` — orchestrates the full 20-seed pipeline

**Analysis and plotting:**
- `plot_aggregate.py` — aggregate metrics visualization
- `plot_sustained.py` — sustained infection reduction plots
- `significance.py` — statistical significance tests
- `make_paper_table.py` — generates paper tables from result JSONs

**Baseline additions:**
- `add_greedy_to_existing.py` — adds greedy heuristic to result files
- `add_centralized_to_existing.py` — adds centralised DQN baseline

## Datasets

- **Barabási–Albert graphs**: generated internally via NetworkX
- **Stanford SNAP Twitter ego-network**: download `twitter_combined.txt` 
  from https://snap.stanford.edu/data/ego-Twitter.html and place in 
  the project root

## Running the Code

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the main 20-seed experiment:
```bash
bash launch_full_run.sh
```

Or run individual components:
```bash
python run_seeds.py           # multi-seed BA training
python run_twitter_sir.py     # Twitter SIR baseline
python significance.py        # paired t-tests
python make_paper_table.py    # generate paper tables
```

Multi-seed runs take multiple hours per seed. For a smoke test, 
use `run_seeds.py` with a reduced seed count.

## Reproducibility Notes

- Fixed random seed used where applicable (`np.random.seed(0)`)
- Aggregate results reported in the paper are over 20 independent seeds
- Trained model weights are not persisted; retraining is required to 
  reproduce exact per-seed curves. Aggregate statistics are stable 
  across retraining runs.follow the official install guide at https://pytorch-geometric.readthedocs.io/
for your exact torch + CUDA/CPU combination.

## Hardware and Runtime
All experiments were run on a MacBook Pro with an Apple M3Pro chip
(11-core CPU, 14-core GPU) and 18GB unified memory. GPU acceleration
was provided by PyTorch's Metal Performance Shaders (MPS) backend.
A single seed of TEMPEST training on a 5000-node Barab\'asi--Albert
graph for 80 episodes completes in approximately 23 minutes. Inference
on the Twitter ego-network ($N = 81,306) completes in under a minute
on the same hardware without retraining.

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
