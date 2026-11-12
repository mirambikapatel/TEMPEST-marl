# TEMPEST: Multi-Agent RL for Rumour Propagation on Social Networks

Code for the paper "TEMPEST: Modeling Peer Pressure as Social Contagion
Using Multi-Agent Reinforcement Learning on Social Networks"
(M. Patel, Md. Arquam, S. Kumari; under revision at IEEE Transactions on
Computational Social Systems).

## Overview

TEMPEST is a multi-agent reinforcement learning framework for modelling
rumour propagation and containment on social networks. Each node is an
autonomous agent that chooses to Share, Refute or Ignore. The framework
combines a peer-pressure contagion kernel, a warm-floor growth-then-fatigue
intensity curve, three behavioural node types with Evolutionary Game Theory
payoffs, and a community-partitioned CTDE architecture with per-community
GAT-GRU Q-networks.

All code is in `marl_multiseed/`. Results (per-seed JSONs, trained models,
figures) are not stored in the repository; running the scripts regenerates
them.

## Code structure

**Core**
- `marl_core.py`: environment, contagion kernel, MARL system (GAT-GRU
  Q-networks, CTDE training), classical SIR baseline, greedy heuristic,
  centralised DQN, `GraphCache`
- `finaloutbreak.ipynb`: original experimental notebook (BA training,
  figures 02/02b, post-hoc window analysis). See the note on the Twitter
  cell below.

**Main 20-seed experiment (Table I)**
- `run_seeds.py`: multi-seed BA runner (TEMPEST vs. classical SIR)
- `add_greedy_to_existing.py`, `add_centralized_to_existing.py`: add the
  greedy heuristic and the centralised DQN to the per-seed results
- `significance.py`, `make_paper_table.py`, `plot_aggregate.py`,
  `plot_sustained.py`: statistics, table and plots for the main run
- `launch_full_run.sh` (repository root): smoke test, then the full 20-seed run

**Revision experiments**
- `baselines.py`: QDOM-adapted and DRLE-adapted learned-intervention
  baselines, the no-learning population, refuted/spontaneous tagging of every
  I→R transition, signed polarization
- `revision_runner.py`: runs every revision experiment (resumable, parallel)
- `summarise_v2.py`: turns `results_v2/` into tables (mean ± SEM, paired
  t-tests against TEMPEST)
- `REVISION_RUN_INSTRUCTIONS.md`: step-by-step instructions for the above

**PHEME rumour events**
- `pheme_prepare.py`: builds each event's follower network (largest connected
  component) and the real sharing/refutation curves
- `pheme_runner.py`: trains TEMPEST from scratch on each event network
- `summarise_pheme.py`: PHEME table (mean ± SEM, paired t-tests)
- `pheme_compare_v2.py`: shape comparison (RMSE) between simulated and real
  cumulative sharing and refutation, with the veracity filter

**Twitter zero-shot transfer**
- `run_twitter_sir.py`: classical SIR baseline on the Twitter graph
- `run_twitter_transfer_fixed.py`: trains TEMPEST on a BA graph and applies it,
  without retraining, to the Twitter graph with communities detected on the
  Twitter graph itself (Louvain if greedy modularity does not finish within an
  hour)

**Figures**
- `make_paper_figures.py`: the figures added in the revision
  (`fig_intensity_T.png`, `fig_ablation_20seeds.png`,
  `fig_learned_interventions.png`)

## Datasets

Place the files at these paths, relative to `marl_multiseed/`:

| Dataset | Source | Path |
|---|---|---|
| Barabási–Albert, Erdős–Rényi, Watts–Strogatz | generated with NetworkX | none |
| SNAP Twitter ego-network (`twitter_combined.txt`) | https://snap.stanford.edu/data/ego-Twitter.html | `twitter_combined.txt` (or `~/Downloads/`) |
| SNAP Facebook (`facebook_combined.txt`) | https://snap.stanford.edu/data/ego-Facebook.html | `OUR_program/outbreak/facebook_combined.txt` |
| PHEME rumour scheme dataset | figshare article 2068650 | `OUR_program/pheme/raw/pheme-rumour-scheme-dataset/` |

The Twitter graph is used as undirected: 81,306 nodes and 1,342,310 edges.
For PHEME, the five English events whose largest connected component has at
least 1,000 users are used (Charlie Hebdo, Ottawa shooting, Sydney siege,
Ferguson, Germanwings crash).

## Installation

```bash
cd marl_multiseed
pip install -r requirements.txt
```

`torch-geometric` wheels are build-specific. If `pip install` fails, follow
the official guide at https://pytorch-geometric.readthedocs.io/ for your
torch + CUDA/CPU combination.

## Reproducing the paper

Run everything from `marl_multiseed/`.

### 1. Main 20-seed experiment

```bash
python run_seeds.py --seeds 20 --workers 4 --eps 80 --nodes 5000 \
    --max-time 60 --n-rumours 10 --n-evals 5 --outdir multiseed_results
python add_greedy_to_existing.py
python add_centralized_to_existing.py
python significance.py
python make_paper_table.py
```

or `bash ../launch_full_run.sh`, which runs a smoke test first.

### 2. Revision experiments

```bash
python revision_runner.py --smoke --workers 4              # quick check
python revision_runner.py --include-optional --seeds 20 --workers 5
python summarise_v2.py                                     # results_v2/summary.md
```

Several experiments (baselines, ablations) are compared seed by seed with the
main run, so run step 1 first; `revision_runner.py` does not modify
`multiseed_results/`. The runner is resumable: run the same command again after
an interruption and finished jobs are skipped. `python revision_runner.py --list` prints the plan.

### 3. PHEME

```bash
python pheme_prepare.py
python pheme_runner.py --seeds 20 --workers 5
python summarise_pheme.py
python pheme_compare_v2.py
```

### 4. Twitter transfer

```bash
python run_twitter_sir.py               # writes outputs/twitter_sir_curves.json
python run_twitter_transfer_fixed.py    # writes outputs/twitter_transfer_fixed/
```

`run_twitter_transfer_fixed.py` reads the SIR baseline written by
`run_twitter_sir.py`, so run that first.

### 5. Figures

```bash
python make_paper_figures.py            # writes figures/
```

### Where each result in the paper comes from

| Paper result | Experiment | Script that prints or plots it |
|---|---|---|
| Comparison of SIR, greedy heuristic, centralised DQN and TEMPEST | `multiseed_results/` | `make_paper_table.py`, `significance.py` |
| Signed polarization, share of R from active refutation | `results_v2/r_composition`, `results_v2/baselines` | `summarise_v2.py` |
| Post-hoc window analysis (65.8–92.3%) | `multiseed_results/` | `finaloutbreak.ipynb`, CONTAIN_T window-sensitivity cell |
| TEMPEST retrained with different T_contain | `results_v2/T_sweep` | `summarise_v2.py` |
| Parameter sensitivity | `results_v2/sensitivity` | `summarise_v2.py` |
| Number of partitions K | `results_v2/k_sweep` | `summarise_v2.py` |
| SIR parameter grid | `results_v2/sir_grid` | `summarise_v2.py` |
| Component ablations | `results_v2/ablation` | `summarise_v2.py`, `make_paper_figures.py` |
| Comparison with QDOM and DRLE | `results_v2/baselines` | `summarise_v2.py`, `make_paper_figures.py` |
| ER, WS and Facebook networks | `results_v2/networks` | `summarise_v2.py` |
| PHEME events and shape RMSE | `results_v2/pheme` | `summarise_pheme.py`, `pheme_compare_v2.py` |
| Twitter zero-shot transfer | `outputs/twitter_transfer_fixed/` | `run_twitter_transfer_fixed.py` (`REPORT.md`, `summary.json`) |
| Rumour intensity and mean infected curve | `multiseed_results/` | `make_paper_figures.py` |

## Notes

- **Metrics no longer reported.** `marl_core.py`, `make_paper_table.py` and
  `plot_aggregate.py` still compute two metrics from the original submission:
  influential efficiency and the symmetric polarization |I − R|/(I + R).
  Neither is reported in the revised paper. Influential efficiency was removed
  because the PageRank weight cancels in its ratio, so it does not measure
  influential nodes specifically. Polarization was replaced by the signed
  polarization (R − I)/(R + I), computed in `baselines.py` and reported by
  `summarise_v2.py`.
- **Twitter cell in `finaloutbreak.ipynb`.** The original transfer cell
  assigns communities with `marl_ba.node_community[:gc_tw.n]`, which covers
  only the first 5,000 Twitter users; the other 76,306 never act on the
  policy. The results in the revised paper come from
  `run_twitter_transfer_fixed.py`, which detects communities on the Twitter
  graph itself.
- **Fixed environment parameters.** The contagion and refutation parameters
  (α_peer = 0.70, β_base = 0.04, p_contain = 0.40, p_growth = 0.10,
  p_casc = 0.12, p_spont = 0.03), the node-type prior and the payoff matrix are
  set in `marl_core.py`. `revision_runner.py` overrides them per experiment for
  the sensitivity analysis.

## Hardware and runtime

All experiments were run on a MacBook Pro with an Apple M3 Pro chip (11-core
CPU, 14-core GPU) and 18 GB unified memory, using PyTorch's Metal Performance
Shaders (MPS) backend.

- TEMPEST training on a 5,000-node BA graph for 80 episodes: about 23 minutes
  per seed.
- Twitter transfer (N = 81,306): community detection about 2.5 hours (once),
  then about 6 minutes per evaluation rollout, without retraining.

## Reproducibility

- Seeds 0–19; each seed sets `random`, `numpy` and `torch`, so the graph,
  node types and initial infected nodes vary across seeds.
- 20 seeds × 5 evaluation rollouts per seed in the main experiment.
- `revision_runner.py` and `pheme_runner.py` save trained models under
  `results_v2/<experiment>/<variant>/models/`, so tables can be recomputed
  without retraining.
