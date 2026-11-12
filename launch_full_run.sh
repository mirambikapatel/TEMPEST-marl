#!/usr/bin/env bash
# Full 20-seed MARL-DQN paper run (main experiment, Table I)
# 20 seeds × 4 parallel workers × 5000 nodes × 80 episodes
# Expected wall time: ~1–3 h depending on hardware
# Logs: marl_multiseed/multiseed_results/logs/seed_NNN.log  (tail any to watch progress)

set -euo pipefail

# The scripts live in marl_multiseed/; run from there wherever this is called from.
cd "$(dirname "$0")/marl_multiseed"

OUTDIR="multiseed_results"

echo "=== Smoke test first (2 seeds, 500 nodes, 5 eps) ==="
python3 run_seeds.py \
    --seeds 2 --workers 2 --eps 5 --nodes 500 \
    --max-time 30 --n-rumours 3 \
    --outdir smoke_test

echo ""
echo "=== Smoke test passed — launching full run ==="
python3 run_seeds.py \
    --seeds   20   \
    --workers  4   \
    --eps     80   \
    --nodes 5000   \
    --max-time 60  \
    --n-rumours 10 \
    --n-evals   5  \
    --outdir "$OUTDIR"

echo ""
echo "=== Generating paper figures ==="
python3 plot_aggregate.py --indir "$OUTDIR" --outdir figures

echo ""
echo "Done. Figures in marl_multiseed/figures/  |  Metrics in marl_multiseed/$OUTDIR/aggregate.json"
