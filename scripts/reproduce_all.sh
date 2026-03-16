#!/bin/bash
# =============================================================================
# Full reproduction script.
# Reproduces ALL experiments from scratch.
#
# Tiers:
#   1. Quick sanity: 1 method, 1 task, short run (~3 min)
#   2. Single full: 1 method, 1 task, full training (~6 min)
#   3. Main comparison: 4 methods, 4 tasks, 1 seed (~1.5 hours)
#   4. Full project: main + ablations + sweeps (~3 hours)
#
# Usage:
#   bash scripts/reproduce_all.sh --tier 1   # Quick sanity
#   bash scripts/reproduce_all.sh --tier 2   # Single full
#   bash scripts/reproduce_all.sh --tier 3   # Main comparison
#   bash scripts/reproduce_all.sh --tier 4   # Everything
# =============================================================================

set -e

TIER=${2:-1}  # Default to tier 1

echo "=================================================="
echo " DynaMITE — Full Reproduction (Tier ${TIER})"
echo "=================================================="

case $TIER in
    1)
        echo "[Tier 1] Quick sanity check (~3 min)"
        echo "Training DynaMITE on flat for 500k steps..."
        python scripts/train.py \
            --task configs/task/flat.yaml \
            --model configs/model/dynamite.yaml \
            --seed 42 \
            --set train.total_timesteps=500000 train.save_interval=100 train.eval_interval=50

        echo ""
        echo "[Tier 1] Evaluating..."
        LATEST=$(ls -td outputs/flat/dynamite/seed_42/*/ | head -1)
        python scripts/eval.py --run_dir "${LATEST}" --num_episodes 10

        echo "[Tier 1] Done. Check outputs in: ${LATEST}"
        ;;

    2)
        echo "[Tier 2] Single full experiment (~6 min)"
        python scripts/train.py \
            --task configs/task/randomized.yaml \
            --model configs/model/dynamite.yaml \
            --seed 42

        LATEST=$(ls -td outputs/randomized/dynamite/seed_42/*/ | head -1)
        python scripts/eval.py --run_dir "${LATEST}" --num_episodes 100
        echo "[Tier 2] Done."
        ;;

    3)
        echo "[Tier 3] Main comparison (~1.5 hours)"
        bash scripts/run_all_baselines.sh
        echo "[Tier 3] Done."
        ;;

    4)
        echo "[Tier 4] Full project (~3 hours)"
        bash scripts/run_all_baselines.sh
        bash scripts/run_ablations.sh
        bash scripts/run_sweeps.sh

        echo ""
        echo "Generating tables and plots..."
        python scripts/generate_tables.py --results_dir results/aggregated --output_dir figures/
        python scripts/plot_results.py --results_dir results/aggregated --output_dir figures/
        echo "[Tier 4] Done."
        ;;

    *)
        echo "Unknown tier: ${TIER}. Use 1, 2, 3, or 4."
        exit 1
        ;;
esac

echo ""
echo "Reproduction tier ${TIER} completed."
