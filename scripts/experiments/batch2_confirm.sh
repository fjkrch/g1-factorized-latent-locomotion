#!/bin/bash
# Batch 2 Confirmation: V2 (wider_all) — 3 seeds × 3 sweeps
set -euo pipefail

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
EVAL="scripts/eval.py"
SWEEPS=("push_magnitude" "friction" "action_delay")
SEEDS=(42 43 44)

for SEED in "${SEEDS[@]}"; do
    # Find the checkpoint dir
    CKPT_DIR=$(find outputs/randomized/dynamite_wider_all/seed_${SEED} -name "best.pt" -path "*/checkpoints/*" | head -1)
    if [ -z "$CKPT_DIR" ]; then
        echo "ERROR: No best.pt found for seed $SEED"
        continue
    fi
    echo "=== Seed $SEED: checkpoint $CKPT_DIR ==="
    
    for SWEEP in "${SWEEPS[@]}"; do
        OUTDIR="results/batch2_confirm/wider_all/seed_${SEED}"
        echo "  Evaluating sweep: $SWEEP"
        $PYTHON $EVAL \
            --checkpoint "$CKPT_DIR" \
            --sweep "configs/sweeps/${SWEEP}.yaml" \
            --num_episodes 50 --seed "$SEED" \
            --output_dir "$OUTDIR"
        echo "  Done: $SWEEP"
    done
done

echo "All evaluations complete!"
