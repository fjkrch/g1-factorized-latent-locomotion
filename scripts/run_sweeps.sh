#!/bin/bash
# =============================================================================
# Run robustness sweep evaluations.
# Evaluates trained checkpoints under varying perturbation conditions.
#
# Prerequisites: trained checkpoints must exist for all methods on 'randomized' task.
#
# Expected runtime: ~20 min total (evaluation only)
# =============================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

MODELS=("mlp" "lstm" "transformer" "dynamite")
SWEEPS=("push_magnitude" "friction" "motor_strength" "action_delay")
SEED=42
TASK="randomized"

run_cmd() {
    local cmd="$1"
    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
    else
        echo "[RUN] $cmd"
        eval "$cmd"
    fi
}

echo "=== DynaMITE: Robustness Sweeps ==="

for model in "${MODELS[@]}"; do
    # Find best checkpoint for this model
    CKPT_DIR="outputs/${TASK}/${model}/seed_${SEED}"
    LATEST_RUN=$(ls -td "${CKPT_DIR}"/*/ 2>/dev/null | head -1)

    if [[ -z "$LATEST_RUN" ]]; then
        echo "WARNING: No run found for ${model} on ${TASK}. Skipping."
        continue
    fi

    CKPT="${LATEST_RUN}checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="${LATEST_RUN}checkpoints/latest.pt"
    fi

    for sweep in "${SWEEPS[@]}"; do
        echo ""
        echo "--- ${model} / sweep=${sweep} ---"
        run_cmd "python scripts/eval.py \
            --checkpoint ${CKPT} \
            --sweep configs/sweeps/${sweep}.yaml \
            --output_dir results/sweeps/${model}/${sweep} \
            --num_episodes 20"
    done
done

echo ""
echo "=== All sweeps completed ==="
