#!/bin/bash
# =============================================================================
# Run ablation experiments on the 'randomized' task.
# Tests each design choice of DynaMITE.
#
# Expected runtime: ~42 min total on RTX 4060
# Per run: ~6 min
# Total runs: 7 ablations x 3 seeds = 21 runs (on 'randomized' task)
# =============================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SEEDS=(42 43 44)
TASK="randomized"

ABLATIONS=(
    "seq_len_4"
    "seq_len_16"
    "no_latent"
    "single_latent"
    "no_aux_loss"
    "depth_1"
    "depth_4"
)

run_cmd() {
    local cmd="$1"
    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
    else
        echo "[RUN] $cmd"
        eval "$cmd"
    fi
}

echo "=== DynaMITE: Ablation Experiments ==="
echo "Task: ${TASK}"
echo "Ablations: ${ABLATIONS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Total runs: $((${#ABLATIONS[@]} * ${#SEEDS[@]}))"
echo "======================================="

for ablation in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "--- ablation=${ablation} / seed=${seed} ---"
        run_cmd "bash scripts/run_train.sh \
            --task ${TASK} \
            --model dynamite \
            --ablation ${ablation} \
            --variant ${ablation} \
            --seed ${seed}"
    done
done

echo ""
echo "=== All ablation runs completed ==="
