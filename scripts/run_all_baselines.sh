#!/bin/bash
# =============================================================================
# Run all baseline and proposed method experiments.
# Each method is trained on all 4 tasks with 3 seeds.
#
# Expected runtime: ~120 hours total on RTX 4060
# Per run: ~2.5 hours (50M steps, 2048 envs)
# Total runs: 4 methods x 4 tasks x 3 seeds = 48 runs
#
# Usage:
#   bash scripts/run_all_baselines.sh
#   bash scripts/run_all_baselines.sh --dry-run    # print commands only
# =============================================================================

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

SEEDS=(42 43 44)
TASKS=("flat" "push" "randomized" "terrain")
MODELS=("mlp" "lstm" "transformer" "dynamite")

run_cmd() {
    local cmd="$1"
    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
    else
        echo "[RUN] $cmd"
        eval "$cmd"
    fi
}

echo "=== DynaMITE: Full Baseline Training ==="
echo "Models: ${MODELS[*]}"
echo "Tasks:  ${TASKS[*]}"
echo "Seeds:  ${SEEDS[*]}"
echo "Total runs: $((${#MODELS[@]} * ${#TASKS[@]} * ${#SEEDS[@]}))"
echo "========================================="

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo ""
            echo "--- ${model} / ${task} / seed=${seed} ---"
            run_cmd "python scripts/train.py \
                --task configs/task/${task}.yaml \
                --model configs/model/${model}.yaml \
                --seed ${seed}"
        done
    done
done

echo ""
echo "=== All baseline runs completed ==="
