#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_post_training.sh — Run ALL post-training analysis (Items 4-9)
# ══════════════════════════════════════════════════════════════════════════════
#
# Assumes training (Items 2-3, 5-6) is complete:
#   - 48 main runs:    4 models × 4 tasks × 3 seeds = outputs/{task}/{model}_full/seed_*/
#   - 21 ablation runs: 7 ablations × 3 seeds = outputs/randomized/dynamite_{abl}/seed_*/
#
# This script runs:
#   Item 4: Evaluate all checkpoints on all 4 tasks
#   Item 5: OOD friction sweep (randomized task)
#   Item 7: OOD push magnitude sweep (randomized task)
#   Item 8: Disentanglement analysis (latent_analysis.py)
#   Item 9: Extra OOD sweeps (motor_strength, action_delay)
#   Final:  Aggregate all results + generate tables/figures
#
# Usage:
#   bash scripts/run_post_training.sh
#   bash scripts/run_post_training.sh --dry-run
#   bash scripts/run_post_training.sh --skip-eval     # skip standard eval, only OOD
#   bash scripts/run_post_training.sh --only-aggregate # only aggregation
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
SKIP_EVAL=false
ONLY_AGGREGATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)          DRY_RUN=true; shift ;;
        --skip-eval)        SKIP_EVAL=true; shift ;;
        --only-aggregate)   ONLY_AGGREGATE=true; shift ;;
        *)                  echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON="${DYNAMITE_PYTHON:-python}"
TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
SEEDS=(42 43 44)
OOD_TASK="randomized"
OOD_SWEEPS=(friction push_magnitude motor_strength action_delay)
ABLATIONS=(seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4)
NUM_EVAL_EPISODES=20
FAILED=0
TOTAL_OPS=0

run_cmd() {
    local label="$1"
    local cmd="$2"
    TOTAL_OPS=$((TOTAL_OPS + 1))
    if $DRY_RUN; then
        echo "  [DRY RUN] $cmd"
        return 0
    fi
    echo "  [$label] $cmd"
    eval "$cmd" || {
        echo "  ⚠ FAILED: $label"
        FAILED=$((FAILED + 1))
    }
}

# Helper: find latest run directory for a given task/model/variant/seed
find_run_dir() {
    local task="$1" model="$2" variant="$3" seed="$4"
    find "outputs/${task}/${model}_${variant}/seed_${seed}" -maxdepth 1 -type d 2>/dev/null | sort | tail -1
}

find_checkpoint() {
    local run_dir="$1"
    if [[ -f "${run_dir}/checkpoints/best.pt" ]]; then
        echo "${run_dir}/checkpoints/best.pt"
    elif [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
        echo "${run_dir}/checkpoints/latest.pt"
    else
        echo ""
    fi
}

echo "══════════════════════════════════════════════════════════════"
echo "  POST-TRAINING ANALYSIS PIPELINE"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

if $ONLY_AGGREGATE; then
    echo ""
    echo "═══ AGGREGATION ONLY ═══"
    run_cmd "aggregate" "$PYTHON scripts/aggregate_seeds.py --root outputs --output results/aggregated"
    run_cmd "tables" "$PYTHON scripts/generate_tables.py --results results/aggregated --output results/tables"
    run_cmd "figures" "$PYTHON scripts/plot_results.py --results results/aggregated --output figures"
    echo "  Done. Total ops: $TOTAL_OPS, Failed: $FAILED"
    exit $FAILED
fi

# ══════════════════════════════════════════════════════════════════════════════
# ITEM 4: Standard evaluation — all models on all tasks
# ══════════════════════════════════════════════════════════════════════════════
if ! $SKIP_EVAL; then
    echo ""
    echo "═══ ITEM 4: STANDARD EVALUATION ═══"
    echo "  ${#MODELS[@]} models × ${#TASKS[@]} tasks × ${#SEEDS[@]} seeds"

    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                RUN_DIR=$(find_run_dir "$task" "$model" "full" "$seed")
                if [[ -z "$RUN_DIR" ]]; then
                    echo "  ⚠ SKIP: no run dir for ${task}/${model}/seed_${seed}"
                    continue
                fi
                CKPT=$(find_checkpoint "$RUN_DIR")
                if [[ -z "$CKPT" ]]; then
                    echo "  ⚠ SKIP: no checkpoint for ${task}/${model}/seed_${seed}"
                    continue
                fi

                # Eval on same task (in-distribution)
                EVAL_OUT="${RUN_DIR}/eval"
                run_cmd "${task}/${model}/s${seed}" \
                    "$PYTHON scripts/eval.py --checkpoint $CKPT --task configs/task/${task}.yaml --num_episodes $NUM_EVAL_EPISODES --output_dir $EVAL_OUT"
            done
        done
    done

    # ── Ablation eval ──
    echo ""
    echo "═══ ABLATION EVALUATION ═══"
    for abl in "${ABLATIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_DIR=$(find_run_dir "$OOD_TASK" "dynamite" "$abl" "$seed")
            if [[ -z "$RUN_DIR" ]]; then
                echo "  ⚠ SKIP: no run dir for ablation=${abl}/seed_${seed}"
                continue
            fi
            CKPT=$(find_checkpoint "$RUN_DIR")
            if [[ -z "$CKPT" ]]; then
                echo "  ⚠ SKIP: no checkpoint for ablation=${abl}/seed_${seed}"
                continue
            fi
            EVAL_OUT="${RUN_DIR}/eval"
            run_cmd "abl/${abl}/s${seed}" \
                "$PYTHON scripts/eval.py --checkpoint $CKPT --task configs/task/${OOD_TASK}.yaml --num_episodes $NUM_EVAL_EPISODES --output_dir $EVAL_OUT"
        done
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# ITEMS 5, 7, 9: OOD sweeps (friction, push_magnitude, motor_strength, action_delay)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ ITEMS 5/7/9: OOD SWEEPS ═══"
echo "  ${#MODELS[@]} models × ${#OOD_SWEEPS[@]} sweeps × ${#SEEDS[@]} seeds"

for model in "${MODELS[@]}"; do
    for sweep in "${OOD_SWEEPS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_DIR=$(find_run_dir "$OOD_TASK" "$model" "full" "$seed")
            if [[ -z "$RUN_DIR" ]]; then
                echo "  ⚠ SKIP: no run dir for ${model}/seed_${seed}"
                continue
            fi
            CKPT=$(find_checkpoint "$RUN_DIR")
            if [[ -z "$CKPT" ]]; then
                echo "  ⚠ SKIP: no checkpoint for ${model}/seed_${seed}"
                continue
            fi

            OOD_OUT="outputs/ood_validated/${OOD_TASK}/${model}_full/seed_${seed}/${sweep}"
            run_cmd "ood/${model}/${sweep}/s${seed}" \
                "$PYTHON scripts/eval_ood_validated.py --checkpoint $CKPT --sweep configs/sweeps/${sweep}.yaml --output_dir $OOD_OUT --num_episodes $NUM_EVAL_EPISODES"
        done
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# ITEM 8: Disentanglement analysis
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ ITEM 8: DISENTANGLEMENT ANALYSIS ═══"

for seed in "${SEEDS[@]}"; do
    RUN_DIR=$(find_run_dir "$OOD_TASK" "dynamite" "full" "$seed")
    if [[ -z "$RUN_DIR" ]]; then
        echo "  ⚠ SKIP: no DynaMITE run for seed_${seed}"
        continue
    fi
    CKPT=$(find_checkpoint "$RUN_DIR")
    if [[ -z "$CKPT" ]]; then
        echo "  ⚠ SKIP: no DynaMITE checkpoint for seed_${seed}"
        continue
    fi
    ANALYSIS_OUT="results/latent_analysis/seed_${seed}"
    mkdir -p "$ANALYSIS_OUT"
    run_cmd "latent/s${seed}" \
        "$PYTHON scripts/analyze_latent.py --checkpoint $CKPT --task configs/task/${OOD_TASK}.yaml --output_dir $ANALYSIS_OUT --num_episodes 50 --seed $seed"
done

# ══════════════════════════════════════════════════════════════════════════════
# FINAL: Aggregate results
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══ AGGREGATION ═══"
run_cmd "aggregate" "$PYTHON scripts/aggregate_seeds.py --root outputs --output results/aggregated"
run_cmd "tables" "$PYTHON scripts/generate_tables.py --results results/aggregated --output results/tables"
run_cmd "figures" "$PYTHON scripts/plot_results.py --results results/aggregated --output figures"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  POST-TRAINING ANALYSIS COMPLETE"
echo "  Total operations: $TOTAL_OPS | Failed: $FAILED"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED operations failed. Check logs above."
    exit 1
fi
