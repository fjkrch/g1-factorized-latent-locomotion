#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_robustness.sh — Run robustness sweep evaluations for all methods
# ══════════════════════════════════════════════════════════════════════════════
#
# Evaluates trained checkpoints under varied perturbation levels.
# Requires completed training on the randomized task.
#
# Usage:
#   bash scripts/run_robustness.sh
#   bash scripts/run_robustness.sh --dry-run
#   bash scripts/run_robustness.sh --seed 42        # single seed only
#
# Total: 4 methods × 4 sweeps × 1 seed = 16 sweep evaluations (~4 hours)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
EVAL_SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        --seed)     EVAL_SEED="$2"; shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TASK="randomized"
MODELS=(mlp lstm transformer dynamite)
SWEEPS=(push_magnitude friction motor_strength action_delay)

TOTAL=$((${#MODELS[@]} * ${#SWEEPS[@]}))
CURRENT=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  ROBUSTNESS SWEEPS: ${#MODELS[@]} methods × ${#SWEEPS[@]} sweeps = $TOTAL evaluations"
echo "  Base task: $TASK, seed: $EVAL_SEED"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for model in "${MODELS[@]}"; do
    for sweep in "${SWEEPS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL] SWEEP: model=$model, sweep=$sweep"

        # Find the run directory
        RUN_DIR=$(find "outputs/${TASK}/${model}_full/seed_${EVAL_SEED}" -maxdepth 1 -type d 2>/dev/null | sort | tail -1)

        if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
            echo "  ⚠ SKIP: no run dir found for $model on $TASK seed_${EVAL_SEED}"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Check for checkpoint
        CKPT="$RUN_DIR/checkpoints/best.pt"
        if [[ ! -f "$CKPT" ]]; then
            CKPT="$RUN_DIR/checkpoints/latest.pt"
        fi
        if [[ ! -f "$CKPT" ]]; then
            echo "  ⚠ SKIP: no checkpoint found in $RUN_DIR"
            FAILED=$((FAILED + 1))
            continue
        fi

        OUTPUT_DIR="results/sweeps/${model}_full/${sweep}"
        mkdir -p "$OUTPUT_DIR"

        if $DRY_RUN; then
            echo "  [DRY RUN] bash scripts/run_eval.sh --checkpoint $CKPT --sweep $sweep --output-dir $OUTPUT_DIR"
            continue
        fi

        bash scripts/run_eval.sh --checkpoint "$CKPT" --sweep "$sweep" --output-dir "$OUTPUT_DIR" || {
            echo "  ⚠ FAILED: $model/$sweep"
            FAILED=$((FAILED + 1))
        }
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ROBUSTNESS SWEEPS COMPLETE"
echo "  Total: $TOTAL  |  Failed: $FAILED"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
