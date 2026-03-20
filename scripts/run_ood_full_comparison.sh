#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_ood_full_comparison.sh — Full OOD sweep: 4 models × 5 seeds × 3 sweeps
# ══════════════════════════════════════════════════════════════════════════════
#
# Extends OOD sweeps from 2 models × 3 seeds to 4 models × 5 seeds.
# Skips runs that already exist in results/sweeps_multiseed/.
#
# Total matrix: 4 × 5 × 3 = 60 sweep evals
# Already done:  2 × 3 × 3 = 18 (dynamite/lstm seeds 42-44)
# New runs:                   42
#
# Estimated time: ~2.5 hours on RTX 4060 Laptop
#
# Usage:
#   bash scripts/run_ood_full_comparison.sh
#   bash scripts/run_ood_full_comparison.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Isaac Lab Python — required for real sim, not mock env
ISAAC_PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "ERROR: Isaac Lab Python not found at $ISAAC_PYTHON"
    exit 1
fi
export PYTHON_CMD="$ISAAC_PYTHON"

TASK="randomized"
MODELS=(mlp lstm transformer dynamite)
SEEDS=(42 43 44 45 46)
SWEEPS=(friction push_magnitude action_delay)
NUM_EPISODES=50
EVAL_SEED=42

TOTAL=0
SKIPPED=0
RUN=0
FAILED=0

# Count total and skipped
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for sweep in "${SWEEPS[@]}"; do
            TOTAL=$((TOTAL + 1))
            OUT_DIR="results/sweeps_multiseed/${sweep}/${model}_seed${seed}"
            if [[ -f "${OUT_DIR}/sweep_${sweep}.json" ]]; then
                SKIPPED=$((SKIPPED + 1))
            fi
        done
    done
done

NEW=$((TOTAL - SKIPPED))

echo "══════════════════════════════════════════════════════════════"
echo "  FULL OOD COMPARISON"
echo "  Matrix: ${#MODELS[@]} models × ${#SEEDS[@]} seeds × ${#SWEEPS[@]} sweeps = $TOTAL total"
echo "  Already done: $SKIPPED  |  New runs: $NEW"
echo "  Episodes per level: $NUM_EPISODES"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

CURRENT=0

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Find checkpoint
        CKPT=$(find "outputs/${TASK}/${model}_full/seed_${seed}" -name "best.pt" 2>/dev/null | head -1)
        if [[ -z "$CKPT" ]]; then
            echo "  ⚠ MISSING checkpoint: $model seed_$seed — skipping all sweeps"
            FAILED=$((FAILED + 3))
            CURRENT=$((CURRENT + 3))
            continue
        fi

        for sweep in "${SWEEPS[@]}"; do
            CURRENT=$((CURRENT + 1))
            OUT_DIR="results/sweeps_multiseed/${sweep}/${model}_seed${seed}"

            # Skip if already done
            if [[ -f "${OUT_DIR}/sweep_${sweep}.json" ]]; then
                echo "[$CURRENT/$TOTAL] SKIP (exists): $model seed$seed $sweep"
                continue
            fi

            RUN=$((RUN + 1))
            echo ""
            echo "[$CURRENT/$TOTAL] RUN #$RUN/$NEW: $model seed$seed $sweep"

            mkdir -p "$OUT_DIR"

            if $DRY_RUN; then
                echo "  [DRY RUN] bash scripts/run_eval.sh --checkpoint $CKPT --sweep $sweep --output-dir $OUT_DIR --num-episodes $NUM_EPISODES --seed $EVAL_SEED"
                continue
            fi

            bash scripts/run_eval.sh \
                --checkpoint "$CKPT" \
                --sweep "$sweep" \
                --output-dir "$OUT_DIR" \
                --num-episodes "$NUM_EPISODES" \
                --seed "$EVAL_SEED" || {
                echo "  ⚠ FAILED: $model seed$seed $sweep"
                FAILED=$((FAILED + 1))
            }
        done
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FULL OOD COMPARISON COMPLETE"
echo "  Total: $TOTAL  |  Skipped: $SKIPPED  |  Ran: $RUN  |  Failed: $FAILED"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
