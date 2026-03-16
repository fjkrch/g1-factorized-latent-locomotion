#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all_main.sh — Run ALL main comparison experiments (4 methods × 4 tasks × 3 seeds)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_all_main.sh            # run all 16 experiments
#   bash scripts/run_all_main.sh --dry-run  # show commands without running
#   bash scripts/run_all_main.sh --skip-existing  # skip runs that already have completed manifests
#
# Total: 16 training runs (~1.5 hours on RTX 4060)
#
# After completion, run:
#   bash scripts/aggregate_all.sh
#   python scripts/plot_results.py --results results/aggregated/ --output figures/
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)        DRY_RUN=true; shift ;;
        --skip-existing)  SKIP_EXISTING=true; shift ;;
        *)                echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
SEEDS=(42)

# Activate conda env with Isaac Lab
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

TOTAL=$((${#TASKS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
CURRENT=0
SKIPPED=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  MAIN COMPARISON: ${#MODELS[@]} methods × ${#TASKS[@]} tasks × ${#SEEDS[@]} seeds = $TOTAL runs"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            LABEL="${task}/${model}_full/seed_${seed}"

            # ── Skip existing? ──
            if $SKIP_EXISTING; then
                EXISTING=$(find "outputs/${task}/${model}_full/seed_${seed}" -name "manifest.json" 2>/dev/null | head -1)
                if [[ -n "$EXISTING" ]]; then
                    STATUS=$(python -c "import json; print(json.load(open('$EXISTING')).get('status',''))" 2>/dev/null || echo "")
                    if [[ "$STATUS" == "completed" ]]; then
                        echo "[$CURRENT/$TOTAL] SKIP (completed): $LABEL"
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                fi
            fi

            echo ""
            echo "[$CURRENT/$TOTAL] RUNNING: $LABEL"

            if $DRY_RUN; then
                echo "  [DRY RUN] bash scripts/run_train.sh --task $task --model $model --seed $seed --variant full"
                continue
            fi

            bash scripts/run_train.sh --task "$task" --model "$model" --seed "$seed" --variant full || {
                echo "  ⚠ FAILED: $LABEL"
                FAILED=$((FAILED + 1))
            }
        done
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  MAIN COMPARISON COMPLETE"
echo "  Total: $TOTAL  |  Skipped: $SKIPPED  |  Failed: $FAILED"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED runs failed. Check logs."
    exit 1
fi
