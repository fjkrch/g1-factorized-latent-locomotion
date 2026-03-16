#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_ablations.sh — Run ALL ablation experiments (7 ablations × 3 seeds)
# ══════════════════════════════════════════════════════════════════════════════
#
# All ablations use task=randomized, model=dynamite as the base.
# Each ablation changes exactly one design decision.
#
# Usage:
#   bash scripts/run_ablations.sh
#   bash scripts/run_ablations.sh --dry-run
#   bash scripts/run_ablations.sh --skip-existing
#
# Total: 21 runs (~52 hours on RTX 4060)
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

TASK="randomized"
MODEL="dynamite"

ABLATIONS=(
    seq_len_4
    seq_len_16
    no_latent
    single_latent
    no_aux_loss
    depth_1
    depth_4
)
SEEDS=(42 43 44)

TOTAL=$((${#ABLATIONS[@]} * ${#SEEDS[@]}))
CURRENT=0
SKIPPED=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  ABLATIONS: ${#ABLATIONS[@]} variants × ${#SEEDS[@]} seeds = $TOTAL runs"
echo "  Base: task=$TASK, model=$MODEL"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for abl in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        LABEL="${TASK}/${MODEL}_${abl}/seed_${seed}"

        if $SKIP_EXISTING; then
            EXISTING=$(find "outputs/${TASK}/${MODEL}_${abl}/seed_${seed}" -name "manifest.json" 2>/dev/null | head -1)
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
            echo "  [DRY RUN] bash scripts/run_train.sh --task $TASK --model $MODEL --seed $seed --ablation $abl --variant $abl"
            continue
        fi

        bash scripts/run_train.sh --task "$TASK" --model "$MODEL" --seed "$seed" \
            --ablation "$abl" --variant "$abl" || {
            echo "  ⚠ FAILED: $LABEL"
            FAILED=$((FAILED + 1))
        }
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ABLATIONS COMPLETE"
echo "  Total: $TOTAL  |  Skipped: $SKIPPED  |  Failed: $FAILED"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED runs failed."
    exit 1
fi
