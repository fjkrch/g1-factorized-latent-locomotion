#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_retrain_push_tasks.sh — Retrain tasks affected by deterministic push fix
#
# Only retrains push, randomized, terrain (flat is unaffected since push_vel=0).
# Also retrains 7 ablations on randomized.
#
# Total: 12 main + 7 ablations = 19 runs (~3.5 hours on RTX 4060)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

echo "══════════════════════════════════════════════════════════════"
echo "  RETRAIN: Push-affected tasks (deterministic push fix)"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

SEED=42
FAILED=0
CURRENT=0

# ── Phase 1: Main runs (3 tasks × 4 models = 12 runs) ──────────────────
TASKS=(push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
TOTAL_MAIN=$((${#TASKS[@]} * ${#MODELS[@]}))

echo ""
echo "Phase 1: Main runs ($TOTAL_MAIN runs)"
echo "────────────────────────────────────────────────────────────"

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[${CURRENT}/${TOTAL_MAIN}] Training ${model} on ${task} (seed=${SEED})"
        echo "  $(date -Iseconds)"

        if python scripts/train.py \
            --task "$task" \
            --model "$model" \
            --seed "$SEED" \
            --variant full; then
            echo "  ✓ Done"
        else
            echo "  ✗ FAILED (exit $?)"
            FAILED=$((FAILED + 1))
        fi
    done
done

# ── Phase 2: Ablations (7 runs on randomized) ──────────────────────────
ABLATIONS=(
    "no_aux_loss"
    "no_latent"
    "single_latent"
    "depth_1"
    "depth_4"
    "seq_len_4"
    "seq_len_16"
)
TOTAL_ABL=${#ABLATIONS[@]}

echo ""
echo "Phase 2: Ablations ($TOTAL_ABL runs on randomized)"
echo "────────────────────────────────────────────────────────────"

for i in "${!ABLATIONS[@]}"; do
    abl="${ABLATIONS[$i]}"
    echo ""
    echo "[$(($i + 1))/${TOTAL_ABL}] Ablation: ${abl} (seed=${SEED})"
    echo "  $(date -Iseconds)"

    if python scripts/train.py \
        --task randomized \
        --model dynamite \
        --ablation "$abl" \
        --variant "$abl" \
        --seed "$SEED"; then
        echo "  ✓ Done"
    else
        echo "  ✗ FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  RETRAIN COMPLETE: $(date -Iseconds)"
echo "  Total: $((TOTAL_MAIN + TOTAL_ABL)) runs, $FAILED failed"
echo "══════════════════════════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
