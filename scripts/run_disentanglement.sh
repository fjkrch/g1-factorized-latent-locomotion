#!/bin/bash
# Run disentanglement metrics (MIG/DCI/SAP) for all models and seeds.
# Usage: bash scripts/run_disentanglement.sh [NUM_EPISODES]

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_CMD="${PYTHON_CMD:-/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3}"
NUM_EPISODES="${1:-200}"
OUTPUT_DIR="results/mechanistic"
SEEDS=(42 43 44 45 46)

echo "=============================================="
echo "  Disentanglement Metrics: MIG / DCI / SAP"
echo "  Episodes: ${NUM_EPISODES}, Seeds: ${SEEDS[*]}"
echo "=============================================="

# ──────────────────────────────────────────────
# DynaMITE
# ──────────────────────────────────────────────
echo ""
echo ">>> DynaMITE (5 seeds)"
for seed in "${SEEDS[@]}"; do
    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}" -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
    if [[ -z "$CKPT" ]]; then
        echo "  [SKIP] No checkpoint for dynamite seed_${seed}"
        continue
    fi

    OUT_FILE="${OUTPUT_DIR}/disentanglement/disentanglement_dynamite_seed${seed}.json"
    if [[ -f "$OUT_FILE" ]]; then
        echo "  [SKIP] Already exists: $OUT_FILE"
        continue
    fi

    echo "  [RUN] dynamite seed=${seed}"
    $PYTHON_CMD scripts/disentanglement_metrics.py \
        --model_type dynamite \
        --ckpt "$CKPT" \
        --seed "$seed" \
        --num_episodes "$NUM_EPISODES" \
        --output_dir "$OUTPUT_DIR"
    echo "  [DONE] dynamite seed=${seed}"
done

# ──────────────────────────────────────────────
# LSTM
# ──────────────────────────────────────────────
echo ""
echo ">>> LSTM (5 seeds)"
for seed in "${SEEDS[@]}"; do
    CKPT=$(find "outputs/randomized/lstm_full/seed_${seed}" -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
    if [[ -z "$CKPT" ]]; then
        echo "  [SKIP] No checkpoint for lstm seed_${seed}"
        continue
    fi

    OUT_FILE="${OUTPUT_DIR}/disentanglement/disentanglement_lstm_seed${seed}.json"
    if [[ -f "$OUT_FILE" ]]; then
        echo "  [SKIP] Already exists: $OUT_FILE"
        continue
    fi

    echo "  [RUN] lstm seed=${seed}"
    $PYTHON_CMD scripts/disentanglement_metrics.py \
        --model_type lstm \
        --ckpt "$CKPT" \
        --seed "$seed" \
        --num_episodes "$NUM_EPISODES" \
        --output_dir "$OUTPUT_DIR"
    echo "  [DONE] lstm seed=${seed}"
done

# ──────────────────────────────────────────────
# Aggregate
# ──────────────────────────────────────────────
echo ""
echo ">>> Aggregating results..."
$PYTHON_CMD scripts/disentanglement_metrics.py --aggregate --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "  All disentanglement metrics complete!"
echo "=============================================="
