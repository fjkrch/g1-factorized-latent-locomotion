#!/usr/bin/env bash
# =============================================================================
# Ablation study: DynaMITE variants on randomized task
#
# 3 ablations × 3 seeds = 9 training runs
# Ablations:
#   no_aux_loss  – remove auxiliary dynamics-identification loss
#   no_latent    – remove latent head entirely (transformer-only)
#   single_latent – collapse factorized latent into one vector
#
# Seeds: 42, 43, 44
# Task: randomized
# Model: dynamite
# ~14 min per run → ~2 hours total (sequential)
# =============================================================================
set -euo pipefail

SEEDS=(42 43 44)
ABLATIONS=(no_aux_loss no_latent single_latent)
TASK=randomized
MODEL=dynamite

LOGDIR="logs/ablations"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#ABLATIONS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "============================================"
echo " Ablation study: ${TOTAL} runs"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for abl in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "[$COUNT/$TOTAL] ablation=${abl} seed=${seed} — $(date '+%H:%M:%S')"

        python3 scripts/train.py \
            --task "${TASK}" \
            --model "${MODEL}" \
            --ablation "configs/ablations/${abl}.yaml" \
            --variant "${abl}" \
            --seed "${seed}" \
            --headless \
            2>&1 | tee "${LOGDIR}/${abl}_seed${seed}.log"

        echo "[$COUNT/$TOTAL] DONE — $(date '+%H:%M:%S')"
    done
done

echo ""
echo "============================================"
echo " Ablation study complete: ${TOTAL}/${TOTAL}"
echo " End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
