#!/usr/bin/env bash
# =============================================================================
# Latent Validation: DynaMITE disentanglement analysis
#
# 3 seeds on randomized task = 3 analysis runs
# Seeds: 42, 43, 44
# Checkpoint source: outputs/randomized/dynamite_full/seed_{seed}/.../best.pt
# Episodes per run: 50
# =============================================================================
set -euo pipefail

SEEDS=(42 43 44)
NUM_EPISODES=50

LOGDIR="logs/latent"
mkdir -p "$LOGDIR"

TOTAL=${#SEEDS[@]}
COUNT=0

echo "============================================"
echo " Latent Validation: ${TOTAL} runs"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for seed in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))

    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}/" \
        -name "best.pt" -type f 2>/dev/null | head -1)

    if [[ -z "$CKPT" ]]; then
        echo "[$COUNT/$TOTAL] SKIP — no checkpoint for seed_${seed}"
        continue
    fi

    OUTDIR="results/latent_analysis/seed_${seed}"
    mkdir -p "$OUTDIR"

    echo ""
    echo "[$COUNT/$TOTAL] seed=${seed} — $(date '+%H:%M:%S')"

    python3 scripts/analyze_latent.py \
        --checkpoint "$CKPT" \
        --task configs/task/randomized.yaml \
        --output_dir "$OUTDIR" \
        --num_episodes "${NUM_EPISODES}" \
        --seed "${seed}" \
        2>&1 | tee "${LOGDIR}/latent_seed${seed}.log"

    echo "[$COUNT/$TOTAL] DONE — $(date '+%H:%M:%S')"
done

echo ""
echo "============================================"
echo " Latent Validation complete: ${TOTAL}/${TOTAL}"
echo " End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
