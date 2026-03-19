#!/usr/bin/env bash
# =============================================================================
# OOD Robustness Sweeps: DynaMITE vs LSTM on randomized checkpoints
#
# 3 sweep types × 2 models × 3 seeds = 18 eval runs
# Sweeps: friction, push_magnitude, action_delay
# Models: dynamite, lstm (top-2 from deterministic eval)
# Seeds: 42, 43, 44
# Checkpoint source: outputs/randomized/{model}_full/seed_{seed}/.../best.pt
# Episodes per sweep point: 50
#
# Each sweep run evaluates across all parameter levels in the sweep config
# and saves a sweep_{name}.json file.
# =============================================================================
set -euo pipefail

SEEDS=(42 43 44)
MODELS=(dynamite lstm)
SWEEPS=(friction push_magnitude action_delay)
NUM_EPISODES=50

LOGDIR="logs/sweeps"
mkdir -p "$LOGDIR"

TOTAL=$(( ${#SWEEPS[@]} * ${#MODELS[@]} * ${#SEEDS[@]} ))
COUNT=0

echo "============================================"
echo " OOD Robustness Sweeps: ${TOTAL} eval runs"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for sweep in "${SWEEPS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))

            # Find checkpoint
            CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" \
                -name "best.pt" -type f 2>/dev/null | head -1)

            if [[ -z "$CKPT" ]]; then
                echo "[$COUNT/$TOTAL] SKIP — no checkpoint for ${model}/seed_${seed}"
                continue
            fi

            # Output dir
            OUTDIR="results/sweeps/${sweep}/${model}_seed${seed}"
            mkdir -p "$OUTDIR"

            echo ""
            echo "[$COUNT/$TOTAL] sweep=${sweep} model=${model} seed=${seed} — $(date '+%H:%M:%S')"

            python3 scripts/eval.py \
                --checkpoint "$CKPT" \
                --task configs/task/randomized.yaml \
                --sweep "configs/sweeps/${sweep}.yaml" \
                --num_episodes "${NUM_EPISODES}" \
                --seed "${seed}" \
                --output_dir "$OUTDIR" \
                2>&1 | tee "${LOGDIR}/sweep_${sweep}_${model}_seed${seed}.log"

            echo "[$COUNT/$TOTAL] DONE — $(date '+%H:%M:%S')"
        done
    done
done

echo ""
echo "============================================"
echo " OOD Sweeps complete: ${TOTAL}/${TOTAL}"
echo " End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
