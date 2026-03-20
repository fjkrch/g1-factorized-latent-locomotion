#!/usr/bin/env bash
# Run push-recovery protocol for all 4 models × 5 seeds.
# Each run launches a fresh Python process (Isaac Lab requires one SimulationApp per process).
#
# Usage: bash scripts/run_push_recovery.sh [NUM_EPISODES]

set -euo pipefail

NUM_EPISODES=${1:-50}
PYTHON_CMD="${PYTHON_CMD:-/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3}"
OUTPUT_BASE="results/push_recovery"
LOGDIR="logs"
mkdir -p "$LOGDIR" "$OUTPUT_BASE"

MODELS=(dynamite lstm transformer mlp)
SEEDS=(42 43 44 45 46)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/push_recovery_${TIMESTAMP}.log"

TOTAL=$((${#MODELS[@]} * ${#SEEDS[@]}))
DONE=0
FAILED=0

echo "Push Recovery Campaign: ${#MODELS[@]} models × ${#SEEDS[@]} seeds = $TOTAL runs"
echo "Episodes per push magnitude: $NUM_EPISODES"
echo "Log: $LOGFILE"

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Find checkpoint
        CKPT=$(find outputs/randomized/${model}_full/seed_${seed}/ -name "best.pt" 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            echo "  [SKIP] No checkpoint for ${model}/seed_${seed}" | tee -a "$LOGFILE"
            FAILED=$((FAILED + 1))
            continue
        fi

        OUT_DIR="${OUTPUT_BASE}/${model}_seed${seed}"
        JSON_FILE="${OUT_DIR}/push_recovery_${model}_seed${seed}.json"

        if [ -f "$JSON_FILE" ]; then
            echo "  [SKIP] Already exists: $JSON_FILE" | tee -a "$LOGFILE"
            DONE=$((DONE + 1))
            continue
        fi

        echo "  [RUN] ${model} seed=${seed} ..." | tee -a "$LOGFILE"
        mkdir -p "$OUT_DIR"
        
        $PYTHON_CMD scripts/push_recovery.py \
            --checkpoint "$CKPT" \
            --num_episodes "$NUM_EPISODES" \
            --seed "$seed" \
            --output_dir "$OUT_DIR" \
            --headless \
            > "${OUT_DIR}/run.log" 2>&1
        
        if [ -f "$JSON_FILE" ]; then
            echo "    [OK] $(grep -o 'non_fall_rate=[0-9.]*' "${OUT_DIR}/run.log" | tail -1)" | tee -a "$LOGFILE"
            DONE=$((DONE + 1))
        else
            echo "    [FAIL] No JSON output. Check ${OUT_DIR}/run.log" | tee -a "$LOGFILE"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "Push Recovery Campaign Complete: $DONE succeeded, $FAILED failed out of $TOTAL"
echo "Results in: $OUTPUT_BASE/"
