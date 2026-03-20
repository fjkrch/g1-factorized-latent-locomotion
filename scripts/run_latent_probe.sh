#!/bin/bash
# Run latent probe experiment: DynaMITE vs LSTM, 3 seeds
# Each model runs in a separate process (SimulationApp constraint)

set -e

PYTHON_CMD="${PYTHON_CMD:-/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3}"
SCRIPT="scripts/latent_probe.py"
OUTPUT_DIR="results/latent_probe"
NUM_EPISODES="${1:-200}"
SEEDS=(42 43 44)

# Find checkpoints for randomized task
find_ckpt() {
    local model=$1
    local seed=$2
    local ckpt=$(find outputs/randomized/${model}_full/seed_${seed}/ -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
    echo "$ckpt"
}

echo "==============================================="
echo "Latent Probe Experiment"
echo "Models: DynaMITE, LSTM"
echo "Seeds: ${SEEDS[*]}"
echo "Episodes: ${NUM_EPISODES}"
echo "==============================================="

TOTAL=0
SUCCESS=0
FAILED=0

for SEED in "${SEEDS[@]}"; do
    for MODEL in dynamite lstm; do
        CKPT=$(find_ckpt "$MODEL" "$SEED")
        if [ -z "$CKPT" ]; then
            echo "[SKIP] No checkpoint for ${MODEL} seed ${SEED}"
            FAILED=$((FAILED + 1))
            continue
        fi

        TOTAL=$((TOTAL + 1))
        echo ""
        echo "--- [${TOTAL}] ${MODEL} seed=${SEED} ---"
        echo "    Checkpoint: ${CKPT}"

        if $PYTHON_CMD $SCRIPT \
            --model_type "$MODEL" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --num_episodes "$NUM_EPISODES" \
            --output_dir "$OUTPUT_DIR" \
            --headless 2>&1; then
            echo "    [OK] ${MODEL} seed ${SEED} completed"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "    [FAIL] ${MODEL} seed ${SEED}"
            FAILED=$((FAILED + 1))
        fi
    done
done

echo ""
echo "==============================================="
echo "Probe collection done: ${SUCCESS} succeeded, ${FAILED} failed out of ${TOTAL}"
echo "==============================================="

# Aggregate results
echo ""
echo "Aggregating results..."
$PYTHON_CMD $SCRIPT --aggregate --output_dir "$OUTPUT_DIR"

echo ""
echo "Done! Results in ${OUTPUT_DIR}/"
