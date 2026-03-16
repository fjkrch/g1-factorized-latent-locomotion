#!/usr/bin/env bash
# Resume all 16 main comparison runs for 10M additional steps
# Each run resumes from its latest.pt checkpoint
set -euo pipefail

cd "$(dirname "$0")/.."

eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
TOTAL=16
CURRENT=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  RESUME ALL: 4 tasks × 4 models, +10M steps each"
echo "  total_timesteps = 10,000,000 (from configs/train/default.yaml)"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        # Find the run directory
        RUN_DIR=$(ls -d "outputs/${task}/${model}_full/seed_42/"*/ 2>/dev/null | head -1)
        if [[ -z "$RUN_DIR" ]]; then
            echo "[$CURRENT/$TOTAL] SKIP (no run dir): ${task}/${model}"
            continue
        fi
        # Verify checkpoint exists
        if [[ ! -f "${RUN_DIR}checkpoints/latest.pt" ]]; then
            echo "[$CURRENT/$TOTAL] SKIP (no checkpoint): ${task}/${model}"
            continue
        fi

        echo ""
        echo "[$CURRENT/$TOTAL] RESUMING: ${task}/${model} from ${RUN_DIR}"

        python scripts/train.py \
            --task "configs/task/${task}.yaml" \
            --model "configs/model/${model}.yaml" \
            --resume "${RUN_DIR}" \
            --seed 42 || {
                echo "  ⚠ FAILED: ${task}/${model}"
                FAILED=$((FAILED + 1))
                continue
            }

        echo "[$CURRENT/$TOTAL] DONE: ${task}/${model}"
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  RESUME COMPLETE: $(date -Iseconds)"
echo "  Failed: $FAILED / $TOTAL"
echo "══════════════════════════════════════════════════════════════"
