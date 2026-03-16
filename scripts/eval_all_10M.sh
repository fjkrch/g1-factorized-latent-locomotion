#!/usr/bin/env bash
# Evaluate all 16 main comparison runs after 10M resume training
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
echo "  EVAL ALL 16 MAIN RUNS (post-10M resume)"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        RUN_DIR=$(ls -d "outputs/${task}/${model}_full/seed_42/"*/ 2>/dev/null | head -1)
        if [[ -z "$RUN_DIR" ]]; then
            echo "[$CURRENT/$TOTAL] SKIP (no dir): ${task}/${model}"
            continue
        fi

        echo "[$CURRENT/$TOTAL] EVAL: ${task}/${model} — ${RUN_DIR}"
        python scripts/eval.py --run_dir "$RUN_DIR" --num_episodes 100 --seed 42 || {
            echo "  ⚠ FAILED: ${task}/${model}"
            FAILED=$((FAILED + 1))
            continue
        }
        echo "[$CURRENT/$TOTAL] DONE: ${task}/${model}"
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  EVAL COMPLETE: $(date -Iseconds)"
echo "  Failed: $FAILED / $TOTAL"
echo "══════════════════════════════════════════════════════════════"
