#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Run ID evaluations for ALL seeds that are missing eval_metrics.json
# Run this after training completes
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

cd "$(dirname "$(dirname "$0")")"

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

COMPLETED=0
FAILED=0
SKIPPED=0

find_checkpoint() {
    local model=$1 seed=$2
    local dir="outputs/randomized/${model}_full/seed_${seed}/"
    if [[ -d "$dir" ]]; then
        find "$dir" -name "best.pt" 2>/dev/null | sort | tail -1 || true
    fi
}

echo "═══════════════════ ID EVALUATIONS ═══════════════════════════════"

for model in dynamite lstm transformer mlp; do
    for seed in 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56; do
        ckpt=$(find_checkpoint "$model" "$seed")
        if [[ -z "$ckpt" ]]; then
            continue
        fi
        run_dir=$(dirname "$(dirname "$ckpt")")
        eval_dir="${run_dir}/eval_recomputed"
        eval_file="${eval_dir}/eval_metrics.json"

        if [[ -f "$eval_file" ]]; then
            SKIPPED=$((SKIPPED + 1))
            echo "[SKIP] ${model} seed_${seed} — already has eval"
            continue
        fi

        echo ""
        echo "  ID EVAL: ${model} seed_${seed}"
        echo "  CKPT: $ckpt"
        echo "  TIME: $(date -Iseconds)"

        mkdir -p "$eval_dir"
        if bash scripts/run_eval.sh \
            --checkpoint "$ckpt" \
            --num-episodes 100 \
            --seed "$seed" \
            --output-dir "$eval_dir"; then
            COMPLETED=$((COMPLETED + 1))
            echo "  [OK] ${model} seed_${seed}"
        else
            FAILED=$((FAILED + 1))
            echo "  [FAIL] ${model} seed_${seed}"
        fi
    done
done

echo ""
echo "ID EVAL SUMMARY: ${COMPLETED} completed, ${FAILED} failed, ${SKIPPED} skipped"
