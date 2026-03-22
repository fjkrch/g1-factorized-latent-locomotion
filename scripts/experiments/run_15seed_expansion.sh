#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# 15-seed expansion: Train + Evaluate all new seeds
# ══════════════════════════════════════════════════════════════════════════════
#
# Phase 1: Train 35 new runs (DynaMITE 52-56, LSTM/Transformer/MLP 47-56)
# Phase 2: OOD evals for all new seeds (47-56 all models)
# Phase 3: Push recovery for all new seeds
#
# Resume-safe: checks for existing checkpoints/results before running
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

LOG_FILE="logs/15seed_expansion_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

exec > >(tee -a "$LOG_FILE") 2>&1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  15-SEED EXPANSION — $(date -Iseconds)                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Conda setup
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

COMPLETED=0
FAILED=0
SKIPPED=0

# ── Helper: find checkpoint for a model/seed ──
find_checkpoint() {
    local model=$1 seed=$2
    local dir="outputs/randomized/${model}_full/seed_${seed}/"
    if [[ -d "$dir" ]]; then
        find "$dir" -name "best.pt" 2>/dev/null | sort | tail -1 || true
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: TRAINING
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════ PHASE 1: TRAINING ═══════════════════════════"

# Define training jobs: model seed
TRAIN_JOBS=""

# DynaMITE: needs seeds 52-56 (already has 42-51)
for s in 52 53 54 55 56; do
    TRAIN_JOBS="$TRAIN_JOBS dynamite:$s"
done

# LSTM, Transformer, MLP: need seeds 47-56 (already have 42-46)
for model in lstm transformer mlp; do
    for s in 47 48 49 50 51 52 53 54 55 56; do
        TRAIN_JOBS="$TRAIN_JOBS ${model}:$s"
    done
done

TOTAL_TRAIN=$(echo $TRAIN_JOBS | wc -w)
TRAIN_IDX=0

for job in $TRAIN_JOBS; do
    model="${job%%:*}"
    seed="${job##*:}"
    TRAIN_IDX=$((TRAIN_IDX + 1))

    # Check if already trained
    ckpt=$(find_checkpoint "$model" "$seed")
    if [[ -n "$ckpt" ]]; then
        echo "[SKIP] ($TRAIN_IDX/$TOTAL_TRAIN) ${model} seed_${seed} — already trained"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  TRAIN ($TRAIN_IDX/$TOTAL_TRAIN): ${model} seed_${seed}"
    echo "  TIME: $(date -Iseconds)"
    echo "──────────────────────────────────────────────────────────────"

    if bash scripts/run_train.sh --task randomized --model "$model" --seed "$seed"; then
        COMPLETED=$((COMPLETED + 1))
        echo "[OK] ${model} seed_${seed} — training complete"
    else
        FAILED=$((FAILED + 1))
        echo "[FAIL] ${model} seed_${seed} — training failed (exit $?)"
    fi
done

echo ""
echo "PHASE 1 SUMMARY: ${COMPLETED} completed, ${FAILED} failed, ${SKIPPED} skipped"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1B: ID EVALUATIONS (for all seeds without eval_metrics.json)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════ PHASE 1B: ID EVALUATIONS ════════════════════════════"

IDEVAL_COMPLETED=0
IDEVAL_FAILED=0
IDEVAL_SKIPPED=0

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
            IDEVAL_SKIPPED=$((IDEVAL_SKIPPED + 1))
            continue
        fi

        echo "  ID EVAL: ${model} seed_${seed}"
        mkdir -p "$eval_dir"
        if bash scripts/run_eval.sh \
            --checkpoint "$ckpt" \
            --num-episodes 100 \
            --seed "$seed" \
            --output-dir "$eval_dir"; then
            IDEVAL_COMPLETED=$((IDEVAL_COMPLETED + 1))
            echo "  [OK] ${model} seed_${seed} ID eval"
        else
            IDEVAL_FAILED=$((IDEVAL_FAILED + 1))
            echo "  [FAIL] ${model} seed_${seed} ID eval"
        fi
    done
done

echo ""
echo "PHASE 1B SUMMARY: ${IDEVAL_COMPLETED} completed, ${IDEVAL_FAILED} failed, ${IDEVAL_SKIPPED} skipped"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: OOD SWEEP EVALUATIONS
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════ PHASE 2: OOD SWEEPS ═════════════════════════════"

EVAL_COMPLETED=0
EVAL_FAILED=0
EVAL_SKIPPED=0

# Run sweeps for new seeds (47-56 for all models)
SWEEP_TYPES="combined_shift friction push_magnitude action_delay action_delay_unseen"

for model in dynamite lstm transformer mlp; do
    for seed in 47 48 49 50 51 52 53 54 55 56; do
        ckpt=$(find_checkpoint "$model" "$seed")
        if [[ -z "$ckpt" ]]; then
            echo "[SKIP] ${model} seed_${seed} — no checkpoint found"
            EVAL_SKIPPED=$((EVAL_SKIPPED + 1))
            continue
        fi

        for sweep in $SWEEP_TYPES; do
            out_dir="results/ood_v2/${sweep}/randomized/${model}_seed${seed}"
            result_file="${out_dir}/sweep_${sweep}.json"

            if [[ -f "$result_file" ]]; then
                echo "[SKIP] ${model} seed_${seed} ${sweep} — already exists"
                EVAL_SKIPPED=$((EVAL_SKIPPED + 1))
                continue
            fi

            echo ""
            echo "  EVAL: ${model} seed_${seed} — ${sweep}"
            echo "  CKPT: $ckpt"
            echo "  TIME: $(date -Iseconds)"

            mkdir -p "$out_dir"
            if bash scripts/run_eval.sh \
                --checkpoint "$ckpt" \
                --sweep "$sweep" \
                --num-episodes 50 \
                --seed "$seed" \
                --output-dir "$out_dir"; then
                EVAL_COMPLETED=$((EVAL_COMPLETED + 1))
                echo "  [OK] ${model} seed_${seed} ${sweep}"
            else
                EVAL_FAILED=$((EVAL_FAILED + 1))
                echo "  [FAIL] ${model} seed_${seed} ${sweep}"
            fi
        done
    done
done

echo ""
echo "PHASE 2 SUMMARY: ${EVAL_COMPLETED} completed, ${EVAL_FAILED} failed, ${EVAL_SKIPPED} skipped"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: PUSH RECOVERY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════ PHASE 3: PUSH RECOVERY ══════════════════════════"

PUSH_COMPLETED=0
PUSH_FAILED=0
PUSH_SKIPPED=0

for model in dynamite lstm transformer mlp; do
    for seed in 47 48 49 50 51 52 53 54 55 56; do
        ckpt=$(find_checkpoint "$model" "$seed")
        if [[ -z "$ckpt" ]]; then
            echo "[SKIP] ${model} seed_${seed} push_recovery — no checkpoint"
            PUSH_SKIPPED=$((PUSH_SKIPPED + 1))
            continue
        fi

        out_dir="results/push_recovery/${model}_seed${seed}"
        result_file="${out_dir}/push_recovery_${model}_seed${seed}.json"

        if [[ -f "$result_file" ]]; then
            echo "[SKIP] ${model} seed_${seed} push_recovery — already exists"
            PUSH_SKIPPED=$((PUSH_SKIPPED + 1))
            continue
        fi

        echo ""
        echo "  PUSH RECOVERY: ${model} seed_${seed}"
        echo "  CKPT: $ckpt"
        echo "  TIME: $(date -Iseconds)"

        mkdir -p "$out_dir"
        if python3 scripts/push_recovery.py \
            --checkpoint "$ckpt" \
            --num_episodes 50 \
            --seed "$seed" \
            --output_dir "$out_dir"; then
            PUSH_COMPLETED=$((PUSH_COMPLETED + 1))
            echo "  [OK] ${model} seed_${seed} push_recovery"
        else
            PUSH_FAILED=$((PUSH_FAILED + 1))
            echo "  [FAIL] ${model} seed_${seed} push_recovery"
        fi
    done
done

echo ""
echo "PHASE 3 SUMMARY: ${PUSH_COMPLETED} completed, ${PUSH_FAILED} failed, ${PUSH_SKIPPED} skipped"

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL PHASES COMPLETE — $(date -Iseconds)                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Training:      ${COMPLETED} ok, ${FAILED} fail, ${SKIPPED} skip"
echo "║  OOD Sweeps:    ${EVAL_COMPLETED} ok, ${EVAL_FAILED} fail, ${EVAL_SKIPPED} skip"
echo "║  Push Recovery: ${PUSH_COMPLETED} ok, ${PUSH_FAILED} fail, ${PUSH_SKIPPED} skip"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Log: $LOG_FILE"
echo "Next: Run aggregation script to compute n=15 statistics"
