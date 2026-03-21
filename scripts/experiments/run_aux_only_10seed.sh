#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# run_aux_only_10seed.sh — Train "Aux Only (no bottleneck)" ablation × 10 seeds
# ══════════════════════════════════════════════════════════════════════════════
#
# 2×2 factorial design to decompose bottleneck vs. aux loss effects:
#   | Cell              | Bottleneck | Aux Loss | Variant        | Status |
#   |-------------------|------------|----------|----------------|--------|
#   | DynaMITE Full     | YES        | YES      | full           | DONE   |
#   | No Aux Loss       | YES        | NO       | no_aux_loss    | DONE   |
#   | No Latent         | NO         | NO       | no_latent      | DONE   |
#   | Aux Only (NEW)    | NO*        | YES      | aux_only       | THIS   |
#
#   *Latent head exists for aux gradient flow, but condition_on_latent=false
#    so the policy pathway does not go through the 24-d tanh bottleneck.
#
# Trains: 10 seeds (42-51) × 1 variant = 10 training runs
# Evals:  10 runs × 100 episodes each = 10 evals
#
# Estimated time: ~2.5 hours training + ~30 min eval
#
# Usage:
#   bash scripts/experiments/run_aux_only_10seed.sh
#   bash scripts/experiments/run_aux_only_10seed.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd /home/chyanin/robotpaper

PYTHON=/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SEEDS=(42 43 44 45 46 47 48 49 50 51)
TASK="randomized"
MODEL="dynamite"
VARIANT="aux_only"
ABLATION="aux_only"
RESULT_DIR="results/ablation_10seed/aux_only"

TOTAL=${#SEEDS[@]}
CURRENT=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  AUX-ONLY (NO BOTTLENECK) ABLATION — 2×2 FACTORIAL"
echo "  ${TOTAL} seeds × 1 variant = ${TOTAL} training runs"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Phase 1: Training ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 1: TRAINING ($TOTAL runs)    ║"
echo "╚══════════════════════════════════════╝"

for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LABEL="${TASK}/${MODEL}_${VARIANT}/seed_${seed}"

    echo ""
    echo "[$CURRENT/$TOTAL] TRAINING: $LABEL"

    # Check if already done (handle both naming patterns)
    EXISTING=$(find "outputs/${TASK}/${MODEL}_${VARIANT}" "outputs/${TASK}/${MODEL}_${VARIANT}_${VARIANT}" -name "best.pt" -path "*/seed_${seed}/*/checkpoints/*" 2>/dev/null | head -1 || true)
    if [[ -n "$EXISTING" ]]; then
        echo "  [SKIP] Already trained: $EXISTING"
        continue
    fi

    CMD="bash scripts/run_train.sh --task $TASK --model $MODEL --seed $seed --variant $VARIANT --ablation $ABLATION"

    if $DRY_RUN; then
        echo "  [DRY RUN] $CMD"
        continue
    fi

    echo "  [START] $(date -Iseconds)"
    if $CMD; then
        echo "  [DONE] $(date -Iseconds)"
    else
        echo "  [FAIL] seed=$seed"
        FAILED=$((FAILED + 1))
    fi
done

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would train $TOTAL runs, then eval $TOTAL runs"
    exit 0
fi

echo ""
echo "Training complete. Failed: $FAILED/$TOTAL"

# ── Phase 2: Evaluation ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 2: EVALUATION ($TOTAL runs)  ║"
echo "╚══════════════════════════════════════╝"

mkdir -p "$RESULT_DIR"

CURRENT=0
for seed in "${SEEDS[@]}"; do
    CURRENT=$((CURRENT + 1))

    CKPT=$(find "outputs/${TASK}/${MODEL}_${VARIANT}" "outputs/${TASK}/${MODEL}_${VARIANT}_${VARIANT}" -name "best.pt" -path "*/seed_${seed}/*/checkpoints/*" 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$CKPT" ]]; then
        echo "[$CURRENT/$TOTAL] [SKIP] No checkpoint for seed $seed"
        continue
    fi

    OUT_FILE="${RESULT_DIR}/seed_${seed}/eval_metrics.json"
    if [[ -f "$OUT_FILE" ]]; then
        echo "[$CURRENT/$TOTAL] [SKIP] Already evaluated: $OUT_FILE"
        continue
    fi

    echo "[$CURRENT/$TOTAL] EVAL: seed=$seed"
    mkdir -p "${RESULT_DIR}/seed_${seed}"

    $PYTHON scripts/eval.py \
        --task configs/task/${TASK}.yaml \
        --model configs/model/${MODEL}.yaml \
        --ablation configs/ablations/${ABLATION}.yaml \
        --ckpt "$CKPT" \
        --seed 42 \
        --num_episodes 100 \
        --output_dir "${RESULT_DIR}/seed_${seed}" \
        --headless

    echo "  [DONE] seed=$seed"
done

# ── Phase 3: Aggregate ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 3: AGGREGATE                 ║"
echo "╚══════════════════════════════════════╝"

$PYTHON << 'PYEOF'
import json, os, glob
import numpy as np

result_dir = "results/ablation_10seed/aux_only"
files = sorted(glob.glob(f"{result_dir}/seed_*/eval_metrics.json"))
rewards = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    rewards.append(d["episode_reward/mean"])
    print(f"  {f}: {d['episode_reward/mean']:.4f}")

if rewards:
    arr = np.array(rewards)
    print(f"\n  Aux Only: {arr.mean():.4f} ± {arr.std():.4f} (n={len(rewards)})")
else:
    print("  No results found!")
PYEOF

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  AUX-ONLY ABLATION COMPLETE"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
