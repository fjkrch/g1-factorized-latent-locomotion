#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Batch 2: Randomization Strengthening Screen
# ══════════════════════════════════════════════════════════════════════════════
# Train 3 DynaMITE variants with wider DR ranges, then evaluate on push sweep.
#
# Variants:
#   V1: wider_push — push_vel_range [0.5, 5.0]
#   V2: wider_all  — push + friction + delay ranges widened
#   V3: aggressive_push — push [0.5, 8.0] + more frequent pushes
#
# Screen: 1 seed (42), 10M timesteps each (~14 min), deterministic push eval
# Expected: ~50 min total (14 min train + 3 min eval × 3 variants)
#
# Outputs:
#   outputs/randomized/{variant}/seed_42/{timestamp}/  (training)
#   results/batch2_dr_strengthening/{variant}/sweep_push_magnitude.json (eval)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
SEED=42
TOTAL_TIMESTEPS=10000000
NUM_EVAL_EPISODES=50

# Variants to train and evaluate
declare -A VARIANTS=(
    ["wider_push"]="configs/task/randomized_wider_push.yaml"
    ["wider_all"]="configs/task/randomized_wider_all.yaml"
    ["aggressive_push"]="configs/task/randomized_aggressive_push.yaml"
)

echo "══════════════════════════════════════════════════════════════"
echo "  BATCH 2: RANDOMIZATION STRENGTHENING SCREEN"
echo "  Seed: ${SEED} | Timesteps: ${TOTAL_TIMESTEPS}"
echo "  Variants: ${!VARIANTS[@]}"
echo "  Started: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

for VARIANT in wider_push wider_all aggressive_push; do
    TASK_CFG="${VARIANTS[$VARIANT]}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  TRAINING: ${VARIANT}"
    echo "  Task config: ${TASK_CFG}"
    echo "  Time: $(date -Iseconds)"
    echo "════════════════════════════════════════════════════════════"

    # Check if already trained (a run dir exists with metrics.csv for this variant)
    EXISTING=$(ls -d outputs/randomized/dynamite_${VARIANT}/seed_${SEED}/2*/metrics.csv 2>/dev/null | head -1)
    if [[ -n "$EXISTING" ]]; then
        RUN_DIR=$(dirname "$EXISTING")
        echo "[SKIP] Already trained: ${RUN_DIR}"
    else
        # Train
        $PYTHON scripts/train.py \
            --task "${TASK_CFG}" \
            --model configs/model/dynamite.yaml \
            --variant "${VARIANT}" \
            --seed ${SEED} \
            --set "train.total_timesteps=${TOTAL_TIMESTEPS}"

        # Find the run directory just created
        RUN_DIR=$(ls -dt outputs/randomized/dynamite_${VARIANT}/seed_${SEED}/2* 2>/dev/null | head -1)
        if [[ -z "$RUN_DIR" ]]; then
            echo "[ERROR] No run directory found after training ${VARIANT}"
            continue
        fi
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  EVALUATING: ${VARIANT} on push_magnitude sweep"
    echo "  Run dir: ${RUN_DIR}"
    echo "  Time: $(date -Iseconds)"
    echo "────────────────────────────────────────────────────────────"

    EVAL_OUT="results/batch2_dr_strengthening/${VARIANT}"
    mkdir -p "${EVAL_OUT}"

    # Skip if already evaluated
    if [[ -f "${EVAL_OUT}/sweep_push_magnitude.json" ]]; then
        echo "[SKIP] Already evaluated: ${EVAL_OUT}/sweep_push_magnitude.json"
    else
        CKPT="${RUN_DIR}/checkpoints/best.pt"
        if [[ ! -f "$CKPT" ]]; then
            CKPT="${RUN_DIR}/checkpoints/latest.pt"
        fi
        if [[ ! -f "$CKPT" ]]; then
            echo "[ERROR] No checkpoint found in ${RUN_DIR}/checkpoints/"
            continue
        fi

        $PYTHON scripts/eval.py \
            --checkpoint "${CKPT}" \
            --sweep configs/sweeps/push_magnitude.yaml \
            --num_episodes ${NUM_EVAL_EPISODES} \
            --seed ${SEED} \
            --output_dir "${EVAL_OUT}"

        echo "[DONE] ${VARIANT} evaluated"
    fi
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL BATCH 2 SCREEN EXPERIMENTS COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# Quick comparison
$PYTHON << 'PYEOF'
import json, os, numpy as np

print("\n" + "=" * 80)
print("BATCH 2 SCREEN RESULTS: Push Magnitude Sweep")
print("=" * 80)

# Baseline (best.pt from existing sweeps)
baseline_f = "results/sweeps_multiseed/push_magnitude/dynamite_seed42/sweep_push_magnitude.json"
if os.path.exists(baseline_f):
    d = json.load(open(baseline_f))
    rewards = [r["episode_reward/mean"] for r in d["results"]]
    nominal = rewards[0]
    worst = min(rewards)
    sens = nominal - worst
    print(f"\n  {'baseline (best.pt)':<25} nominal={nominal:.4f}  worst={worst:.4f}  sens={sens:.4f}  OOD_avg={np.mean(rewards):.4f}")

for variant in ["wider_push", "wider_all", "aggressive_push"]:
    f = f"results/batch2_dr_strengthening/{variant}/sweep_push_magnitude.json"
    if os.path.exists(f):
        d = json.load(open(f))
        rewards = [r["episode_reward/mean"] for r in d["results"]]
        nominal = rewards[0]
        worst = min(rewards)
        sens = nominal - worst
        print(f"  {variant:<25} nominal={nominal:.4f}  worst={worst:.4f}  sens={sens:.4f}  OOD_avg={np.mean(rewards):.4f}")
    else:
        print(f"  {variant:<25} NO DATA")

print("\nDecision: Promote if worst-case improves ≥0.15 OR sensitivity reduces ≥15% with nominal loss ≤0.10")
PYEOF
