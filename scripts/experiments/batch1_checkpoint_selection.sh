#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Batch 1: Checkpoint Selection Screening
# ══════════════════════════════════════════════════════════════════════════════
# Hypothesis: The best-nominal checkpoint (best.pt, ~6.6M steps) may not be
#   optimal for OOD robustness. Earlier checkpoints might sacrifice nominal
#   performance for better worst-case and sensitivity.
#
# Design:
#   Phase 1 (screen): Evaluate 5 checkpoints from seed 42 on push_magnitude
#     sweep (DynaMITE's only meaningful OOD vulnerability).
#   Phase 2 (confirm): If a checkpoint improves worst-case by ≥0.15 or
#     sensitivity by ≥15%, promote to 3-seed evaluation across all sweeps.
#
# Checkpoints selected:
#   ckpt_4300800  (35% of training, early convergence)
#   ckpt_5529600  (45%, mid-training, reward stabilizing)
#   best.pt       (~54%, nominal peak)
#   ckpt_7372800  (60%, post-peak decline)
#   ckpt_9216000  (75%, late training)
#
# Expected runtime: ~5 min per checkpoint × 5 = ~25 min total
# Resources: 1× RTX 4060 Laptop (8GB VRAM)
#
# Outputs:
#   results/batch1_checkpoint_selection/seed_42/{ckpt_name}/sweep_push_magnitude.json
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

# ── Config ──
SEED=42
TASK="randomized"
MODEL="dynamite"
SWEEP="push_magnitude"
NUM_EPISODES=50   # 50 episodes per sweep level (matches existing OOD protocol)

# Run directory for seed 42
RUN_DIR=$(ls -d outputs/${TASK}/${MODEL}_full/seed_${SEED}/2* | head -1)
CKPT_DIR="${RUN_DIR}/checkpoints"
OUTPUT_BASE="results/batch1_checkpoint_selection/seed_${SEED}"

# Checkpoints to evaluate
CHECKPOINTS=(
    "ckpt_4300800.pt"
    "ckpt_5529600.pt"
    "best.pt"
    "ckpt_7372800.pt"
    "ckpt_9216000.pt"
)

echo "══════════════════════════════════════════════════════════════"
echo "  BATCH 1: CHECKPOINT SELECTION SCREENING"
echo "  Seed: ${SEED} | Task: ${TASK} | Model: ${MODEL}"
echo "  Sweep: ${SWEEP} | Episodes per level: ${NUM_EPISODES}"
echo "  Run dir: ${RUN_DIR}"
echo "  Checkpoints: ${#CHECKPOINTS[@]}"
echo "  Started: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# Verify all checkpoints exist
for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "${CKPT_DIR}/${CKPT}" ]]; then
        echo "ERROR: Checkpoint not found: ${CKPT_DIR}/${CKPT}"
        exit 1
    fi
done
echo "All checkpoints verified."

# ── Run evaluations ──
RESULTS=()
for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="${CKPT%.pt}"
    OUT_DIR="${OUTPUT_BASE}/${CKPT_NAME}"
    mkdir -p "${OUT_DIR}"

    # Skip if already evaluated
    if [[ -f "${OUT_DIR}/sweep_${SWEEP}.json" ]]; then
        echo "[SKIP] ${CKPT_NAME} — already evaluated"
        RESULTS+=("${OUT_DIR}/sweep_${SWEEP}.json")
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Evaluating: ${CKPT_NAME}"
    echo "  Output: ${OUT_DIR}"
    echo "  Time: $(date -Iseconds)"
    echo "────────────────────────────────────────────────────────────"

    python3 scripts/eval.py \
        --checkpoint "${CKPT_DIR}/${CKPT}" \
        --sweep "configs/sweeps/${SWEEP}.yaml" \
        --num_episodes ${NUM_EPISODES} \
        --seed ${SEED} \
        --output_dir "${OUT_DIR}"

    RESULTS+=("${OUT_DIR}/sweep_${SWEEP}.json")
    echo "[DONE] ${CKPT_NAME}"
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL EVALUATIONS COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Quick summary ──
echo ""
echo "Results summary:"
python3 << 'PYEOF'
import json, os, sys
import numpy as np

output_base = "results/batch1_checkpoint_selection/seed_42"
checkpoints = ["ckpt_4300800", "ckpt_5529600", "best", "ckpt_7372800", "ckpt_9216000"]

print(f"{'Checkpoint':<18} {'Nominal':>10} {'Push 2-3':>10} {'Push 3-5':>10} {'Push 5-8':>10} {'Worst':>10} {'Sensitivity':>12}")
print("─" * 82)

for ckpt in checkpoints:
    f = os.path.join(output_base, ckpt, "sweep_push_magnitude.json")
    if not os.path.exists(f):
        print(f"{ckpt:<18} {'MISSING':>10}")
        continue
    d = json.load(open(f))
    results = d["results"]
    # Push levels: [0,0], [0.5,1], [1,2], [2,3], [3,5], [5,8]
    rewards = [r["episode_reward/mean"] for r in results]
    nominal = rewards[0]
    worst = min(rewards)
    sensitivity = nominal - worst
    print(f"{ckpt:<18} {nominal:>10.4f} {rewards[3]:>10.4f} {rewards[4]:>10.4f} {rewards[5]:>10.4f} {worst:>10.4f} {sensitivity:>12.4f}")

# Decision criteria
print("\nDecision thresholds:")
print("  Promote if: worst_case improves ≥0.15, OR sensitivity reduces ≥15%")
print("  Reject if: nominal degrades >0.10 with no robustness gain")
PYEOF
