#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Batch 1 Phase 2: Confirm promoted checkpoints across 3 seeds × 3 sweeps
# ══════════════════════════════════════════════════════════════════════════════
# After screening, ckpt_5529600 and ckpt_7372800 passed decision thresholds.
# This script evaluates both checkpoints for seeds 42-44 on all 3 sweep types
# (friction, push_magnitude, action_delay) to match existing LSTM/DynaMITE
# baseline sweep data.
#
# Expected runtime: ~5 min per (checkpoint × seed × sweep) × 2 × 3 × 3 = ~90 min
# But Isaac Sim must restart per eval (os._exit), so actual ~2-3 min each.
# Total: ~36-54 min
#
# Outputs:
#   results/batch1_checkpoint_selection/{ckpt_name}/seed_{N}/sweep_{type}.json
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
NUM_EPISODES=50
SEEDS=(42 43 44)
SWEEPS=(friction push_magnitude action_delay)
CHECKPOINTS=(ckpt_5529600 ckpt_7372800)

echo "══════════════════════════════════════════════════════════════"
echo "  BATCH 1 PHASE 2: 3-SEED CONFIRMATION"
echo "  Checkpoints: ${CHECKPOINTS[*]}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Sweeps: ${SWEEPS[*]}"
echo "  Episodes per level: ${NUM_EPISODES}"
echo "  Started: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

TOTAL=$((${#CHECKPOINTS[@]} * ${#SEEDS[@]} * ${#SWEEPS[@]}))
COUNT=0

for CKPT in "${CHECKPOINTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Find run directory for this seed
        RUN_DIR=$(ls -d outputs/randomized/dynamite_full/seed_${SEED}/2* 2>/dev/null | head -1)
        if [[ -z "$RUN_DIR" ]]; then
            echo "[ERROR] No run directory found for seed ${SEED}"
            continue
        fi
        CKPT_PATH="${RUN_DIR}/checkpoints/${CKPT}.pt"
        if [[ ! -f "$CKPT_PATH" ]]; then
            echo "[ERROR] Checkpoint not found: ${CKPT_PATH}"
            continue
        fi

        for SWEEP in "${SWEEPS[@]}"; do
            COUNT=$((COUNT + 1))
            OUT_DIR="results/batch1_checkpoint_selection/${CKPT}/seed_${SEED}"
            mkdir -p "${OUT_DIR}"

            # Skip if already evaluated
            if [[ -f "${OUT_DIR}/sweep_${SWEEP}.json" ]]; then
                echo "[${COUNT}/${TOTAL}] SKIP ${CKPT} seed${SEED} ${SWEEP}"
                continue
            fi

            echo ""
            echo "──────────────────────────────────────────────────────"
            echo "  [${COUNT}/${TOTAL}] ${CKPT} | seed ${SEED} | ${SWEEP}"
            echo "  Checkpoint: ${CKPT_PATH}"
            echo "  Output: ${OUT_DIR}"
            echo "  Time: $(date -Iseconds)"
            echo "──────────────────────────────────────────────────────"

            $PYTHON scripts/eval.py \
                --checkpoint "${CKPT_PATH}" \
                --sweep "configs/sweeps/${SWEEP}.yaml" \
                --num_episodes ${NUM_EPISODES} \
                --seed ${SEED} \
                --output_dir "${OUT_DIR}"

            echo "[DONE] ${CKPT} seed${SEED} ${SWEEP}"
        done
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL 3-SEED CONFIRMATIONS COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# Quick comparison
$PYTHON << 'PYEOF'
import json, os, numpy as np

checkpoints = ["ckpt_5529600", "ckpt_7372800"]
seeds = [42, 43, 44]
sweeps = ["friction", "push_magnitude", "action_delay"]

# Also load baseline (best.pt) from existing multi-seed sweeps for comparison
def load_baseline_sweep(sweep_type, seed):
    f = f"results/sweeps_multiseed/{sweep_type}/dynamite_seed{seed}/sweep_{sweep_type}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        return [r["episode_reward/mean"] for r in d["results"]]
    return None

print("\n" + "=" * 80)
print("3-SEED CONFIRMATION RESULTS")
print("=" * 80)

for sweep in sweeps:
    print(f"\n--- {sweep} ---")
    
    # Baseline (best.pt)
    baseline_per_seed = []
    for s in seeds:
        rewards = load_baseline_sweep(sweep, s)
        if rewards:
            baseline_per_seed.append(rewards)
    
    if baseline_per_seed:
        baseline_mean = np.mean(baseline_per_seed, axis=0)
        baseline_worst = min(np.mean(baseline_per_seed, axis=0))
        baseline_nominal = baseline_mean[0]
        baseline_sens = baseline_nominal - baseline_worst
        print(f"  best.pt:       nominal={baseline_nominal:.4f}  worst={baseline_worst:.4f}  sens={baseline_sens:.4f}")
    
    # Promoted checkpoints
    for ckpt in checkpoints:
        per_seed = []
        for s in seeds:
            f = f"results/batch1_checkpoint_selection/{ckpt}/seed_{s}/sweep_{sweep}.json"
            if os.path.exists(f):
                d = json.load(open(f))
                per_seed.append([r["episode_reward/mean"] for r in d["results"]])
        
        if per_seed:
            mean_rewards = np.mean(per_seed, axis=0)
            worst = min(mean_rewards)
            nominal = mean_rewards[0]
            sens = nominal - worst
            print(f"  {ckpt}: nominal={nominal:.4f}  worst={worst:.4f}  sens={sens:.4f}")
        else:
            print(f"  {ckpt}: NO DATA")

PYEOF
