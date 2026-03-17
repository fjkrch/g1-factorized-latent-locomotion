#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all_ablations_12M.sh — Resume all 7 ablations to 12M timesteps,
#                            evaluate, aggregate, and regenerate plots/tables
# ══════════════════════════════════════════════════════════════════════════════
#
# Previous ablation runs trained for ~2M steps (160 iterations).
# This script resumes each from its latest checkpoint and continues
# training to 12,000,000 total timesteps, matching the main model budget.
#
# GPU-optimised for RTX 4060 Laptop (8 GB VRAM):
#   - num_envs reduced to 256 (from 512) to avoid OOM
#   - CUDA memory allocator tuned to reduce fragmentation
#   - GPU cache cleared between runs
#   - save_interval raised to reduce checkpoint I/O
#
# Usage:
#   bash scripts/run_all_ablations_12M.sh              # full pipeline
#   bash scripts/run_all_ablations_12M.sh --dry-run    # preview commands
#   bash scripts/run_all_ablations_12M.sh --eval-only  # skip training, just eval+aggregate
#
# Expected runtime: ~2 hours on RTX 4060 (7 runs × ~17 min each at 256 envs)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── GPU optimisation env vars ──
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

# ── Parse args ──
DRY_RUN=false
EVAL_ONLY=false
NUM_ENVS=256          # safe for 8 GB VRAM; override with --num-envs N
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true; shift ;;
        --eval-only)  EVAL_ONLY=true; shift ;;
        --num-envs)   NUM_ENVS="$2"; shift 2 ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

# ── Configuration ──
TASK="randomized"
SEED=42
TOTAL_TIMESTEPS=12000000

# Ablation name → output directory name
# Directory pattern: outputs/randomized/{DIR_NAME}/seed_42/{timestamp}/
declare -A ABL_DIRS=(
    [seq_len_4]="dynamite_seq_len_4"
    [seq_len_16]="dynamite_seq_len_16"
    [no_latent]="dynamite_no_latent_no_latent"
    [single_latent]="dynamite_single_latent_single_latent"
    [no_aux_loss]="dynamite_no_aux_no_aux_loss"
    [depth_1]="dynamite_depth1_depth_1"
    [depth_4]="dynamite_depth4_depth_4"
)

ABLATIONS=(seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4)
TOTAL=${#ABLATIONS[@]}

echo "══════════════════════════════════════════════════════════════"
echo "  ABLATION 12M PIPELINE  (GPU-optimised)"
echo "  ${TOTAL} ablations × 1 seed, target: ${TOTAL_TIMESTEPS} total steps"
echo "  Task: ${TASK}  |  Seed: ${SEED}  |  num_envs: ${NUM_ENVS}"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Helper: clear GPU memory between runs ──
gpu_cleanup() {
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('[GPU] Cache cleared')" 2>/dev/null || true
    sleep 2
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Resume training to 12M steps
# ══════════════════════════════════════════════════════════════════════════════
CURRENT=0
FAILED=0
SKIPPED=0

if ! $EVAL_ONLY; then
    echo ""
    echo "━━━ PHASE 1: Training (resume to ${TOTAL_TIMESTEPS} steps, ${NUM_ENVS} envs) ━━━"
    echo ""

    for abl in "${ABLATIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        dir_name="${ABL_DIRS[$abl]}"

        # Find existing run directory
        RUN_DIR=$(ls -d "outputs/${TASK}/${dir_name}/seed_${SEED}/"*/ 2>/dev/null | head -1)

        if [[ -z "$RUN_DIR" ]]; then
            echo "[$CURRENT/$TOTAL] ⚠ NO EXISTING RUN: ${abl} — training from scratch"

            CMD="python scripts/train.py \
                --task configs/task/${TASK}.yaml \
                --model configs/model/dynamite.yaml \
                --ablation configs/ablations/${abl}.yaml \
                --variant ${abl} \
                --seed ${SEED} \
                --set train.total_timesteps=${TOTAL_TIMESTEPS} task.num_envs=${NUM_ENVS} train.save_interval=100"

            if $DRY_RUN; then
                echo "  [DRY RUN] $CMD"
                continue
            fi

            echo "[$CURRENT/$TOTAL] TRAINING: ${abl} (from scratch)"
            gpu_cleanup
            eval "$CMD" || {
                echo "  ⚠ FAILED: ${abl}"
                FAILED=$((FAILED + 1))
                continue
            }
            echo "[$CURRENT/$TOTAL] DONE: ${abl}"
            gpu_cleanup
            continue
        fi

        # Verify checkpoint exists
        if [[ ! -f "${RUN_DIR}checkpoints/latest.pt" ]]; then
            echo "[$CURRENT/$TOTAL] ⚠ NO CHECKPOINT: ${abl} at ${RUN_DIR}"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Check if already at target
        METRICS="${RUN_DIR}metrics.csv"
        if [[ -f "$METRICS" ]]; then
            LAST_STEP=$(tail -1 "$METRICS" | cut -d, -f2)
            if [[ "$LAST_STEP" -ge 11000000 ]] 2>/dev/null; then
                echo "[$CURRENT/$TOTAL] SKIP (already at step ${LAST_STEP}): ${abl}"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            echo "[$CURRENT/$TOTAL] RESUMING: ${abl} from step ${LAST_STEP} → ${TOTAL_TIMESTEPS}"
        fi

        CMD="python scripts/train.py \
            --task configs/task/${TASK}.yaml \
            --model configs/model/dynamite.yaml \
            --ablation configs/ablations/${abl}.yaml \
            --resume ${RUN_DIR} \
            --seed ${SEED} \
            --set train.total_timesteps=${TOTAL_TIMESTEPS} task.num_envs=${NUM_ENVS} train.save_interval=100"

        if $DRY_RUN; then
            echo "  [DRY RUN] $CMD"
            continue
        fi

        gpu_cleanup
        eval "$CMD" || {
            echo "  ⚠ FAILED: ${abl}"
            FAILED=$((FAILED + 1))
            continue
        }
        echo "[$CURRENT/$TOTAL] DONE: ${abl}"
        gpu_cleanup
    done

    echo ""
    echo "━━━ PHASE 1 COMPLETE ━━━"
    echo "  Trained: $((CURRENT - FAILED - SKIPPED))  |  Skipped: $SKIPPED  |  Failed: $FAILED"

    if [[ $FAILED -gt 0 ]]; then
        echo "WARNING: $FAILED training runs failed."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Evaluate all ablation checkpoints
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ PHASE 2: Evaluation ━━━"
echo ""

CURRENT=0
EVAL_FAILED=0

for abl in "${ABLATIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    dir_name="${ABL_DIRS[$abl]}"
    RUN_DIR=$(ls -d "outputs/${TASK}/${dir_name}/seed_${SEED}/"*/ 2>/dev/null | head -1)

    if [[ -z "$RUN_DIR" ]]; then
        echo "[$CURRENT/$TOTAL] SKIP (no dir): ${abl}"
        continue
    fi

    echo "[$CURRENT/$TOTAL] EVAL: ${abl} — ${RUN_DIR}"

    if $DRY_RUN; then
        echo "  [DRY RUN] python scripts/eval.py --run_dir $RUN_DIR --num_episodes 100 --seed $SEED"
        continue
    fi

    python scripts/eval.py --run_dir "$RUN_DIR" --num_episodes 100 --seed "$SEED" || {
        echo "  ⚠ EVAL FAILED: ${abl}"
        EVAL_FAILED=$((EVAL_FAILED + 1))
        continue
    }
    echo "[$CURRENT/$TOTAL] EVAL DONE: ${abl}"
done

echo ""
echo "━━━ PHASE 2 COMPLETE ━━━"
echo "  Evaluated: $((CURRENT - EVAL_FAILED))  |  Failed: $EVAL_FAILED"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Aggregate results + regenerate plots/tables
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ PHASE 3: Aggregation & Plotting ━━━"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] bash scripts/aggregate_all.sh --skip-eval"
    echo "[DRY RUN] python scripts/plot_results.py --results results/aggregated/ --output figures/"
    echo "[DRY RUN] python scripts/generate_tables.py --results_dir results/aggregated/ --output_dir results/tables/"
else
    bash scripts/aggregate_all.sh --skip-eval || echo "⚠ Aggregation had errors"

    echo ""
    echo "  Regenerating plots..."
    python scripts/plot_results.py --results results/aggregated/ --output figures/ || echo "⚠ Plot generation had errors"

    echo ""
    echo "  Regenerating tables..."
    python scripts/generate_tables.py --results_dir results/aggregated/ --output_dir results/tables/ || echo "⚠ Table generation had errors"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ABLATION 12M PIPELINE COMPLETE"
echo "  End: $(date -Iseconds)"
echo ""
echo "  Results:  results/aggregated/ablation_*.json"
echo "  Tables:   results/tables/"
echo "  Figures:  figures/"
echo ""
echo "  Next: Update README with new 12M ablation numbers."
echo "══════════════════════════════════════════════════════════════"
