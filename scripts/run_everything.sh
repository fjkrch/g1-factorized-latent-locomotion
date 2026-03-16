#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_everything.sh — Run the FULL experiment pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════
#
# Pipeline:
#   1. Sanity checks (env, GPU, disk)
#   2. Main comparison: 4 models × 4 tasks × 1 seed  =  16 training runs  (~96 min)
#   3. Ablation study:  7 ablations × 1 seed          =   7 training runs  (~42 min)
#   4. Evaluation of all trained runs                                       (~15 min)
#   5. Robustness sweeps: 4 models × 4 sweeps         =  16 eval runs      (~10 min)
#   6. Aggregation, tables & plots                                          (~2 min)
#
# Estimated total runtime: ~2.5–3 hours on RTX 4060 (2M steps / 512 envs per run)
#
# Usage:
#   bash scripts/run_everything.sh            # run everything
#   bash scripts/run_everything.sh --dry-run  # print commands, don't execute
#
# Prerequisites:
#   - conda env "env_isaaclab" with Isaac Lab + this project installed
#   - NVIDIA GPU with ≥8 GB VRAM
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Helpers ──
section() { echo -e "\n\n══════════════════════════════════════════════════════════════"; echo "  $1"; echo "  $(date -Iseconds)"; echo "══════════════════════════════════════════════════════════════"; }
run()     { if $DRY_RUN; then echo "[DRY RUN] $*"; else echo "[RUN] $*"; "$@"; fi; }
elapsed() { local t=$SECONDS; printf '%dh %dm %ds' $((t/3600)) $((t%3600/60)) $((t%60)); }

SECONDS=0
FAILED=0

section "0/6  SANITY CHECKS"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: $(python --version 2>&1)"
echo "  Disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "  Project root: $PROJECT_ROOT"

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Main Comparison Training (16 runs)
# ══════════════════════════════════════════════════════════════════════════════
section "1/6  MAIN COMPARISON — 4 models × 4 tasks × 1 seed = 16 runs"

TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
SEED=42
TOTAL_MAIN=$((${#TASKS[@]} * ${#MODELS[@]}))
CURRENT=0

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL_MAIN] Training: ${model} on ${task}  (elapsed: $(elapsed))"
        run bash scripts/run_train.sh --task "$task" --model "$model" --seed "$SEED" --variant full || {
            echo "  ⚠ FAILED: ${model}/${task}"
            FAILED=$((FAILED + 1))
        }
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Ablation Study (7 runs)
# ══════════════════════════════════════════════════════════════════════════════
section "2/6  ABLATIONS — 7 variants on randomized task"

ABLATIONS=(seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4)
TOTAL_ABL=${#ABLATIONS[@]}
CURRENT=0

for abl in "${ABLATIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_ABL] Ablation: ${abl}  (elapsed: $(elapsed))"
    run python scripts/train.py \
        --task configs/task/randomized.yaml \
        --model configs/model/dynamite.yaml \
        --ablation "configs/ablations/${abl}.yaml" \
        --seed "$SEED" \
        --variant "$abl" \
        --headless || {
        echo "  ⚠ FAILED: ablation ${abl}"
        FAILED=$((FAILED + 1))
    }
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Evaluation of ALL trained runs
# ══════════════════════════════════════════════════════════════════════════════
section "3/6  EVALUATION — all completed runs"

EVAL_COUNT=0
for run_dir in $(find outputs/ -mindepth 4 -maxdepth 4 -type d 2>/dev/null); do
    # Skip if already evaluated
    if [[ -f "$run_dir/eval_metrics.json" ]]; then
        echo "  SKIP (already evaluated): $run_dir"
        continue
    fi
    CKPT="$run_dir/checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="$run_dir/checkpoints/latest.pt"
    fi
    if [[ -f "$CKPT" ]]; then
        EVAL_COUNT=$((EVAL_COUNT + 1))
        echo "  [$EVAL_COUNT] Evaluating: $run_dir"
        run python scripts/eval.py --run_dir "$run_dir" --num_episodes 50 || {
            echo "  ⚠ eval failed: $run_dir"
            FAILED=$((FAILED + 1))
        }
    fi
done
echo "  Evaluated: $EVAL_COUNT runs"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Robustness Sweeps (eval only)
# ══════════════════════════════════════════════════════════════════════════════
section "4/6  ROBUSTNESS SWEEPS — 4 models × 4 perturbation types"

SWEEPS=(push_magnitude friction motor_strength action_delay)
SWEEP_TASK="randomized"

for model in "${MODELS[@]}"; do
    CKPT_DIR="outputs/${SWEEP_TASK}/${model}_full/seed_${SEED}"
    LATEST_RUN=$(ls -td "${CKPT_DIR}"/*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "  WARNING: No run found for ${model} on ${SWEEP_TASK}. Skipping sweeps."
        continue
    fi
    CKPT="${LATEST_RUN}checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="${LATEST_RUN}checkpoints/latest.pt"
    fi
    for sweep in "${SWEEPS[@]}"; do
        echo "  Sweep: ${model} / ${sweep}"
        run python scripts/eval.py \
            --checkpoint "$CKPT" \
            --sweep "configs/sweeps/${sweep}.yaml" \
            --output_dir "results/sweeps/${model}/${sweep}" \
            --num_episodes 20 || {
            echo "  ⚠ sweep failed: ${model}/${sweep}"
            FAILED=$((FAILED + 1))
        }
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Aggregation
# ══════════════════════════════════════════════════════════════════════════════
section "5/6  AGGREGATION — aggregate seeds, generate tables"
run bash scripts/aggregate_all.sh --skip-eval || {
    echo "  ⚠ Aggregation had errors (non-fatal)"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Plots & Tables
# ══════════════════════════════════════════════════════════════════════════════
section "6/6  PLOTS & TABLES"
mkdir -p figures/
run python scripts/plot_results.py --results_dir results/aggregated --output_dir figures/ || {
    echo "  ⚠ Plotting failed (non-fatal)"
}
run python scripts/generate_tables.py --results_dir results/aggregated --output_dir figures/ || {
    echo "  ⚠ Table generation failed (non-fatal)"
}

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section "COMPLETE"
echo "  Total elapsed: $(elapsed)"
echo "  Training runs: $((TOTAL_MAIN + TOTAL_ABL))  (main: $TOTAL_MAIN, ablation: $TOTAL_ABL)"
echo "  Evaluations:   $EVAL_COUNT"
echo "  Failed:        $FAILED"
echo ""
echo "  Outputs:       outputs/"
echo "  Results:       results/"
echo "  Figures:       figures/"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "  ⚠ $FAILED step(s) failed. Check logs above."
    exit 1
else
    echo "  ✓ All steps completed successfully!"
fi
