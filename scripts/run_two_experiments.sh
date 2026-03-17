#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_two_experiments.sh — Run the 2 highest-impact paper-strengthening experiments:
#   1. Latent–dynamics correlation analysis  (~5 min)
#   2. OOD robustness sweeps                  (~20 min)
#
# Both are evaluation-only (no training). Uses existing 12M checkpoints.
# GPU-optimised: num_envs=32 for eval, CUDA memory allocator tuned.
#
# Usage:
#   bash scripts/run_two_experiments.sh
#   bash scripts/run_two_experiments.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Activate conda
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

SEED=42
TASK="randomized"

echo "══════════════════════════════════════════════════════════════"
echo "  TWO HIGH-IMPACT EXPERIMENTS"
echo "  1. Latent–dynamics correlation analysis"
echo "  2. OOD robustness sweeps (friction, delay, push)"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Latent–Dynamics Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ EXPERIMENT 1: Latent–Dynamics Correlation Analysis ━━━"
echo ""

DYNAMITE_CKPT=$(ls outputs/${TASK}/dynamite_full/seed_${SEED}/*/checkpoints/best.pt 2>/dev/null | head -1)
if [[ -z "$DYNAMITE_CKPT" ]]; then
    echo "ERROR: No DynaMITE checkpoint found"
    exit 1
fi
echo "  Checkpoint: $DYNAMITE_CKPT"

mkdir -p results/latent_analysis figures

if $DRY_RUN; then
    echo "  [DRY RUN] python scripts/run_latent_analysis.py --checkpoint $DYNAMITE_CKPT"
else
    python scripts/run_latent_analysis.py \
        --checkpoint "$DYNAMITE_CKPT" \
        --num_episodes 100 \
        --output_dir results/latent_analysis \
        --figure_dir figures \
        --seed $SEED
    echo "  ✓ Latent analysis complete"
fi

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: OOD Robustness Sweeps
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ EXPERIMENT 2: OOD Robustness Sweeps ━━━"
echo ""

MODELS=("mlp" "lstm" "transformer" "dynamite")
SWEEPS=("friction" "action_delay" "push_magnitude")
mkdir -p results/sweeps

for model in "${MODELS[@]}"; do
    CKPT_DIR="outputs/${TASK}/${model}_full/seed_${SEED}"
    LATEST_RUN=$(ls -td "${CKPT_DIR}"/*/ 2>/dev/null | head -1)

    if [[ -z "$LATEST_RUN" ]]; then
        echo "  WARNING: No run for ${model}. Skipping."
        continue
    fi

    CKPT="${LATEST_RUN}checkpoints/best.pt"
    [[ ! -f "$CKPT" ]] && CKPT="${LATEST_RUN}checkpoints/latest.pt"

    for sweep in "${SWEEPS[@]}"; do
        OUT_DIR="results/sweeps/${model}/${sweep}"
        mkdir -p "$OUT_DIR"
        echo "  [${model}] sweep=${sweep}"

        if $DRY_RUN; then
            echo "    [DRY RUN] python scripts/eval.py --checkpoint $CKPT --sweep configs/sweeps/${sweep}.yaml --output_dir $OUT_DIR --num_episodes 20 --seed $SEED"
        else
            python scripts/eval.py \
                --checkpoint "$CKPT" \
                --sweep "configs/sweeps/${sweep}.yaml" \
                --output_dir "$OUT_DIR" \
                --num_episodes 20 \
                --seed $SEED || {
                echo "    ⚠ FAILED: ${model}/${sweep}"
                continue
            }
            echo "    ✓ ${model}/${sweep}"
        fi
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PLOT SWEEP RESULTS
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━ Generating sweep plots ━━━"

if ! $DRY_RUN; then
    python scripts/plot_sweeps.py \
        --results_dir results/sweeps \
        --output_dir figures || echo "  ⚠ Sweep plotting had errors"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE"
echo "  End: $(date -Iseconds)"
echo ""
echo "  Latent analysis: results/latent_analysis/"
echo "  Sweep results:   results/sweeps/"
echo "  Figures:          figures/"
echo "══════════════════════════════════════════════════════════════"
