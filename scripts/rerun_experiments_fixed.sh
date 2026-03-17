#!/usr/bin/env bash
# Wait for depth ablation training to finish, then rerun both experiments.
# Fixes applied:
#   Exp 1: Latent analysis — seeded history buffer, NaN filtering in correlations
#   Exp 2: Sweeps — reuse env across sweep values (prevents mock env fallback)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

SEED=42
TASK="randomized"

echo "══════════════════════════════════════════════════════════════"
echo "  RERUNNING FIXED EXPERIMENTS"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# Clear old results
rm -rf results/latent_analysis results/sweeps
mkdir -p results/latent_analysis results/sweeps figures

# ── EXPERIMENT 1: Latent Analysis (fixed) ──
echo ""
echo "━━━ EXPERIMENT 1: Latent–Dynamics Correlation Analysis (FIXED) ━━━"

DYNAMITE_CKPT=$(ls outputs/${TASK}/dynamite_full/seed_${SEED}/*/checkpoints/best.pt 2>/dev/null | head -1)
if [[ -z "$DYNAMITE_CKPT" ]]; then
    echo "ERROR: No DynaMITE checkpoint found"
    exit 1
fi
echo "  Checkpoint: $DYNAMITE_CKPT"

python scripts/run_latent_analysis.py \
    --checkpoint "$DYNAMITE_CKPT" \
    --num_episodes 100 \
    --output_dir results/latent_analysis \
    --figure_dir figures \
    --seed $SEED
echo "  ✓ Latent analysis complete"

# ── EXPERIMENT 2: OOD Robustness Sweeps (fixed) ──
echo ""
echo "━━━ EXPERIMENT 2: OOD Robustness Sweeps (FIXED) ━━━"

MODELS=("mlp" "lstm" "transformer" "dynamite")
SWEEPS=("friction" "action_delay" "push_magnitude")

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

        # Each (model, sweep) runs as separate process (clean Isaac Sim session)
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
    done
done

# ── PLOT SWEEP RESULTS ──
echo ""
echo "━━━ Generating sweep plots ━━━"
python scripts/plot_sweeps.py \
    --results_dir results/sweeps \
    --output_dir figures || echo "  ⚠ Sweep plotting had errors"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE — All experiments complete"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
