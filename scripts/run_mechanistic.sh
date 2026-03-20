#!/usr/bin/env bash
# ==========================================================
# Mechanistic Analysis Campaign
# ==========================================================
#
# Runs all 3 mechanistic analyses:
#   1) Gradient flow (retrain DynaMITE, 3 seeds)
#   2) Representation geometry (2 models × 5 seeds)
#   3) MINE mutual information (combined with geometry)
# Then generates figures and writes summary.
#
# Usage:
#   bash scripts/run_mechanistic.sh          # full pipeline
#   bash scripts/run_mechanistic.sh --skip-gradient  # skip retraining
# ==========================================================

set -euo pipefail

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
PROJECT="/home/chyanin/robotpaper"
RESULTS="$PROJECT/results/mechanistic"
FIGURES="$PROJECT/figures"
OUTPUTS="$PROJECT/outputs/randomized"

SKIP_GRADIENT=false
for arg in "$@"; do
    [[ "$arg" == "--skip-gradient" ]] && SKIP_GRADIENT=true
done

mkdir -p "$RESULTS/gradient_analysis"
mkdir -p "$RESULTS/geometry_analysis"
mkdir -p "$RESULTS/mi_analysis"
mkdir -p "$FIGURES"

cd "$PROJECT"

echo "========================================================"
echo " Mechanistic Analysis Campaign"
echo " Started: $(date)"
echo "========================================================"

# ──────────────────────────────────────────────────────────
# 1. Gradient Flow Analysis (retrain DynaMITE, 3 seeds)
# ──────────────────────────────────────────────────────────
if [[ "$SKIP_GRADIENT" == false ]]; then
    echo ""
    echo "=== Phase 1: Gradient Flow Analysis ==="
    for SEED in 42 43 44; do
        OUT_FILE="$RESULTS/gradient_analysis/seed_${SEED}.json"
        if [[ -f "$OUT_FILE" ]]; then
            echo "  [SKIP] seed_${SEED} already complete"
            continue
        fi
        echo "  Training DynaMITE seed=$SEED with gradient instrumentation..."
        $PYTHON scripts/gradient_flow_analysis.py \
            --seed $SEED \
            --output_dir "$RESULTS/gradient_analysis" \
            --gradient_log_interval 10 \
            2>&1 | tee "$RESULTS/gradient_analysis/log_seed_${SEED}.txt"
        echo "  Completed seed=$SEED at $(date)"
    done
    echo "=== Phase 1 complete ==="
else
    echo ""
    echo "=== Phase 1: SKIPPED (--skip-gradient) ==="
fi

# ──────────────────────────────────────────────────────────
# 2+3. Representation Geometry + MINE (existing checkpoints)
# ──────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2+3: Representation Geometry + MINE ==="

for MODEL in dynamite lstm; do
    for SEED in 42 43 44 45 46; do
        GEO_FILE="$RESULTS/geometry_analysis/geometry_${MODEL}_seed${SEED}.json"
        MI_FILE="$RESULTS/mi_analysis/mine_${MODEL}_seed${SEED}.json"

        if [[ -f "$GEO_FILE" ]] && [[ -f "$MI_FILE" ]]; then
            echo "  [SKIP] ${MODEL} seed=${SEED} already complete"
            continue
        fi

        # Find checkpoint
        CKPT_DIR="$OUTPUTS/${MODEL}_full/seed_${SEED}"
        CKPT=$(find "$CKPT_DIR" -name "best.pt" -type f 2>/dev/null | head -1)
        if [[ -z "$CKPT" ]]; then
            echo "  [WARN] No checkpoint for ${MODEL} seed=${SEED}, skipping"
            continue
        fi

        echo "  Analyzing ${MODEL} seed=${SEED}..."
        echo "    Checkpoint: $CKPT"
        $PYTHON scripts/representation_analysis.py \
            --model_type $MODEL \
            --ckpt "$CKPT" \
            --seed $SEED \
            --output_dir "$RESULTS" \
            --num_episodes 200 \
            2>&1 | tee "$RESULTS/log_repr_${MODEL}_seed${SEED}.txt"
        echo "  Completed ${MODEL} seed=${SEED} at $(date)"
    done
done

echo "=== Phases 2+3 complete ==="

# ──────────────────────────────────────────────────────────
# Aggregate results
# ──────────────────────────────────────────────────────────
echo ""
echo "=== Aggregating results ==="
$PYTHON scripts/representation_analysis.py --aggregate --output_dir "$RESULTS"

# ──────────────────────────────────────────────────────────
# Generate figures
# ──────────────────────────────────────────────────────────
echo ""
echo "=== Generating figures ==="
$PYTHON scripts/plot_mechanistic.py \
    --results_dir "$RESULTS" \
    --figures_dir "$FIGURES"

echo ""
echo "========================================================"
echo " Mechanistic Analysis Campaign Complete"
echo " Finished: $(date)"
echo "========================================================"
echo ""
echo "Outputs:"
echo "  Gradient data: $RESULTS/gradient_analysis/"
echo "  Geometry data: $RESULTS/geometry_analysis/"
echo "  MINE data:     $RESULTS/mi_analysis/"
echo "  Figures:       $FIGURES/gradient_norms.png"
echo "                 $FIGURES/cosine_similarity.png"
echo "                 $FIGURES/geometry_comparison.png"
echo "                 $FIGURES/mi_comparison.png"
