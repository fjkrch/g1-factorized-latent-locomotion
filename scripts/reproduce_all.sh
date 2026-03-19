#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# reproduce_all.sh — Full pipeline: train + eval + analysis (Items 1-9)
# ══════════════════════════════════════════════════════════════════════════════
#
# This is the ONE script to reproduce ALL results in the paper.
#
# Items:
#   1. Push validation (BLOCKER)
#   2. Train 4 models × 4 tasks × 3 seeds = 48 runs at 10M steps
#   3. Train 7 ablations × 3 seeds = 21 runs at 10M steps
#   4. Evaluate all checkpoints
#   5. OOD friction sweep
#   6. OOD push magnitude sweep
#   7. Disentanglement analysis
#   8. Extra OOD sweeps (motor_strength, action_delay)
#   9. Aggregate results + generate tables + figures
#
# Total: 69 training runs + evaluation + analysis
# Estimated time: ~12 hours on RTX 4060 Laptop GPU
#
# Usage:
#   bash scripts/reproduce_all.sh
#   bash scripts/reproduce_all.sh --dry-run
#   bash scripts/reproduce_all.sh --skip-training    # skip training, do eval/analysis only
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DYNAMITE_PYTHON="${DYNAMITE_PYTHON:-python}"

DRY_RUN=false
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)         DRY_RUN=true; shift ;;
        --skip-training)   SKIP_TRAINING=true; shift ;;
        *)                 echo "Unknown argument: $1"; exit 1 ;;
    esac
done

DRY_FLAG=""
if $DRY_RUN; then
    DRY_FLAG="--dry-run"
fi

echo "══════════════════════════════════════════════════════════════"
echo "  DynaMITE: FULL REPRODUCTION PIPELINE"
echo "  Start: $(date -Iseconds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Seeds: 42 43 44 | Timesteps: 10M"
echo "══════════════════════════════════════════════════════════════"

# ── Item 1: Push validation ──
echo ""
echo "════ ITEM 1: PUSH VALIDATION ════"
$DYNAMITE_PYTHON scripts/validate_push.py || {
    echo "FATAL: Push validation failed. Cannot proceed."
    exit 1
}
echo "[OK] Push validation passed"

if ! $SKIP_TRAINING; then
    # ── Item 2: Main training (48 runs) ──
    echo ""
    echo "════ ITEM 2: MAIN TRAINING (48 runs) ════"
    bash scripts/run_all_main.sh $DRY_FLAG

    # ── Item 3: Ablation training (21 runs) ──
    echo ""
    echo "════ ITEM 3: ABLATION TRAINING (21 runs) ════"
    bash scripts/run_ablations.sh $DRY_FLAG
fi

# ── Items 4-9: Post-training analysis ──
echo ""
echo "════ ITEMS 4-9: POST-TRAINING ANALYSIS ════"
bash scripts/run_post_training.sh $DRY_FLAG

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  FULL PIPELINE COMPLETE"
echo "  End: $(date -Iseconds)"
echo "  Results: results/"
echo "  Figures: figures/"
echo "══════════════════════════════════════════════════════════════"
