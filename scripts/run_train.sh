#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_train.sh — Launch a SINGLE training run with full logging & manifest
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/run_train.sh --task flat --model mlp --seed 42
#   bash scripts/run_train.sh --task randomized --model dynamite --seed 43 --variant full
#   bash scripts/run_train.sh --task randomized --model dynamite --seed 42 --ablation no_latent --variant no_latent
#   bash scripts/run_train.sh --task flat --model mlp --seed 42 --override "train.total_timesteps=1000000"
#   bash scripts/run_train.sh --task flat --model mlp --seed 42 --dry-run
#
# Outputs:
#   outputs/{task}/{model}_{variant}/seed_{seed}/{timestamp}/
#     ├── config.yaml
#     ├── manifest.json
#     ├── metrics.csv
#     ├── tb/
#     ├── checkpoints/
#     ├── stdout.log
#     └── stderr.log
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Defaults ──
TASK=""
MODEL=""
SEED=42
VARIANT="full"
ABLATION=""
OVERRIDES=""
DRY_RUN=false
RESUME=""
CONDA_ENV="${DYNAMITE_CONDA_ENV:-env_isaaclab}"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)       TASK="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --variant)    VARIANT="$2"; shift 2 ;;
        --ablation)   ABLATION="$2"; shift 2 ;;
        --override)   OVERRIDES="$OVERRIDES --set $2"; shift 2 ;;
        --resume)     RESUME="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASK" || -z "$MODEL" ]]; then
    echo "ERROR: --task and --model are required"
    echo "Usage: bash scripts/run_train.sh --task flat --model mlp --seed 42"
    exit 1
fi

# ── Build command ──
CMD="python scripts/train.py"
CMD="$CMD --task configs/task/${TASK}.yaml"
CMD="$CMD --model configs/model/${MODEL}.yaml"
CMD="$CMD --seed $SEED"
CMD="$CMD --variant $VARIANT"
CMD="$CMD --headless"

if [[ -n "$ABLATION" ]]; then
    CMD="$CMD --ablation configs/ablations/${ABLATION}.yaml"
fi

if [[ -n "$RESUME" ]]; then
    CMD="$CMD --resume $RESUME"
fi

if [[ -n "$OVERRIDES" ]]; then
    CMD="$CMD $OVERRIDES"
fi

# ── Run ID for logging ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LABEL="${TASK}_${MODEL}_${VARIANT}_seed${SEED}"
LOG_PREFIX="outputs/${TASK}/${MODEL}_${VARIANT}/seed_${SEED}"

echo "════════════════════════════════════════════════════════════════"
echo "  RUN: $RUN_LABEL"
echo "  CMD: $CMD"
echo "  TIME: $(date -Iseconds)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "════════════════════════════════════════════════════════════════"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute: $CMD"
    exit 0
fi

# ── Activate conda environment ──
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ── Create log directory ──
mkdir -p "$LOG_PREFIX"

# ── Execute with stdout/stderr capture ──
$CMD \
    2>&1 | tee "${LOG_PREFIX}/train_${TIMESTAMP}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo "════════════════════════════════════════════════════════════════"
echo "  DONE: $RUN_LABEL"
echo "  EXIT: $EXIT_CODE"
echo "  TIME: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
