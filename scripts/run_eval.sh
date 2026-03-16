#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_eval.sh — Evaluate a single checkpoint or an entire run directory
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   # Eval best checkpoint from a run
#   bash scripts/run_eval.sh --run-dir outputs/randomized/dynamite_full/seed_42/20260316_*/
#
#   # Eval specific checkpoint on a different task
#   bash scripts/run_eval.sh --checkpoint path/to/best.pt --task push
#
#   # Robustness sweep
#   bash scripts/run_eval.sh --run-dir outputs/randomized/dynamite_full/seed_42/20260316_*/ --sweep push_magnitude
#
# Outputs:
#   {output_dir}/eval_metrics.json        (standard eval)
#   {output_dir}/sweep_{name}.json        (sweep eval)
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Defaults ──
CHECKPOINT=""
RUN_DIR=""
TASK=""
SWEEP=""
NUM_EPISODES=100
SEED=42
OUTPUT_DIR=""
DRY_RUN=false

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
        --run-dir)       RUN_DIR="$2"; shift 2 ;;
        --task)          TASK="$2"; shift 2 ;;
        --sweep)         SWEEP="$2"; shift 2 ;;
        --num-episodes)  NUM_EPISODES="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        *)               echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build command ──
CMD="python scripts/eval.py"

if [[ -n "$CHECKPOINT" ]]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
elif [[ -n "$RUN_DIR" ]]; then
    CMD="$CMD --run_dir $RUN_DIR"
else
    echo "ERROR: --checkpoint or --run-dir required"
    exit 1
fi

if [[ -n "$TASK" ]]; then
    CMD="$CMD --task configs/task/${TASK}.yaml"
fi

if [[ -n "$SWEEP" ]]; then
    CMD="$CMD --sweep configs/sweeps/${SWEEP}.yaml"
fi

CMD="$CMD --num_episodes $NUM_EPISODES --seed $SEED"

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  EVAL"
echo "  CMD: $CMD"
echo "  TIME: $(date -Iseconds)"
echo "════════════════════════════════════════════════════════════════"

if $DRY_RUN; then
    echo "[DRY RUN] Would execute: $CMD"
    exit 0
fi

$CMD
EXIT_CODE=$?

echo "════════════════════════════════════════════════════════════════"
echo "  EVAL DONE — exit $EXIT_CODE"
echo "════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
