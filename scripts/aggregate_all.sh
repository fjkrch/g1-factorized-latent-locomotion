#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# aggregate_all.sh — Run full aggregation pipeline after experiments complete
# ══════════════════════════════════════════════════════════════════════════════
#
# Steps:
#   1. Validate all runs
#   2. Run eval on any runs missing eval_metrics.json
#   3. Aggregate seeds for main comparison
#   4. Aggregate seeds for ablations
#   5. Generate summary tables
#
# Usage:
#   bash scripts/aggregate_all.sh
#   bash scripts/aggregate_all.sh --skip-eval     # skip re-evaluation
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SKIP_EVAL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-eval) SKIP_EVAL=true; shift ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════════"
echo "  AGGREGATION PIPELINE"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Step 1: Validate runs ──
echo ""
echo "── Step 1: Validating all runs ──"
python3 -m src.utils.validate_runs --base-dir outputs/ || true

# ── Step 2: Run missing evals ──
if ! $SKIP_EVAL; then
    echo ""
    echo "── Step 2: Running missing evaluations ──"
    for run_dir in $(find outputs/ -mindepth 4 -maxdepth 4 -type d 2>/dev/null); do
        if [[ ! -f "$run_dir/eval_metrics.json" ]]; then
            CKPT="$run_dir/checkpoints/best.pt"
            if [[ -f "$CKPT" ]]; then
                echo "  Evaluating: $run_dir"
                python3 scripts/eval.py --run_dir "$run_dir" --num_episodes 50 || echo "  ⚠ eval failed: $run_dir"
            fi
        fi
    done
fi

# ── Step 3: Aggregate main comparison ──
echo ""
echo "── Step 3: Aggregating main comparison ──"
mkdir -p results/aggregated

TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        VARIANT="full"
        SEED_DIRS=""
        for seed_dir in outputs/${task}/${model}_${VARIANT}/seed_*/; do
            if [[ -d "$seed_dir" ]]; then
                # Find the most recent run (by timestamp dir)
                LATEST_RUN=$(find "$seed_dir" -maxdepth 1 -type d | sort | tail -1)
                if [[ -n "$LATEST_RUN" && "$LATEST_RUN" != "$seed_dir" ]]; then
                    SEED_DIRS="$SEED_DIRS $LATEST_RUN"
                fi
            fi
        done
        if [[ -n "$SEED_DIRS" ]]; then
            echo "  Aggregating: ${task}/${model}_${VARIANT} ($(echo $SEED_DIRS | wc -w) seeds)"
            python3 scripts/aggregate_seeds.py \
                --run_dirs $SEED_DIRS \
                --output "results/aggregated/${task}_${model}_${VARIANT}.json" \
                2>/dev/null || echo "  ⚠ aggregation failed: ${task}/${model}"
        fi
    done
done

# ── Step 4: Aggregate ablations ──
echo ""
echo "── Step 4: Aggregating ablations ──"

# Ablation variant names match --variant passed in run_ablations.sh
ABLATIONS=(depth_1 depth_4 no_aux_loss no_latent single_latent seq_len_4 seq_len_16)
TASK="randomized"

for abl in "${ABLATIONS[@]}"; do
    dir_name="dynamite_${abl}"
    SEED_DIRS=""
    for seed_dir in outputs/${TASK}/${dir_name}/seed_*/; do
        if [[ -d "$seed_dir" ]]; then
            LATEST_RUN=$(find "$seed_dir" -maxdepth 1 -type d | sort | tail -1)
            if [[ -n "$LATEST_RUN" && "$LATEST_RUN" != "$seed_dir" ]]; then
                SEED_DIRS="$SEED_DIRS $LATEST_RUN"
            fi
        fi
    done
    if [[ -n "$SEED_DIRS" ]]; then
        echo "  Aggregating ablation: ${abl} ($(echo $SEED_DIRS | wc -w) seeds)"
        python3 scripts/aggregate_seeds.py \
            --run_dirs $SEED_DIRS \
            --output "results/aggregated/ablation_${abl}.json" \
            2>/dev/null || echo "  ⚠ aggregation failed: ablation_${abl}"
    fi
done

# ── Step 5: Generate tables ──
echo ""
echo "── Step 5: Generating tables ──"
mkdir -p results/tables
python3 scripts/generate_tables.py \
    --results_dir results/aggregated/ \
    --output_dir results/tables/ \
    2>/dev/null || echo "  ⚠ table generation failed"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  AGGREGATION COMPLETE"
echo "  Results:  results/aggregated/"
echo "  Tables:   results/tables/"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
