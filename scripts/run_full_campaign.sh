#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_full_campaign.sh — Full experiment campaign: training → eval → aggregate
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs:
#   Phase 1: Main experiments (push/randomized/terrain × 4 models × 5 seeds = 60 runs)
#   Phase 2: Ablations (7 ablations × 5 seeds = 35 runs)
#   Phase 3: Re-evaluate ALL checkpoints (fresh, consistent protocol)
#   Phase 4: Robustness sweeps (4 models × 4 sweeps = 16 evals)
#   Phase 5: Aggregate results & generate tables
#
# Usage:
#   bash scripts/run_full_campaign.sh              # run everything
#   bash scripts/run_full_campaign.sh --dry-run    # preview only
#
# Launch detached:
#   setsid bash -c 'cd /home/chyanin/robotpaper && bash scripts/run_full_campaign.sh > logs/campaign_$(date +%Y%m%d_%H%M%S).log 2>&1' & disown
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

CAMPAIGN_START=$(date -Iseconds)
PHASE_FAILURES=0

log_phase() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  PHASE $1: $2"
    echo "║  Time: $(date -Iseconds)"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

log_status() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  STATUS REPORT — $(date -Iseconds)"
    echo "│  Campaign started: $CAMPAIGN_START"
    echo "│  Phase failures so far: $PHASE_FAILURES"
    echo "└──────────────────────────────────────────────────────────────┘"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Main experiments (skip flat — already done with 5 seeds)
# ══════════════════════════════════════════════════════════════════════════════
log_phase 1 "Main experiments (push/randomized/terrain)"

TASKS=(push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
SEEDS=(42 43 44 45 46)

P1_TOTAL=$((${#TASKS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))
P1_CURRENT=0
P1_SKIPPED=0
P1_FAILED=0

echo "  Runs: ${#TASKS[@]} tasks × ${#MODELS[@]} models × ${#SEEDS[@]} seeds = $P1_TOTAL"

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            P1_CURRENT=$((P1_CURRENT + 1))
            LABEL="${task}/${model}_full/seed_${seed}"

            # Skip completed
            EXISTING=$(find "outputs/${task}/${model}_full/seed_${seed}" -name "manifest.json" -print -quit 2>/dev/null || true)
            if [[ -n "$EXISTING" ]]; then
                STATUS=$(python3 -c "import json; print(json.load(open('$EXISTING')).get('status',''))" 2>/dev/null || echo "")
                if [[ "$STATUS" == "completed" ]]; then
                    echo "[P1 $P1_CURRENT/$P1_TOTAL] SKIP: $LABEL"
                    P1_SKIPPED=$((P1_SKIPPED + 1))
                    continue
                fi
            fi

            echo ""
            echo "[P1 $P1_CURRENT/$P1_TOTAL] RUN: $LABEL"

            if $DRY_RUN; then
                echo "  [DRY] bash scripts/run_train.sh --task $task --model $model --seed $seed --variant full"
                continue
            fi

            bash scripts/run_train.sh --task "$task" --model "$model" --seed "$seed" --variant full || {
                echo "  ⚠ FAILED: $LABEL"
                P1_FAILED=$((P1_FAILED + 1))
            }
        done
    done
done

echo ""
echo "  Phase 1 complete: $P1_TOTAL total | $P1_SKIPPED skipped | $P1_FAILED failed"
PHASE_FAILURES=$((PHASE_FAILURES + P1_FAILED))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Ablation experiments
# ══════════════════════════════════════════════════════════════════════════════
log_phase 2 "Ablation experiments (randomized/dynamite)"

ABL_TASK="randomized"
ABL_MODEL="dynamite"
ABLATIONS=(seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4)

P2_TOTAL=$((${#ABLATIONS[@]} * ${#SEEDS[@]}))
P2_CURRENT=0
P2_SKIPPED=0
P2_FAILED=0

echo "  Runs: ${#ABLATIONS[@]} ablations × ${#SEEDS[@]} seeds = $P2_TOTAL"

for abl in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        P2_CURRENT=$((P2_CURRENT + 1))
        LABEL="${ABL_TASK}/${ABL_MODEL}_${abl}/seed_${seed}"

        # Skip completed
        EXISTING=$(find "outputs/${ABL_TASK}/${ABL_MODEL}_${abl}/seed_${seed}" -name "manifest.json" -print -quit 2>/dev/null || true)
        if [[ -n "$EXISTING" ]]; then
            STATUS=$(python3 -c "import json; print(json.load(open('$EXISTING')).get('status',''))" 2>/dev/null || echo "")
            if [[ "$STATUS" == "completed" ]]; then
                echo "[P2 $P2_CURRENT/$P2_TOTAL] SKIP: $LABEL"
                P2_SKIPPED=$((P2_SKIPPED + 1))
                continue
            fi
        fi

        echo ""
        echo "[P2 $P2_CURRENT/$P2_TOTAL] RUN: $LABEL"

        if $DRY_RUN; then
            echo "  [DRY] bash scripts/run_train.sh --task $ABL_TASK --model $ABL_MODEL --seed $seed --ablation $abl --variant $abl"
            continue
        fi

        bash scripts/run_train.sh --task "$ABL_TASK" --model "$ABL_MODEL" --seed "$seed" \
            --ablation "$abl" --variant "$abl" || {
            echo "  ⚠ FAILED: $LABEL"
            P2_FAILED=$((P2_FAILED + 1))
        }
    done
done

echo ""
echo "  Phase 2 complete: $P2_TOTAL total | $P2_SKIPPED skipped | $P2_FAILED failed"
PHASE_FAILURES=$((PHASE_FAILURES + P2_FAILED))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Re-evaluate ALL checkpoints (fresh, uniform protocol)
# ══════════════════════════════════════════════════════════════════════════════
log_phase 3 "Re-evaluate all checkpoints (100 episodes each)"

P3_TOTAL=0
P3_DONE=0
P3_FAILED=0

# Collect all completed run dirs
EVAL_DIRS=()
for run_dir in $(find outputs/ -name "manifest.json" -exec dirname {} \; 2>/dev/null | sort); do
    STATUS=$(python3 -c "import json; print(json.load(open('$run_dir/manifest.json')).get('status',''))" 2>/dev/null || echo "")
    if [[ "$STATUS" == "completed" ]]; then
        EVAL_DIRS+=("$run_dir")
    fi
done

P3_TOTAL=${#EVAL_DIRS[@]}
echo "  Found $P3_TOTAL completed runs to re-evaluate"

for run_dir in "${EVAL_DIRS[@]}"; do
    P3_DONE=$((P3_DONE + 1))
    SHORT_LABEL=$(echo "$run_dir" | sed "s|outputs/||")
    echo ""
    echo "[P3 $P3_DONE/$P3_TOTAL] EVAL: $SHORT_LABEL"

    CKPT="$run_dir/checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="$run_dir/checkpoints/latest.pt"
    fi
    if [[ ! -f "$CKPT" ]]; then
        echo "  ⚠ SKIP: no checkpoint"
        P3_FAILED=$((P3_FAILED + 1))
        continue
    fi

    # Store re-evaluation separately
    EVAL_OUTPUT="$run_dir/eval_recomputed"
    mkdir -p "$EVAL_OUTPUT"

    if $DRY_RUN; then
        echo "  [DRY] python3 scripts/eval.py --run_dir $run_dir --num_episodes 100 --seed 42 --output_dir $EVAL_OUTPUT"
        continue
    fi

    python3 scripts/eval.py --run_dir "$run_dir" --num_episodes 100 --seed 42 \
        --output_dir "$EVAL_OUTPUT" || {
        echo "  ⚠ FAILED eval: $SHORT_LABEL"
        P3_FAILED=$((P3_FAILED + 1))
    }
done

echo ""
echo "  Phase 3 complete: $P3_TOTAL total | $P3_FAILED failed"
PHASE_FAILURES=$((PHASE_FAILURES + P3_FAILED))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Robustness sweeps (on randomized task, seed 42)
# ══════════════════════════════════════════════════════════════════════════════
log_phase 4 "Robustness sweeps"

SWEEP_TASK="randomized"
SWEEP_MODELS=(mlp lstm transformer dynamite)
SWEEPS=(push_magnitude friction motor_strength action_delay)
SWEEP_SEED=42

P4_TOTAL=$((${#SWEEP_MODELS[@]} * ${#SWEEPS[@]}))
P4_CURRENT=0
P4_FAILED=0

echo "  Runs: ${#SWEEP_MODELS[@]} models × ${#SWEEPS[@]} sweeps = $P4_TOTAL"

for model in "${SWEEP_MODELS[@]}"; do
    for sweep in "${SWEEPS[@]}"; do
        P4_CURRENT=$((P4_CURRENT + 1))
        echo ""
        echo "[P4 $P4_CURRENT/$P4_TOTAL] SWEEP: model=$model sweep=$sweep"

        RUN_DIR=$(find "outputs/${SWEEP_TASK}/${model}_full/seed_${SWEEP_SEED}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -1 || true)
        if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" || "$RUN_DIR" == "outputs/${SWEEP_TASK}/${model}_full/seed_${SWEEP_SEED}" ]]; then
            echo "  ⚠ SKIP: no run dir for $model"
            P4_FAILED=$((P4_FAILED + 1))
            continue
        fi

        CKPT="$RUN_DIR/checkpoints/best.pt"
        [[ ! -f "$CKPT" ]] && CKPT="$RUN_DIR/checkpoints/latest.pt"
        if [[ ! -f "$CKPT" ]]; then
            echo "  ⚠ SKIP: no checkpoint"
            P4_FAILED=$((P4_FAILED + 1))
            continue
        fi

        OUTPUT_DIR="results/sweeps/${model}_full/${sweep}"
        mkdir -p "$OUTPUT_DIR"

        if $DRY_RUN; then
            echo "  [DRY] bash scripts/run_eval.sh --checkpoint $CKPT --sweep $sweep --output-dir $OUTPUT_DIR"
            continue
        fi

        bash scripts/run_eval.sh --checkpoint "$CKPT" --sweep "$sweep" --output-dir "$OUTPUT_DIR" || {
            echo "  ⚠ FAILED: $model/$sweep"
            P4_FAILED=$((P4_FAILED + 1))
        }
    done
done

echo ""
echo "  Phase 4 complete: $P4_TOTAL total | $P4_FAILED failed"
PHASE_FAILURES=$((PHASE_FAILURES + P4_FAILED))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Aggregate & verify
# ══════════════════════════════════════════════════════════════════════════════
log_phase 5 "Aggregate results & generate tables"

if $DRY_RUN; then
    echo "  [DRY] bash scripts/aggregate_all.sh --skip-eval"
else
    bash scripts/aggregate_all.sh --skip-eval || {
        echo "  ⚠ Aggregation had failures"
        PHASE_FAILURES=$((PHASE_FAILURES + 1))
    }
fi

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      CAMPAIGN COMPLETE                             ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Started:  $CAMPAIGN_START"
echo "║  Ended:    $(date -Iseconds)"
echo "║                                                                    ║"
echo "║  Phase 1 (Main):      $P1_TOTAL runs, $P1_SKIPPED skipped, $P1_FAILED failed"
echo "║  Phase 2 (Ablations): $P2_TOTAL runs, $P2_SKIPPED skipped, $P2_FAILED failed"
echo "║  Phase 3 (Re-eval):   $P3_TOTAL evals, $P3_FAILED failed"
echo "║  Phase 4 (Sweeps):    $P4_TOTAL sweeps, $P4_FAILED failed"
echo "║  Phase 5 (Aggregate): done"
echo "║                                                                    ║"
echo "║  Total failures: $PHASE_FAILURES"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [[ $PHASE_FAILURES -gt 0 ]]; then
    echo "WARNING: $PHASE_FAILURES total failures across campaign. Review logs."
    exit 1
fi
