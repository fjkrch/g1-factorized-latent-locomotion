#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_ood_campaign.sh — Complete OOD evaluation campaign
# ══════════════════════════════════════════════════════════════════════════════
#
# Must-do 1: Re-run all existing sweeps with behavioral metrics (randomized task)
#   friction, push_magnitude, action_delay × 4 models × 5 seeds = 60
#
# Must-do 2: Push-task + terrain-task OOD on push_magnitude sweep
#   push_magnitude × 4 models × 5 seeds × 2 tasks = 40
#
# Must-do 3: Unseen-range delay + combined-shift stress test (randomized task)
#   action_delay_unseen × 4 models × 5 seeds = 20
#   combined_shift × 4 models × 5 seeds = 20
#
# Total: 60 + 40 + 40 = 140 sweep evaluations
#
# Results go to: results/ood_v2/{sweep_type}/{task}/{model}_seed{N}/
#
# Usage:
#   bash scripts/run_ood_campaign.sh
#   bash scripts/run_ood_campaign.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ISAAC_PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
    echo "ERROR: Isaac Lab Python not found at $ISAAC_PYTHON"
    exit 1
fi
export PYTHON_CMD="$ISAAC_PYTHON"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

MODELS=(mlp lstm transformer dynamite)
SEEDS=(42 43 44 45 46)
EPISODES=50
EVAL_SEED=42
RESULTS_BASE="results/ood_v2"

TOTAL=0
SKIPPED=0
RAN=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  OOD EVALUATION CAMPAIGN"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

find_checkpoint() {
    local task="$1" model="$2" seed="$3"
    local pattern="outputs/${task}/${model}_full/seed_${seed}/*/checkpoints/best.pt"
    # shellcheck disable=SC2086
    local ckpt
    ckpt=$(ls -1 $pattern 2>/dev/null | head -1)
    echo "$ckpt"
}

run_sweep() {
    local task="$1" model="$2" seed="$3" sweep="$4" label="$5"
    TOTAL=$((TOTAL + 1))

    local ckpt
    ckpt=$(find_checkpoint "$task" "$model" "$seed")
    if [[ -z "$ckpt" ]]; then
        echo "  [WARN] No checkpoint: ${task}/${model}/seed_${seed}"
        FAILED=$((FAILED + 1))
        return
    fi

    local outdir="${RESULTS_BASE}/${sweep}/${task}/${model}_seed${seed}"
    local outfile="${outdir}/sweep_${sweep}.json"

    if [[ -f "$outfile" ]]; then
        # Check if it has the new behavioral metrics
        if grep -q "tracking_error" "$outfile" 2>/dev/null; then
            echo "  [SKIP] ${label} (already has behavioral metrics)"
            SKIPPED=$((SKIPPED + 1))
            return
        fi
    fi

    RAN=$((RAN + 1))
    if $DRY_RUN; then
        echo "  [DRY] ${label}"
        return
    fi

    echo "  [RUN] ${label}"
    local logfile="${outdir}/eval.log"
    mkdir -p "$outdir"
    if bash scripts/run_eval.sh \
        --checkpoint "$ckpt" \
        --sweep "$sweep" \
        --output-dir "$outdir" \
        --num-episodes "$EPISODES" \
        --seed "$EVAL_SEED" > "$logfile" 2>&1; then
        # Verify JSON was actually written
        if [[ -f "$outdir/sweep_${sweep}.json" ]]; then
            echo "  [OK]  ${label}"
            # Print summary from log
            grep -E "reward=|fail=|track_err=" "$logfile" | tail -3 || true
        else
            echo "  [FAIL] ${label} (no JSON output)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  [FAIL] ${label} (exit code $?)"
        tail -5 "$logfile" 2>/dev/null || true
        FAILED=$((FAILED + 1))
    fi
}

# ══════════════════════════════════════════════════════════════
# MUST-DO 1: Re-run existing sweeps with behavioral metrics
# (randomized task, friction/push_magnitude/action_delay)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── MUST-DO 1: Behavioral metrics on existing sweeps (randomized) ──"
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for sweep in friction push_magnitude action_delay; do
            run_sweep "randomized" "$model" "$seed" "$sweep" \
                "M1: ${model} s${seed} ${sweep}"
        done
    done
done

# ══════════════════════════════════════════════════════════════
# MUST-DO 2: Push-task + terrain-task OOD (push_magnitude sweep)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── MUST-DO 2: Cross-task OOD — push_magnitude on push + terrain ──"
for task in push terrain; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_sweep "$task" "$model" "$seed" "push_magnitude" \
                "M2: ${task}/${model} s${seed} push_mag"
        done
    done
done

# ══════════════════════════════════════════════════════════════
# MUST-DO 3: Unseen-range delay + combined-shift (randomized)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── MUST-DO 3: Unseen range + combined shift (randomized) ──"
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_sweep "randomized" "$model" "$seed" "action_delay_unseen" \
            "M3: ${model} s${seed} delay_unseen"
        run_sweep "randomized" "$model" "$seed" "combined_shift" \
            "M3: ${model} s${seed} combined"
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  OOD CAMPAIGN COMPLETE"
echo "  Total: ${TOTAL}  |  Skipped: ${SKIPPED}  |  Ran: ${RAN}  |  Failed: ${FAILED}"
echo "  End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
