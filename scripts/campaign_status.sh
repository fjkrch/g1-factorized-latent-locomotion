#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# campaign_status.sh — Check status of the full experiment campaign
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "══════════════════════════════════════════════════════════════"
echo "  CAMPAIGN STATUS — $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Main comparison ──
echo ""
echo "▶ MAIN COMPARISON (4 models × 4 tasks × 5 seeds = 80 runs)"
MAIN_DONE=0
MAIN_MISSING=""
for task in flat push randomized terrain; do
    for model in mlp lstm transformer dynamite; do
        for seed in 42 43 44 45 46; do
            dir="outputs/${task}/${model}_full/seed_${seed}"
            manifest=$(find "$dir" -name "manifest.json" -print -quit 2>/dev/null || true)
            if [[ -n "$manifest" ]]; then
                status=$(python3 -c "import json; print(json.load(open('$manifest')).get('status',''))" 2>/dev/null || echo "unknown")
                if [[ "$status" == "completed" ]]; then
                    MAIN_DONE=$((MAIN_DONE + 1))
                else
                    MAIN_MISSING="$MAIN_MISSING  $dir ($status)\n"
                fi
            else
                MAIN_MISSING="$MAIN_MISSING  $dir (not started)\n"
            fi
        done
    done
done
echo "  Completed: $MAIN_DONE / 80"
if [[ -n "$MAIN_MISSING" ]]; then
    echo "  Missing/incomplete:"
    echo -e "$MAIN_MISSING"
fi

# ── 2. Ablations ──
echo "▶ ABLATIONS (7 variants × 5 seeds = 35 runs)"
ABL_DONE=0
ABL_MISSING=""
for abl in seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4; do
    for seed in 42 43 44 45 46; do
        dir="outputs/randomized/dynamite_${abl}/seed_${seed}"
        manifest=$(find "$dir" -name "manifest.json" -print -quit 2>/dev/null || true)
        if [[ -n "$manifest" ]]; then
            status=$(python3 -c "import json; print(json.load(open('$manifest')).get('status',''))" 2>/dev/null || echo "unknown")
            if [[ "$status" == "completed" ]]; then
                ABL_DONE=$((ABL_DONE + 1))
            else
                ABL_MISSING="$ABL_MISSING  $dir ($status)\n"
            fi
        else
            ABL_MISSING="$ABL_MISSING  $dir (not started)\n"
        fi
    done
done
echo "  Completed: $ABL_DONE / 35"
if [[ -n "$ABL_MISSING" ]]; then
    echo "  Missing: $(echo -e "$ABL_MISSING" | grep -c "not started") not started"
fi

# ── 3. Evaluations ──
echo ""
echo "▶ EVALUATIONS (recomputed)"
EVAL_DONE=$(find outputs/ -name "eval_recomputed.json" 2>/dev/null | wc -l)
TOTAL_CKPTS=$(find outputs/ -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | wc -l)
echo "  Recomputed evals: $EVAL_DONE / $TOTAL_CKPTS checkpoints"

# ── 4. Robustness sweeps ──
echo ""
echo "▶ ROBUSTNESS SWEEPS"
SWEEP_FILES=$(find results/sweeps/ -name "sweep_*.json" 2>/dev/null | wc -l)
echo "  Sweep results: $SWEEP_FILES / 16 expected"

# ── 5. Check for errors in latest campaign log ──
echo ""
echo "▶ HEALTH CHECK"
LATEST_LOG=$(ls -t logs/campaign_*.log 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    ERRORS=$(grep -ci "error\|exception\|traceback\|NaN\|nan\|FAILED" "$LATEST_LOG" 2>/dev/null || echo 0)
    echo "  Latest campaign log: $LATEST_LOG"
    echo "  Error mentions: $ERRORS"
    if [[ "$ERRORS" -gt 0 ]]; then
        echo "  Last errors:"
        grep -i "error\|exception\|traceback\|NaN\|FAILED" "$LATEST_LOG" | tail -5
    fi
else
    echo "  No campaign log found"
fi

# ── 6. GPU status ──
echo ""
echo "▶ GPU STATUS"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"

# ── 7. Active training process ──
echo ""
echo "▶ ACTIVE PROCESSES"
PROCS=$(pgrep -af "scripts/train.py" 2>/dev/null | head -5 || echo "  None")
echo "  $PROCS"

# ── 8. Currently running run (from log tail) ──  
echo ""
if [[ -n "$LATEST_LOG" ]]; then
    echo "▶ CURRENT ACTIVITY (last log lines):"
    tail -5 "$LATEST_LOG" 2>/dev/null
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  STATUS CHECK COMPLETE — $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
