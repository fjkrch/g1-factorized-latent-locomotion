#!/usr/bin/env bash
# =============================================================================
# Remaining pub experiments — OOM-safe version
#
# Fixes: Uses 256 envs (vs 512) to reduce memory ~40%.
#        Each job is independent — if one dies, the rest still run.
# 
# Ablations:  8 training runs  (~16 min each with 256 envs = ~2h 8m)
# OOD Sweeps: 18 eval runs     (~3-4 min each = ~1h)
# Latent:     3 analysis runs  (~5 min each = ~15 min)
# Total: ~3h 20m
# =============================================================================
set -uo pipefail

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="logs/pub_final_${TIMESTAMP}.log"
mkdir -p logs/ablations logs/sweeps logs/latent

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOGFILE"; }

log "================================================================"
log " Publication Experiments — OOM-safe — ${TIMESTAMP}"
log " RAM: $(free -h | awk '/Mem:/{print $2}'), Swap: $(free -h | awk '/Swap:/{print $2}')"
log "================================================================"

FAIL_COUNT=0

# Helper: run a command, catch failures without exiting the whole script
# Also sets OOM score to -500 so kernel kills Chrome before training
run_job() {
    local label="$1"; shift
    log "START: ${label}"
    "$@" 2>&1 &
    local job_pid=$!
    # Protect from OOM killer (requires earlier sudo caching)
    sudo sh -c "echo -500 > /proc/$job_pid/oom_score_adj" 2>/dev/null || true
    if wait $job_pid; then
        log "DONE:  ${label}"
    else
        log "FAIL:  ${label} (exit code $?)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ----------------------------------------------------------------
# PHASE 1: Ablation training (8 runs)
# ----------------------------------------------------------------
log ""
log "=== PHASE 1: Ablations ==="

for abl in no_aux_loss no_latent single_latent; do
    for seed in 42 43 44; do
        # Skip completed no_aux_loss/42
        if [[ "$abl" == "no_aux_loss" && "$seed" == "42" ]]; then
            log "SKIP:  ${abl}/seed_${seed} (already completed)"
            continue
        fi

        run_job "ABL ${abl}/seed_${seed}" \
            python3 scripts/train.py \
                --task randomized --model dynamite \
                --ablation "configs/ablations/${abl}.yaml" \
                --variant "${abl}" --seed "${seed}" --headless \
                --set env.num_envs=256
    done
done

log ""
log "=== PHASE 1 COMPLETE ==="

# ----------------------------------------------------------------
# PHASE 2: OOD sweeps (18 runs)
# ----------------------------------------------------------------
log ""
log "=== PHASE 2: OOD Sweeps ==="

for sweep in friction push_magnitude action_delay; do
    for model in dynamite lstm; do
        for seed in 42 43 44; do
            CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" \
                -name "best.pt" -type f 2>/dev/null | head -1)

            if [[ -z "$CKPT" ]]; then
                log "SKIP:  SW ${sweep}/${model}/seed_${seed} — no checkpoint"
                continue
            fi

            OUTDIR="results/sweeps_multiseed/${sweep}/${model}_seed${seed}"
            mkdir -p "$OUTDIR"

            run_job "SW ${sweep}/${model}/seed_${seed}" \
                python3 scripts/eval.py \
                    --checkpoint "$CKPT" \
                    --task configs/task/randomized.yaml \
                    --sweep "configs/sweeps/${sweep}.yaml" \
                    --num_episodes 50 --seed "${seed}" \
                    --output_dir "$OUTDIR"
        done
    done
done

log ""
log "=== PHASE 2 COMPLETE ==="

# ----------------------------------------------------------------
# PHASE 3: Latent validation (3 runs)
# ----------------------------------------------------------------
log ""
log "=== PHASE 3: Latent Analysis ==="

for seed in 42 43 44; do
    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}/" \
        -name "best.pt" -type f 2>/dev/null | head -1)

    if [[ -z "$CKPT" ]]; then
        log "SKIP:  LAT seed_${seed} — no checkpoint"
        continue
    fi

    OUTDIR="results/latent_analysis/seed_${seed}"
    mkdir -p "$OUTDIR"

    run_job "LAT seed_${seed}" \
        python3 scripts/analyze_latent.py \
            --checkpoint "$CKPT" \
            --task configs/task/randomized.yaml \
            --output_dir "$OUTDIR" \
            --num_episodes 50 --seed "${seed}"
done

log ""
log "=== PHASE 3 COMPLETE ==="

log ""
log "================================================================"
log " ALL DONE — Failures: ${FAIL_COUNT}"
log " $(date '+%Y-%m-%d %H:%M:%S')"
log "================================================================"
