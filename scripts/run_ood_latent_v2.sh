#!/usr/bin/env bash
# ====================================================================
# Phase 2: OOD Sweeps (18 runs) + Phase 3: Latent Analysis (3 runs)
# Fixed: --task uses YAML path, build_model device fix
# ====================================================================
set -uo pipefail

cd /home/chyanin/robotpaper

LOGFILE="logs/ood_latent_v2_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p logs results/sweeps_multiseed results/latent_analysis

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOGFILE"; }

log "================================================================"
log " OOD Sweeps + Latent Analysis (v2 — fixed)"
log " RAM: $(free -h | awk '/Mem:/{print $2}'), Swap: $(free -h | awk '/Swap:/{print $2}')"
log "================================================================"

FAIL_COUNT=0

run_job() {
    local label="$1"; shift
    log "START: ${label}"
    "$@" 2>&1 &
    local job_pid=$!
    sudo sh -c "echo -500 > /proc/$job_pid/oom_score_adj" 2>/dev/null || true
    if wait $job_pid; then
        log "DONE:  ${label}"
    else
        log "FAIL:  ${label} (exit code $?)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ----------------------------------------------------------------
# PHASE 2: OOD Sweeps (18 runs: 3 sweep types × 2 models × 3 seeds)
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
                    --output_dir "$OUTDIR" --headless
        done
    done
done

log ""
log "=== PHASE 2 COMPLETE ==="

# ----------------------------------------------------------------
# PHASE 3: Latent Analysis (3 runs)
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
            --num_episodes 50 --seed "${seed}" --headless
done

log ""
log "=== PHASE 3 COMPLETE ==="

log ""
log "================================================================"
log " ALL DONE — Failures: ${FAIL_COUNT}"
log " $(date)"
log "================================================================"
