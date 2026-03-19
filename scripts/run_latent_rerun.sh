#!/usr/bin/env bash
# Rerun latent analysis only (3 seeds) with fixed build_model
set -uo pipefail
cd /home/chyanin/robotpaper

LOGFILE="logs/latent_rerun_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOGFILE"; }

log "=== Latent Analysis Rerun ==="
FAIL_COUNT=0

for seed in 42 43 44; do
    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}/" \
        -name "best.pt" -type f 2>/dev/null | head -1)
    OUTDIR="results/latent_analysis/seed_${seed}"
    mkdir -p "$OUTDIR"

    log "START: LAT seed_${seed}"
    python3 scripts/analyze_latent.py \
        --checkpoint "$CKPT" \
        --task configs/task/randomized.yaml \
        --output_dir "$OUTDIR" \
        --num_episodes 50 --seed "${seed}" --headless 2>&1 | tee -a "$LOGFILE"

    if [[ -f "$OUTDIR/latent_analysis.json" ]]; then
        log "DONE:  LAT seed_${seed}"
    else
        log "FAIL:  LAT seed_${seed} — no output file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

log "=== ALL DONE — Failures: ${FAIL_COUNT} ==="
