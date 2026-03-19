#!/usr/bin/env bash
# =============================================================================
# Remaining publication experiments: ablations + OOD sweeps + latent analysis
#
# Ablations:  3 variants × 3 seeds = 9, minus 1 completed = 8 training runs
# OOD sweeps: 3 sweeps × 2 models × 3 seeds = 18 eval runs
# Latent:     3 seeds = 3 analysis runs
# Total: 29 jobs, estimated ~3.5h
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="logs/pub_remaining_${TIMESTAMP}.log"
mkdir -p logs/ablations logs/sweeps logs/latent

log() { echo "$1" | tee -a "$LOGFILE"; }

log "================================================================"
log " Remaining Publication Experiments — ${TIMESTAMP}"
log "================================================================"

# ----------------------------------------------------------------
# PHASE 1: Ablation training (8 runs, ~1h 52m)
# ----------------------------------------------------------------
ABLATIONS=(no_aux_loss no_latent single_latent)
SEEDS=(42 43 44)
ABL_COUNT=0
ABL_TOTAL=8

log ""
log "=== PHASE 1: Ablations (${ABL_TOTAL} runs) ==="

for abl in "${ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Skip already-completed no_aux_loss seed=42
        if [[ "$abl" == "no_aux_loss" && "$seed" == "42" ]]; then
            log "  [SKIP] ${abl} seed=${seed} (already completed)"
            continue
        fi

        ABL_COUNT=$((ABL_COUNT + 1))
        log ""
        log "[ABL ${ABL_COUNT}/${ABL_TOTAL}] ${abl} seed=${seed} — $(date '+%H:%M:%S')"

        python3 scripts/train.py \
            --task randomized --model dynamite \
            --ablation "configs/ablations/${abl}.yaml" \
            --variant "${abl}" --seed "${seed}" --headless \
            2>&1 | tee "logs/ablations/${abl}_seed${seed}.log" >> "$LOGFILE"

        log "[ABL ${ABL_COUNT}/${ABL_TOTAL}] DONE — $(date '+%H:%M:%S')"
    done
done

log ""
log "=== PHASE 1 COMPLETE: Ablations ==="

# ----------------------------------------------------------------
# PHASE 2: OOD sweeps (18 runs, ~1h)
# ----------------------------------------------------------------
SWEEPS=(friction push_magnitude action_delay)
MODELS=(dynamite lstm)
SWEEP_SEEDS=(42 43 44)
SW_COUNT=0
SW_TOTAL=18

log ""
log "=== PHASE 2: OOD Sweeps (${SW_TOTAL} runs) ==="

for sweep in "${SWEEPS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SWEEP_SEEDS[@]}"; do
            SW_COUNT=$((SW_COUNT + 1))

            CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" \
                -name "best.pt" -type f 2>/dev/null | head -1)

            if [[ -z "$CKPT" ]]; then
                log "[SW ${SW_COUNT}/${SW_TOTAL}] SKIP — no checkpoint ${model}/seed_${seed}"
                continue
            fi

            OUTDIR="results/sweeps_multiseed/${sweep}/${model}_seed${seed}"
            mkdir -p "$OUTDIR"

            log ""
            log "[SW ${SW_COUNT}/${SW_TOTAL}] ${sweep} ${model} seed=${seed} — $(date '+%H:%M:%S')"

            python3 scripts/eval.py \
                --checkpoint "$CKPT" \
                --task configs/task/randomized.yaml \
                --sweep "configs/sweeps/${sweep}.yaml" \
                --num_episodes 50 --seed "${seed}" \
                --output_dir "$OUTDIR" \
                2>&1 | tee "logs/sweeps/sweep_${sweep}_${model}_seed${seed}.log" >> "$LOGFILE"

            log "[SW ${SW_COUNT}/${SW_TOTAL}] DONE — $(date '+%H:%M:%S')"
        done
    done
done

log ""
log "=== PHASE 2 COMPLETE: OOD Sweeps ==="

# ----------------------------------------------------------------
# PHASE 3: Latent validation (3 runs, ~15m)
# ----------------------------------------------------------------
LAT_COUNT=0
LAT_TOTAL=3

log ""
log "=== PHASE 3: Latent Analysis (${LAT_TOTAL} runs) ==="

for seed in 42 43 44; do
    LAT_COUNT=$((LAT_COUNT + 1))

    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}/" \
        -name "best.pt" -type f 2>/dev/null | head -1)

    if [[ -z "$CKPT" ]]; then
        log "[LAT ${LAT_COUNT}/${LAT_TOTAL}] SKIP — no checkpoint seed_${seed}"
        continue
    fi

    OUTDIR="results/latent_analysis/seed_${seed}"
    mkdir -p "$OUTDIR"

    log ""
    log "[LAT ${LAT_COUNT}/${LAT_TOTAL}] seed=${seed} — $(date '+%H:%M:%S')"

    python3 scripts/analyze_latent.py \
        --checkpoint "$CKPT" \
        --task configs/task/randomized.yaml \
        --output_dir "$OUTDIR" \
        --num_episodes 50 --seed "${seed}" \
        2>&1 | tee "logs/latent/latent_seed${seed}.log" >> "$LOGFILE"

    log "[LAT ${LAT_COUNT}/${LAT_TOTAL}] DONE — $(date '+%H:%M:%S')"
done

log ""
log "=== PHASE 3 COMPLETE: Latent Analysis ==="

log ""
log "================================================================"
log " ALL PHASES COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
log "================================================================"
