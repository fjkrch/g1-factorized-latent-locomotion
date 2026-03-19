#!/usr/bin/env bash
# =============================================================================
# Publication experiments: ALL tiers, sequential
#
# Tier 1 (critical):   no_aux_loss ablation + friction/push sweeps
# Tier 2 (important):  no_latent/single_latent ablations + delay sweep
# Tier 3 (supporting): latent validation
#
# Total: 9 training + 18 sweep evals + 3 latent analyses = 30 jobs
# Estimated time: ~4h total on RTX 4060 Laptop
# =============================================================================
set -euo pipefail

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOGFILE="logs/pub_campaign_${TIMESTAMP}.log"
mkdir -p logs/ablations logs/sweeps logs/latent

echo "================================================================" | tee "$LOGFILE"
echo " Publication Campaign — ${TIMESTAMP}" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"

# ----------------------------------------------------------------
# TIER 1: Critical experiments
# ----------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "=== TIER 1: Critical experiments ===" | tee -a "$LOGFILE"

# 1a. no_aux_loss ablation (3 seeds, ~42 min)
echo "[TIER1] no_aux_loss ablation — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for seed in 42 43 44; do
    echo "  [TIER1] no_aux_loss seed=${seed}" | tee -a "$LOGFILE"
    python3 scripts/train.py \
        --task randomized --model dynamite \
        --ablation configs/ablations/no_aux_loss.yaml \
        --variant no_aux_loss --seed "${seed}" --headless \
        2>&1 | tee "logs/ablations/no_aux_loss_seed${seed}.log" >> "$LOGFILE"
done

# 1b. Friction sweep — DynaMITE + LSTM (6 runs, ~20 min)
echo "[TIER1] Friction sweeps — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for model in dynamite lstm; do
    for seed in 42 43 44; do
        CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" -name "best.pt" | head -1)
        OUTDIR="results/sweeps/friction/${model}_seed${seed}"
        mkdir -p "$OUTDIR"
        echo "  [TIER1] friction ${model} seed=${seed}" | tee -a "$LOGFILE"
        python3 scripts/eval.py \
            --checkpoint "$CKPT" --task configs/task/randomized.yaml \
            --sweep configs/sweeps/friction.yaml \
            --num_episodes 50 --seed "${seed}" --output_dir "$OUTDIR" \
            2>&1 | tee "logs/sweeps/sweep_friction_${model}_seed${seed}.log" >> "$LOGFILE"
    done
done

# 1c. Push magnitude sweep — DynaMITE + LSTM (6 runs, ~20 min)
echo "[TIER1] Push sweeps — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for model in dynamite lstm; do
    for seed in 42 43 44; do
        CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" -name "best.pt" | head -1)
        OUTDIR="results/sweeps/push_magnitude/${model}_seed${seed}"
        mkdir -p "$OUTDIR"
        echo "  [TIER1] push ${model} seed=${seed}" | tee -a "$LOGFILE"
        python3 scripts/eval.py \
            --checkpoint "$CKPT" --task configs/task/randomized.yaml \
            --sweep configs/sweeps/push_magnitude.yaml \
            --num_episodes 50 --seed "${seed}" --output_dir "$OUTDIR" \
            2>&1 | tee "logs/sweeps/sweep_push_${model}_seed${seed}.log" >> "$LOGFILE"
    done
done

echo "[TIER1] COMPLETE — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"

# ----------------------------------------------------------------
# TIER 2: Important experiments
# ----------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "=== TIER 2: Important experiments ===" | tee -a "$LOGFILE"

# 2a. no_latent ablation (3 seeds, ~42 min)
echo "[TIER2] no_latent ablation — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for seed in 42 43 44; do
    echo "  [TIER2] no_latent seed=${seed}" | tee -a "$LOGFILE"
    python3 scripts/train.py \
        --task randomized --model dynamite \
        --ablation configs/ablations/no_latent.yaml \
        --variant no_latent --seed "${seed}" --headless \
        2>&1 | tee "logs/ablations/no_latent_seed${seed}.log" >> "$LOGFILE"
done

# 2b. single_latent ablation (3 seeds, ~42 min)
echo "[TIER2] single_latent ablation — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for seed in 42 43 44; do
    echo "  [TIER2] single_latent seed=${seed}" | tee -a "$LOGFILE"
    python3 scripts/train.py \
        --task randomized --model dynamite \
        --ablation configs/ablations/single_latent.yaml \
        --variant single_latent --seed "${seed}" --headless \
        2>&1 | tee "logs/ablations/single_latent_seed${seed}.log" >> "$LOGFILE"
done

# 2c. Action delay sweep — DynaMITE + LSTM (6 runs, ~20 min)
echo "[TIER2] Action delay sweeps — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for model in dynamite lstm; do
    for seed in 42 43 44; do
        CKPT=$(find "outputs/randomized/${model}_full/seed_${seed}/" -name "best.pt" | head -1)
        OUTDIR="results/sweeps/action_delay/${model}_seed${seed}"
        mkdir -p "$OUTDIR"
        echo "  [TIER2] delay ${model} seed=${seed}" | tee -a "$LOGFILE"
        python3 scripts/eval.py \
            --checkpoint "$CKPT" --task configs/task/randomized.yaml \
            --sweep configs/sweeps/action_delay.yaml \
            --num_episodes 50 --seed "${seed}" --output_dir "$OUTDIR" \
            2>&1 | tee "logs/sweeps/sweep_delay_${model}_seed${seed}.log" >> "$LOGFILE"
    done
done

echo "[TIER2] COMPLETE — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"

# ----------------------------------------------------------------
# TIER 3: Supporting experiments
# ----------------------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "=== TIER 3: Supporting experiments ===" | tee -a "$LOGFILE"

# 3. Latent validation (3 seeds, ~15 min)
echo "[TIER3] Latent validation — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"
for seed in 42 43 44; do
    CKPT=$(find "outputs/randomized/dynamite_full/seed_${seed}/" -name "best.pt" | head -1)
    OUTDIR="results/latent_analysis/seed_${seed}"
    mkdir -p "$OUTDIR"
    echo "  [TIER3] latent seed=${seed}" | tee -a "$LOGFILE"
    python3 scripts/analyze_latent.py \
        --checkpoint "$CKPT" --task configs/task/randomized.yaml \
        --output_dir "$OUTDIR" --num_episodes 50 --seed "${seed}" \
        2>&1 | tee "logs/latent/latent_seed${seed}.log" >> "$LOGFILE"
done

echo "[TIER3] COMPLETE — $(date '+%H:%M:%S')" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
echo " ALL TIERS COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOGFILE"
echo "================================================================" | tee -a "$LOGFILE"
