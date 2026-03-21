#!/usr/bin/env bash
# Run combined-shift severe OOD evaluation for 2×2 factorial design
# 4 variants × 10 seeds = 40 sweep runs (levels 3 and 4 only)
set -euo pipefail

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
BASE_DIR="/home/chyanin/robotpaper"
SWEEP_CFG="${BASE_DIR}/configs/sweeps/combined_shift_severe.yaml"
NUM_EPISODES=50
EVAL_SEED=42

# Output root for factorial OOD results
OUT_ROOT="${BASE_DIR}/results/factorial_ood/combined_shift_severe"

# Checkpoint directories
declare -A VARIANT_DIRS
VARIANT_DIRS=(
    ["full"]="outputs/randomized/dynamite_full"
    ["no_aux_loss"]="outputs/randomized/dynamite_no_aux_no_aux_loss"
    ["no_latent"]="outputs/randomized/dynamite_no_latent_no_latent"
    ["aux_only"]="outputs/randomized/dynamite_aux_only_aux_only"
)

SEEDS=(42 43 44 45 46 47 48 49 50 51)

total=$((${#VARIANT_DIRS[@]} * ${#SEEDS[@]}))
count=0
failed=0

echo "============================================"
echo " 2×2 Factorial OOD Evaluation"
echo " Combined-Shift Severe (Levels 3 & 4)"
echo " ${#VARIANT_DIRS[@]} variants × ${#SEEDS[@]} seeds = ${total} runs"
echo " ${NUM_EPISODES} episodes per level"
echo "============================================"

for variant in full no_aux_loss no_latent aux_only; do
    variant_dir="${VARIANT_DIRS[$variant]}"
    for seed in "${SEEDS[@]}"; do
        count=$((count + 1))
        
        # Find the checkpoint (there's a timestamp subdir)
        ckpt=$(find "${BASE_DIR}/${variant_dir}/seed_${seed}" -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
        
        if [[ -z "$ckpt" ]]; then
            echo "[${count}/${total}] SKIP ${variant}/seed_${seed} — no checkpoint found"
            failed=$((failed + 1))
            continue
        fi
        
        out_dir="${OUT_ROOT}/${variant}/seed_${seed}"
        
        # Skip if already completed
        if [[ -f "${out_dir}/sweep_combined_shift_severe.json" ]]; then
            echo "[${count}/${total}] SKIP ${variant}/seed_${seed} — already done"
            continue
        fi
        
        mkdir -p "${out_dir}"
        echo "[${count}/${total}] EVAL ${variant}/seed_${seed} ..."
        
        if ${PYTHON} "${BASE_DIR}/scripts/eval.py" \
            --checkpoint "${ckpt}" \
            --sweep "${SWEEP_CFG}" \
            --num_episodes ${NUM_EPISODES} \
            --seed ${EVAL_SEED} \
            --output_dir "${out_dir}" \
            --headless 2>&1 | tail -5; then
            echo "  -> OK"
        else
            echo "  -> FAILED (exit code $?)"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "============================================"
echo " DONE: $((count - failed))/${total} successful"
if [[ $failed -gt 0 ]]; then
    echo " FAILED: ${failed}"
fi
echo "============================================"
