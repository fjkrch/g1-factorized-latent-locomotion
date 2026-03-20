#!/bin/bash
# Run deterministic 100-episode eval for all 3 ablation variants × 5 seeds
set -e
cd /home/chyanin/robotpaper
PYTHON=/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3
OUTBASE=results/ablation_5seed

declare -A VARIANT_DIRS
VARIANT_DIRS[no_aux_loss]=dynamite_no_aux_no_aux_loss
VARIANT_DIRS[no_latent]=dynamite_no_latent_no_latent
VARIANT_DIRS[single_latent]=dynamite_single_latent_single_latent

SEEDS=(42 43 44 45 46)
COUNT=0
TOTAL=15

for variant in no_aux_loss no_latent single_latent; do
  dir_name=${VARIANT_DIRS[$variant]}
  for seed in "${SEEDS[@]}"; do
    COUNT=$((COUNT + 1))
    outdir="${OUTBASE}/${variant}/seed_${seed}"
    
    # Skip if already done
    if [ -f "${outdir}/eval_metrics.json" ]; then
      echo "[${COUNT}/${TOTAL}] SKIP ${variant} seed ${seed} (already exists)"
      continue
    fi
    
    # Find best.pt
    best=$(find outputs/randomized/${dir_name}/seed_${seed}/ -name "best.pt" 2>/dev/null | head -1)
    if [ -z "$best" ]; then
      echo "[${COUNT}/${TOTAL}] ERROR: No best.pt for ${variant} seed ${seed}"
      continue
    fi
    
    mkdir -p "${outdir}"
    echo "[${COUNT}/${TOTAL}] EVAL ${variant} seed ${seed} → ${outdir}"
    echo "  checkpoint: ${best}"
    
    $PYTHON scripts/eval.py \
      --checkpoint "$best" \
      --num_episodes 100 \
      --seed 42 \
      --output_dir "${outdir}" 2>&1 | tail -3
    
    echo "[${COUNT}/${TOTAL}] DONE ${variant} seed ${seed}"
    echo ""
  done
done

echo "=== ALL ${TOTAL} EVALS COMPLETE ==="
