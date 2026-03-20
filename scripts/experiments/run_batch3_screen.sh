#!/bin/bash
# Batch 3 Screening: 2-seed push_magnitude sweep for A1 and A2
# Protocol v2: 2-seed minimum screening

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
EVAL_SCRIPT="scripts/eval.py"
SWEEP="configs/sweeps/push_magnitude.yaml"
EPISODES=100
BASE_DIR="outputs/randomized"

# Define checkpoints to evaluate
declare -A CHECKPOINTS
CHECKPOINTS["a1_seed42"]="$BASE_DIR/dynamite_aux_1p0/seed_42/20260319_232828/checkpoints/best.pt"
CHECKPOINTS["a1_seed43"]="$BASE_DIR/dynamite_aux_1p0/seed_43/20260320_083917/checkpoints/best.pt"
CHECKPOINTS["a2_seed42"]="$BASE_DIR/dynamite_aux_2p0/seed_42/20260319_234815/checkpoints/best.pt"
CHECKPOINTS["a2_seed43"]="$BASE_DIR/dynamite_aux_2p0/seed_43/20260320_085734/checkpoints/best.pt"

# Output directories
declare -A OUTDIRS
OUTDIRS["a1_seed42"]="results/batch3_screen/a1_aux1p0/seed_42"
OUTDIRS["a1_seed43"]="results/batch3_screen/a1_aux1p0/seed_43"
OUTDIRS["a2_seed42"]="results/batch3_screen/a2_aux2p0/seed_42"
OUTDIRS["a2_seed43"]="results/batch3_screen/a2_aux2p0/seed_43"

cd /home/chyanin/robotpaper

for key in a1_seed42 a1_seed43 a2_seed42 a2_seed43; do
    ckpt="${CHECKPOINTS[$key]}"
    outdir="${OUTDIRS[$key]}"
    
    if [ -f "$outdir/sweep_push_magnitude.json" ]; then
        echo "[SKIP] $key: already evaluated"
        continue
    fi
    
    mkdir -p "$outdir"
    echo "[EVAL] $key: $ckpt -> $outdir"
    $PYTHON $EVAL_SCRIPT \
        --checkpoint "$ckpt" \
        --sweep "$SWEEP" \
        --num_episodes $EPISODES \
        --output_dir "$outdir" \
        --headless
    echo "[DONE] $key"
done

echo "All Batch 3 screen evaluations complete."
