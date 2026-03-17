#!/bin/bash
# Run friction OOD sweep for all 4 main baselines
set -e

PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
SCRIPT="scripts/eval_ood_validated.py"
SWEEP="configs/sweeps/friction.yaml"

declare -A CKPTS
CKPTS[mlp]="outputs/randomized/mlp_full/seed_42/20260316_204510/checkpoints/best.pt"
CKPTS[lstm]="outputs/randomized/lstm_full/seed_42/20260316_204743/checkpoints/best.pt"
CKPTS[transformer]="outputs/randomized/transformer_full/seed_42/20260316_205020/checkpoints/best.pt"

for model in mlp lstm transformer; do
    echo ""
    echo "======================================================================"
    echo "  Running friction sweep for: $model"
    echo "======================================================================"
    $PYTHON $SCRIPT \
        --checkpoint "${CKPTS[$model]}" \
        --sweep "$SWEEP" \
        --num_episodes 50 \
        --num_envs 32 \
        --seed 42
done

echo ""
echo "All model friction sweeps completed."
