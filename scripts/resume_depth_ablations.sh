#!/usr/bin/env bash
# Resume depth_1 and depth_4 ablations from ~2M to 12M steps.
# These two failed in the previous ablation pipeline run.
# Runs sequentially with GPU optimizations.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

SEED=42
TASK="randomized"
TOTAL_TIMESTEPS=12000000
NUM_ENVS=256

gpu_cleanup() {
    python -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null || true
}

echo "══════════════════════════════════════════════════════════════"
echo "  Resuming depth_1 and depth_4 ablations to ${TOTAL_TIMESTEPS} steps"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── depth_1 ──
echo ""
echo "━━━ [1/2] depth_1 ━━━"
RUN_DIR_1="outputs/randomized/dynamite_depth1_depth_1/seed_42/20260316_212654"
python scripts/train.py \
    --task configs/task/${TASK}.yaml \
    --model configs/model/dynamite.yaml \
    --ablation configs/ablations/depth_1.yaml \
    --resume "${RUN_DIR_1}" \
    --seed ${SEED} \
    --set train.total_timesteps=${TOTAL_TIMESTEPS} task.num_envs=${NUM_ENVS} train.save_interval=100
echo "  ✓ depth_1 DONE"

gpu_cleanup
sleep 5

# ── depth_4 ──
echo ""
echo "━━━ [2/2] depth_4 ━━━"
RUN_DIR_4="outputs/randomized/dynamite_depth4_depth_4/seed_42/20260316_213007"
python scripts/train.py \
    --task configs/task/${TASK}.yaml \
    --model configs/model/dynamite.yaml \
    --ablation configs/ablations/depth_4.yaml \
    --resume "${RUN_DIR_4}" \
    --seed ${SEED} \
    --set train.total_timesteps=${TOTAL_TIMESTEPS} task.num_envs=${NUM_ENVS} train.save_interval=100
echo "  ✓ depth_4 DONE"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Both ablations complete. End: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
