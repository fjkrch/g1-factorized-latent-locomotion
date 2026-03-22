#!/usr/bin/env bash
# Expand randomized task models to 15 total seeds (n=10 new)
# DynaMITE: 42-51 (10) → add 52-56 (5) → 15 total
# LSTM/Transformer/MLP: 42-46 (5) → add 47-56 (10) → 15 total
set -u
PYTHON="/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3"
BASE_DIR="/home/chyanin/robotpaper"

echo "=========================================="
echo "EXPANSION: 10 new seeds (Phase 1: Training)"
echo "=========================================="

# DynaMITE seeds 52-56
echo "[PHASE 1A] Training DynaMITE seeds 52-56..."
for seed in 52 53 54 55 56; do
  echo "[$(date '+%H:%M:%S')] TRAIN DynaMITE seed $seed (1/15 total)"
  $PYTHON "${BASE_DIR}/scripts/train.py" \
    --base "${BASE_DIR}/configs/base.yaml" \
    --model "${BASE_DIR}/configs/model/dynamite.yaml" \
    --task "${BASE_DIR}/configs/task/randomized.yaml" \
    --train "${BASE_DIR}/configs/train/default.yaml" \
    --seed $seed --headless 2>&1 | tail -3
  echo ""
done

# LSTM seeds 47-56 (10 new)
echo "[PHASE 1B] Training LSTM seeds 47-56..."
for seed in {47..56}; do
  echo "[$(date '+%H:%M:%S')] TRAIN LSTM seed $seed (1/15 total)"
  $PYTHON "${BASE_DIR}/scripts/train.py" \
    --base "${BASE_DIR}/configs/base.yaml" \
    --model "${BASE_DIR}/configs/model/lstm.yaml" \
    --task "${BASE_DIR}/configs/task/randomized.yaml" \
    --train "${BASE_DIR}/configs/train/default.yaml" \
    --seed $seed --headless 2>&1 | tail -3
  echo ""
done

# Transformer seeds 47-56 (10 new)
echo "[PHASE 1C] Training Transformer seeds 47-56..."
for seed in {47..56}; do
  echo "[$(date '+%H:%M:%S')] TRAIN Transformer seed $seed (1/15 total)"
  $PYTHON "${BASE_DIR}/scripts/train.py" \
    --base "${BASE_DIR}/configs/base.yaml" \
    --model "${BASE_DIR}/configs/model/transformer.yaml" \
    --task "${BASE_DIR}/configs/task/randomized.yaml" \
    --train "${BASE_DIR}/configs/train/default.yaml" \
    --seed $seed --headless 2>&1 | tail -3
  echo ""
done

# MLP seeds 47-56 (10 new)
echo "[PHASE 1D] Training MLP seeds 47-56..."
for seed in {47..56}; do
  echo "[$(date '+%H:%M:%S')] TRAIN MLP seed $seed (1/15 total)"
  $PYTHON "${BASE_DIR}/scripts/train.py" \
    --base "${BASE_DIR}/configs/base.yaml" \
    --model "${BASE_DIR}/configs/model/mlp.yaml" \
    --task "${BASE_DIR}/configs/task/randomized.yaml" \
    --train "${BASE_DIR}/configs/train/default.yaml" \
    --seed $seed --headless 2>&1 | tail -3
  echo ""
done

echo ""
echo "=========================================="
echo "PHASE 2: OOD Evaluations"
echo "=========================================="

declare -A CKPT_DIRS=([dynamite]="dynamite_full" [lstm]="lstm_full" [transformer]="transformer_full" [mlp]="mlp_full")
SEEDS_DYNAMITE=(52 53 54 55 56)
SEEDS_OTHERS=(47 48 49 50 51 52 53 54 55 56)
SWEEPS=(combined_shift push_magnitude friction)

# DynaMITE seeds 52-56 OOD evals
echo "[PHASE 2A] OOD evals for DynaMITE seeds 52-56..."
for seed in "${SEEDS_DYNAMITE[@]}"; do
  ckpt=$(find "${BASE_DIR}/outputs/randomized/dynamite_full/seed_${seed}" -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
  if [[ -z "$ckpt" ]]; then
    echo "  [SKIP] DynaMITE seed $seed — no checkpoint"
    continue
  fi
  for sweep in "${SWEEPS[@]}"; do
    out="${BASE_DIR}/results/ood_v2/${sweep}/randomized/dynamite_seed${seed}"
    mkdir -p "$out"
    if [[ -f "${out}/sweep_${sweep}.json" ]]; then
      echo "  [SKIP] DynaMITE seed $seed / $sweep — already done"
      continue
    fi
    echo "  [EVAL] DynaMITE seed $seed / $sweep..."
    $PYTHON "${BASE_DIR}/scripts/eval.py" \
      --checkpoint "$ckpt" \
      --sweep "${BASE_DIR}/configs/sweeps/${sweep}.yaml" \
      --num_episodes 50 --seed 42 --output_dir "$out" --headless 2>&1 | tail -2
  done
done

# LSTM/Transformer/MLP seeds 47-56 OOD evals
for model in lstm transformer mlp; do
  echo "[PHASE 2B] OOD evals for $model seeds 47-56..."
  ckpt_dir="${CKPT_DIRS[$model]}"
  for seed in "${SEEDS_OTHERS[@]}"; do
    ckpt=$(find "${BASE_DIR}/outputs/randomized/${ckpt_dir}/seed_${seed}" -name "best.pt" -path "*/checkpoints/*" 2>/dev/null | head -1)
    if [[ -z "$ckpt" ]]; then
      echo "  [SKIP] $model seed $seed — no checkpoint"
      continue
    fi
    for sweep in "${SWEEPS[@]}"; do
      out="${BASE_DIR}/results/ood_v2/${sweep}/randomized/${model}_seed${seed}"
      mkdir -p "$out"
      if [[ -f "${out}/sweep_${sweep}.json" ]]; then
        echo "  [SKIP] $model seed $seed / $sweep — already done"
        continue
      fi
      echo "  [EVAL] $model seed $seed / $sweep..."
      $PYTHON "${BASE_DIR}/scripts/eval.py" \
        --checkpoint "$ckpt" \
        --sweep "${BASE_DIR}/configs/sweeps/${sweep}.yaml" \
        --num_episodes 50 --seed 42 --output_dir "$out" --headless 2>&1 | tail -2
    done
  done
done

echo ""
echo "=========================================="
echo "EXPANSION COMPLETE"
echo "=========================================="
echo "Run aggregation: python scripts/aggregate_seeds_10seed.py"
