#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# run_ablation_10seed.sh — Train 5 new ablation seeds (47-51) + baselines
# ══════════════════════════════════════════════════════════════════════════════
#
# Trains:
#   - DynaMITE-full baseline × seeds 47-51 (5 runs)
#   - 3 ablation variants × seeds 47-51 (15 runs)
# Then evals all 20 new runs (100 episodes each)
#
# Total: 20 training runs (~4.7 hours) + 20 evals (~1 hour)
#
# Usage:
#   bash scripts/experiments/run_ablation_10seed.sh
#   bash scripts/experiments/run_ablation_10seed.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd /home/chyanin/robotpaper

PYTHON=/home/chyanin/miniconda3/envs/env_isaaclab/bin/python3
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

NEW_SEEDS=(47 48 49 50 51)
TASK="randomized"
MODEL="dynamite"

# Ablation variants + full baseline
ABLATIONS=("" "no_aux_loss" "no_latent" "single_latent")
VARIANTS=("full" "no_aux_loss" "no_latent" "single_latent")

TOTAL=$((${#ABLATIONS[@]} * ${#NEW_SEEDS[@]}))
CURRENT=0
FAILED=0

echo "══════════════════════════════════════════════════════════════"
echo "  ABLATION 10-SEED EXPANSION"
echo "  ${#ABLATIONS[@]} configs × ${#NEW_SEEDS[@]} new seeds = $TOTAL training runs"
echo "  Start: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Phase 1: Training ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 1: TRAINING ($TOTAL runs)    ║"
echo "╚══════════════════════════════════════╝"

for i in "${!ABLATIONS[@]}"; do
    abl="${ABLATIONS[$i]}"
    variant="${VARIANTS[$i]}"
    
    for seed in "${NEW_SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        if [[ -z "$abl" ]]; then
            LABEL="${TASK}/${MODEL}_${variant}/seed_${seed} (baseline)"
        else
            LABEL="${TASK}/${MODEL}_${variant}/seed_${seed} (ablation: $abl)"
        fi
        
        echo ""
        echo "[$CURRENT/$TOTAL] TRAINING: $LABEL"
        
        # Build command
        CMD="bash scripts/run_train.sh --task $TASK --model $MODEL --seed $seed --variant $variant"
        if [[ -n "$abl" ]]; then
            CMD="$CMD --ablation $abl"
        fi
        
        if $DRY_RUN; then
            echo "  [DRY RUN] $CMD"
            continue
        fi
        
        $CMD || {
            echo "  ⚠ FAILED: $LABEL"
            FAILED=$((FAILED + 1))
        }
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE"
echo "  Total: $TOTAL  |  Failed: $FAILED"
echo "  Time: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

if [[ $FAILED -gt 0 ]]; then
    echo "WARNING: $FAILED training runs failed"
fi

# ── Phase 2: Evaluation ──
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  PHASE 2: EVALUATION ($TOTAL evals) ║"
echo "╚══════════════════════════════════════╝"

# Map variant -> directory name (model.name from ablation config + variant)
declare -A DIR_MAP
DIR_MAP[full]="dynamite_full"
DIR_MAP[no_aux_loss]="dynamite_no_aux_no_aux_loss"
DIR_MAP[no_latent]="dynamite_no_latent_no_latent"
DIR_MAP[single_latent]="dynamite_single_latent_single_latent"

EVAL_CURRENT=0
EVAL_FAILED=0
OUTBASE="results/ablation_10seed"

for variant in full no_aux_loss no_latent single_latent; do
    dir_name="${DIR_MAP[$variant]}"
    
    for seed in "${NEW_SEEDS[@]}"; do
        EVAL_CURRENT=$((EVAL_CURRENT + 1))
        outdir="${OUTBASE}/${variant}/seed_${seed}"
        
        # Find best.pt
        best=$(find "outputs/randomized/${dir_name}/seed_${seed}/" -name "best.pt" 2>/dev/null | head -1)
        if [[ -z "$best" ]]; then
            echo "[$EVAL_CURRENT/$TOTAL] ERROR: No best.pt for ${variant} seed ${seed} (dir: ${dir_name})"
            EVAL_FAILED=$((EVAL_FAILED + 1))
            continue
        fi
        
        # Skip if already done
        if [[ -f "${outdir}/eval_metrics.json" ]]; then
            echo "[$EVAL_CURRENT/$TOTAL] SKIP ${variant} seed ${seed} (already exists)"
            continue
        fi
        
        mkdir -p "${outdir}"
        echo "[$EVAL_CURRENT/$TOTAL] EVAL ${variant} seed ${seed}"
        echo "  checkpoint: ${best}"
        
        if $DRY_RUN; then
            echo "  [DRY RUN] $PYTHON scripts/eval.py --checkpoint $best --num_episodes 100 --seed 42 --output_dir $outdir"
            continue
        fi
        
        $PYTHON scripts/eval.py \
            --checkpoint "$best" \
            --num_episodes 100 \
            --seed 42 \
            --output_dir "${outdir}" 2>&1 | tail -3 || {
            echo "  ⚠ EVAL FAILED: ${variant} seed ${seed}"
            EVAL_FAILED=$((EVAL_FAILED + 1))
        }
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  EVALUATION COMPLETE"
echo "  Total: $TOTAL  |  Failed: $EVAL_FAILED"
echo "  Time: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"

# ── Phase 3: Also eval OLD seeds (42-46) into ablation_10seed dir ──
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: EVAL OLD SEEDS (42-46) into 10seed dir       ║"
echo "╚══════════════════════════════════════════════════════════╝"

OLD_SEEDS=(42 43 44 45 46)
OLD_TOTAL=$((4 * ${#OLD_SEEDS[@]}))
OLD_CURRENT=0

for variant in full no_aux_loss no_latent single_latent; do
    dir_name="${DIR_MAP[$variant]}"
    
    for seed in "${OLD_SEEDS[@]}"; do
        OLD_CURRENT=$((OLD_CURRENT + 1))
        outdir="${OUTBASE}/${variant}/seed_${seed}"
        
        if [[ -f "${outdir}/eval_metrics.json" ]]; then
            echo "[$OLD_CURRENT/$OLD_TOTAL] SKIP ${variant} seed ${seed} (already exists)"
            continue
        fi
        
        best=$(find "outputs/randomized/${dir_name}/seed_${seed}/" -name "best.pt" 2>/dev/null | head -1)
        if [[ -z "$best" ]]; then
            echo "[$OLD_CURRENT/$OLD_TOTAL] ERROR: No best.pt for ${variant} seed ${seed}"
            continue
        fi
        
        mkdir -p "${outdir}"
        echo "[$OLD_CURRENT/$OLD_TOTAL] EVAL ${variant} seed ${seed}"
        
        if $DRY_RUN; then
            echo "  [DRY RUN] $PYTHON scripts/eval.py --checkpoint $best --num_episodes 100 --seed 42 --output_dir $outdir"
            continue
        fi
        
        $PYTHON scripts/eval.py \
            --checkpoint "$best" \
            --num_episodes 100 \
            --seed 42 \
            --output_dir "${outdir}" 2>&1 | tail -3 || true
    done
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL PHASES COMPLETE"
echo "  Time: $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════"
