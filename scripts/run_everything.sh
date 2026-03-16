#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_everything.sh — Run the FULL experiment pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════
#
# Pipeline:
#   1. Sanity checks (env, GPU, disk)
#   2. Main comparison: 4 models × 4 tasks × 1 seed  =  16 training runs  (~96 min)
#   3. Ablation study:  7 ablations × 1 seed          =   7 training runs  (~42 min)
#   4. Evaluation of all trained runs                                       (~15 min)
#   5. Robustness sweeps: 4 models × 4 sweeps         =  16 eval runs      (~10 min)
#   6. Aggregation, tables & plots                                          (~2 min)
#
# Estimated total runtime: ~2.5–3 hours on RTX 4060 (2M steps / 512 envs per run)
#
# Usage:
#   bash scripts/run_everything.sh            # run everything
#   bash scripts/run_everything.sh --dry-run  # print commands, don't execute
#
# Prerequisites:
#   - conda env "env_isaaclab" with Isaac Lab + this project installed
#   - NVIDIA GPU with ≥8 GB VRAM
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# ── Helpers ──
section() { echo -e "\n\n══════════════════════════════════════════════════════════════"; echo "  $1"; echo "  $(date -Iseconds)"; echo "══════════════════════════════════════════════════════════════"; }
run()     { if $DRY_RUN; then echo "[DRY RUN] $*"; else echo "[RUN] $*"; "$@"; fi; }
elapsed() { local t=$SECONDS; printf '%dh %dm %ds' $((t/3600)) $((t%3600/60)) $((t%60)); }

SECONDS=0
FAILED=0

section "0/6  SANITY CHECKS"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python: $(python --version 2>&1)"
echo "  Disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "  Project root: $PROJECT_ROOT"

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "${DYNAMITE_CONDA_ENV:-env_isaaclab}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Main Comparison Training (16 runs)
# ══════════════════════════════════════════════════════════════════════════════
section "1/6  MAIN COMPARISON — 4 models × 4 tasks × 1 seed = 16 runs"

TASKS=(flat push randomized terrain)
MODELS=(mlp lstm transformer dynamite)
SEED=42
TOTAL_MAIN=$((${#TASKS[@]} * ${#MODELS[@]}))
CURRENT=0

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[$CURRENT/$TOTAL_MAIN] Training: ${model} on ${task}  (elapsed: $(elapsed))"
        run bash scripts/run_train.sh --task "$task" --model "$model" --seed "$SEED" --variant full || {
            echo "  ⚠ FAILED: ${model}/${task}"
            FAILED=$((FAILED + 1))
        }
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Ablation Study (7 runs)
# ══════════════════════════════════════════════════════════════════════════════
section "2/6  ABLATIONS — 7 variants on randomized task"

ABLATIONS=(seq_len_4 seq_len_16 no_latent single_latent no_aux_loss depth_1 depth_4)
TOTAL_ABL=${#ABLATIONS[@]}
CURRENT=0

for abl in "${ABLATIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_ABL] Ablation: ${abl}  (elapsed: $(elapsed))"
    run python scripts/train.py \
        --task configs/task/randomized.yaml \
        --model configs/model/dynamite.yaml \
        --ablation "configs/ablations/${abl}.yaml" \
        --seed "$SEED" \
        --variant "$abl" \
        --headless || {
        echo "  ⚠ FAILED: ablation ${abl}"
        FAILED=$((FAILED + 1))
    }
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Evaluation of ALL trained runs
# ══════════════════════════════════════════════════════════════════════════════
section "3/6  EVALUATION — all completed runs"

EVAL_COUNT=0
for run_dir in $(find outputs/ -mindepth 4 -maxdepth 4 -type d 2>/dev/null); do
    # Skip if already evaluated
    if [[ -f "$run_dir/eval_metrics.json" ]]; then
        echo "  SKIP (already evaluated): $run_dir"
        continue
    fi
    CKPT="$run_dir/checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="$run_dir/checkpoints/latest.pt"
    fi
    if [[ -f "$CKPT" ]]; then
        EVAL_COUNT=$((EVAL_COUNT + 1))
        echo "  [$EVAL_COUNT] Evaluating: $run_dir"
        run python scripts/eval.py --run_dir "$run_dir" --num_episodes 50 || {
            echo "  ⚠ eval failed: $run_dir"
            FAILED=$((FAILED + 1))
        }
    fi
done
echo "  Evaluated: $EVAL_COUNT runs"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Robustness Sweeps (eval only)
# ══════════════════════════════════════════════════════════════════════════════
section "4/6  ROBUSTNESS SWEEPS — 4 models × 4 perturbation types"

SWEEPS=(push_magnitude friction motor_strength action_delay)
SWEEP_TASK="randomized"

for model in "${MODELS[@]}"; do
    CKPT_DIR="outputs/${SWEEP_TASK}/${model}_full/seed_${SEED}"
    LATEST_RUN=$(ls -td "${CKPT_DIR}"/*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "  WARNING: No run found for ${model} on ${SWEEP_TASK}. Skipping sweeps."
        continue
    fi
    CKPT="${LATEST_RUN}checkpoints/best.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT="${LATEST_RUN}checkpoints/latest.pt"
    fi
    for sweep in "${SWEEPS[@]}"; do
        echo "  Sweep: ${model} / ${sweep}"
        run python scripts/eval.py \
            --checkpoint "$CKPT" \
            --sweep "configs/sweeps/${sweep}.yaml" \
            --output_dir "results/sweeps/${model}/${sweep}" \
            --num_episodes 20 || {
            echo "  ⚠ sweep failed: ${model}/${sweep}"
            FAILED=$((FAILED + 1))
        }
    done
done

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Aggregation
# ══════════════════════════════════════════════════════════════════════════════
section "5/6  AGGREGATION — aggregate seeds, generate tables"
run bash scripts/aggregate_all.sh --skip-eval || {
    echo "  ⚠ Aggregation had errors (non-fatal)"
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Plots & Tables
# ══════════════════════════════════════════════════════════════════════════════
section "6/6  PLOTS & TABLES"
mkdir -p figures/

# Generate summary files for plotting from per-experiment aggregated JSONs
python3 << 'PYEOF'
import json, csv
from pathlib import Path
import numpy as np

tasks = ["flat", "push", "randomized", "terrain"]
models = ["mlp", "lstm", "transformer", "dynamite"]
agg_dir = Path("results/aggregated")
agg_dir.mkdir(parents=True, exist_ok=True)

# 1. main_comparison.json: {task -> {method -> {mean, std}}}
mc = {}
for task in tasks:
    mc[task] = {}
    for model in models:
        jpath = agg_dir / f"{task}_{model}_full.json"
        if jpath.exists():
            d = json.load(open(jpath))
            rm = d.get("episode_reward/mean", d.get("reward_mean", {}))
            mc[task][model] = {
                "mean": rm.get("mean", 0) if isinstance(rm, dict) else (rm if isinstance(rm, (int,float)) else 0),
                "std": rm.get("std", 0) if isinstance(rm, dict) else 0,
            }
json.dump(mc, open(agg_dir / "main_comparison.json", "w"), indent=2)

# 2. training_curves.json: {method -> {steps, values}}
tc = {}
for model in models:
    runs = sorted([d for d in Path(f"outputs/flat/{model}_full/seed_42").iterdir() if d.is_dir()]) if Path(f"outputs/flat/{model}_full/seed_42").exists() else []
    if not runs: continue
    csv_path = runs[-1] / "metrics.csv"
    if not csv_path.exists(): continue
    steps, vals = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gs = row.get("global_step","") or row.get("step","")
            rm = row.get("reward/mean","") or row.get("reward_mean","")
            if gs and rm:
                try:
                    s, v = int(float(gs)), float(rm)
                    if not np.isnan(v): steps.append(s); vals.append(v)
                except: pass
    if steps: tc[model] = {"steps": steps, "values": vals}
json.dump(tc, open(agg_dir / "training_curves.json", "w"), indent=2)

# 3. ablation_results.json: {name -> {mean, std}}
ablations = {"seq_len_4":"Seq Len 4","seq_len_16":"Seq Len 16","no_latent":"No Latent",
             "single_latent":"Single Latent","no_aux_loss":"No Aux Loss","depth_1":"Depth 1","depth_4":"Depth 4"}
ar = {}
full = agg_dir / "randomized_dynamite_full.json"
if full.exists():
    d = json.load(open(full)); rm = d.get("episode_reward/mean", d.get("reward_mean", {}))
    ar["DynaMITE (Full)"] = {"mean": rm.get("mean",0) if isinstance(rm,dict) else rm, "std": rm.get("std",0) if isinstance(rm,dict) else 0}
for k, label in ablations.items():
    p = agg_dir / f"ablation_{k}.json"
    if p.exists():
        d = json.load(open(p)); rm = d.get("episode_reward/mean", d.get("reward_mean", {}))
        ar[label] = {"mean": rm.get("mean",0) if isinstance(rm,dict) else rm, "std": rm.get("std",0) if isinstance(rm,dict) else 0}
json.dump(ar, open(agg_dir / "ablation_results.json", "w"), indent=2)
print(f"Summary files created: main_comparison ({len(mc)} tasks), training_curves ({len(tc)} methods), ablation ({len(ar)} variants)")
PYEOF

run python scripts/plot_results.py --results_dir results/aggregated --output_dir figures/ || {
    echo "  ⚠ Plotting failed (non-fatal)"
}
run python scripts/generate_tables.py --results_dir results/aggregated --output_dir figures/ || {
    echo "  ⚠ Table generation failed (non-fatal)"
}

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section "COMPLETE"
echo "  Total elapsed: $(elapsed)"
echo "  Training runs: $((TOTAL_MAIN + TOTAL_ABL))  (main: $TOTAL_MAIN, ablation: $TOTAL_ABL)"
echo "  Evaluations:   $EVAL_COUNT"
echo "  Failed:        $FAILED"
echo ""
echo "  Outputs:       outputs/"
echo "  Results:       results/"
echo "  Figures:       figures/"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "  ⚠ $FAILED step(s) failed. Check logs above."
    exit 1
else
    echo "  ✓ All steps completed successfully!"
fi
