# DynaMITE — Experiment Operations Protocol

> **Purpose**: This document is the single authoritative reference for running, recording, evaluating, aggregating, and reproducing every experiment in the DynaMITE project. It covers 15 operational phases end-to-end.
>
> **Hardware assumption**: Single RTX 4060 laptop (8 GB VRAM). All timing estimates use this baseline.
>
> **Codebase root**: `/home/chyanin/robotpaper/`

---

## Table of Contents

1. [Phase 1 — Project Interface](#phase-1--project-interface)
2. [Phase 2 — Naming System](#phase-2--naming-system)
3. [Phase 3 — Directory Structure](#phase-3--directory-structure)
4. [Phase 4 — Training Protocol](#phase-4--training-protocol)
5. [Phase 5 — Evaluation Protocol](#phase-5--evaluation-protocol)
6. [Phase 6 — Manifest System](#phase-6--manifest-system)
7. [Phase 7 — Metrics Recording Standard](#phase-7--metrics-recording-standard)
8. [Phase 8 — Result Aggregation](#phase-8--result-aggregation)
9. [Phase 9 — Figure & Table Regeneration](#phase-9--figure--table-regeneration)
10. [Phase 10 — Master Run Plan](#phase-10--master-run-plan)
11. [Phase 11 — Failure & Recovery Protocol](#phase-11--failure--recovery-protocol)
12. [Phase 12 — Reproducibility Checklist](#phase-12--reproducibility-checklist)
13. [Phase 13 — README: Running Experiments](#phase-13--readme-running-experiments)
14. [Phase 14 — Files Inventory](#phase-14--files-inventory)
15. [Phase 15 — Final Audit](#phase-15--final-audit)

---

## Phase 1 — Project Interface

### 1.1 Training Entrypoint

**Script**: `scripts/train.py`

**CLI Signature**:
```bash
python scripts/train.py \
    --base   configs/base.yaml \
    --task   configs/task/{flat,push,randomized,terrain}.yaml \
    --model  configs/model/{mlp,lstm,transformer,dynamite}.yaml \
    --train  configs/train/default.yaml \
    [--ablation configs/ablations/{name}.yaml] \
    [--resume  outputs/{task}/{model}_{variant}/seed_{seed}/{ts}] \
    [--seed 42] \
    [--set key=value ...]
```

**Contracts**:
- `--task` and `--model` are **required**.
- `--base` defaults to `configs/base.yaml`. Never omit it intentionally.
- `--ablation` is optional; when provided, its YAML is deep-merged AFTER model config and BEFORE CLI overrides.
- `--set` accepts dotted-key overrides: `train.total_timesteps=10000000`.
- `--seed` overrides `seed` in the merged config.
- `--resume` points to an existing run directory; the script looks for `latest.pt` inside `checkpoints/`.

**Config merge order** (later wins):
```
base.yaml → task/{task}.yaml → model/{model}.yaml → train/default.yaml → ablations/{abl}.yaml → --set CLI
```

**Outputs** (written to `run_dir`):
| File | Description |
|---|---|
| `config.yaml` | Saved merged config (frozen for this run) |
| `manifest.json` | Experiment manifest (see Phase 6) |
| `metrics.csv` | Step-level training metrics |
| `tb/` | TensorBoard event files |
| `checkpoints/ckpt_{step}.pt` | Periodic checkpoints |
| `checkpoints/latest.pt` | Most recent checkpoint |
| `checkpoints/best.pt` | Best by eval reward |

### 1.2 Evaluation Entrypoint

**Script**: `scripts/eval.py`

**CLI Signature**:
```bash
python scripts/eval.py \
    --checkpoint outputs/.../checkpoints/best.pt \
    [--run_dir outputs/.../] \
    [--task configs/task/{name}.yaml] \
    [--eval_config configs/eval/default.yaml] \
    [--output_dir results/eval/] \
    [--sweep configs/sweeps/{name}.yaml] \
    [--num_episodes 100] \
    [--seed 42]
```

**Contracts**:
- Either `--checkpoint` or `--run_dir` must be given.
- When `--run_dir` is given, eval uses `best.pt` from that run's checkpoints.
- `--sweep` triggers robustness sweep mode (evaluates across a range of domain parameters).
- Standard eval outputs `eval_metrics.json` in the run's output dir.
- Sweep eval outputs `sweep_{name}.json`.

### 1.3 Config Files

| Config | Path | Purpose |
|---|---|---|
| Base | `configs/base.yaml` | All defaults: env, reward, training, eval, output |
| Tasks | `configs/task/{flat,push,randomized,terrain}.yaml` | Per-task overrides (DR, terrain, pushes) |
| Models | `configs/model/{mlp,lstm,transformer,dynamite}.yaml` | Architecture hyperparams |
| Training | `configs/train/default.yaml` | PPO hyperparams (can override base) |
| Evaluation | `configs/eval/default.yaml` | Eval settings |
| Ablations | `configs/ablations/{7 files}.yaml` | Ablation overlays |
| Sweeps | `configs/sweeps/{4 files}.yaml` | Robustness sweep definitions |

**Available ablation configs**:
```
seq_len_4.yaml       – history_len=4
seq_len_16.yaml      – history_len=16
no_latent.yaml       – disable latent head
single_latent.yaml   – unfactored single latent
no_aux_loss.yaml     – zero auxiliary loss weight
depth_1.yaml         – 1 transformer layer
depth_4.yaml         – 4 transformer layers
```

**Available sweep configs**:
```
push_magnitude.yaml  – sweep push velocity [0, 2, 4, 6, 8, 10]
friction.yaml        – sweep ground friction [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]
motor_strength.yaml  – sweep motor multiplier [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
action_delay.yaml    – sweep delay steps [0, 1, 2, 3, 4]
```

---

## Phase 2 — Naming System

All naming is handled by `src/utils/run_naming.py`.

### 2.1 Run ID

```
Format:  {YYYYMMDD}_{HHMMSS}_{project}_{task}_{model}_{variant}_seed{N}
Example: 20260316_143022_dynamite_randomized_dynamite_full_seed42
```

**Fields**:
| Field | Source | Examples |
|---|---|---|
| `YYYYMMDD_HHMMSS` | Wall-clock at run start | `20260316_143022` |
| `project` | `cfg.project.name` | `dynamite` |
| `task` | `cfg.task.name` | `flat`, `push`, `randomized`, `terrain` |
| `model` | `cfg.model.name` | `mlp`, `lstm`, `transformer`, `dynamite` |
| `variant` | Ablation tag or `full` | `full`, `no_latent`, `seq_len_4` |
| `seed` | `cfg.seed` | `42`, `123`, `456` |

### 2.2 Group ID

A group collects all seed runs for the same experiment:
```
Format:  {project}_{task}_{model}_{variant}
Example: dynamite_randomized_dynamite_full
```

### 2.3 Run Directory Path

```
Format:  {base_dir}/{task}/{model}_{variant}/seed_{seed}/{YYYYMMDD_HHMMSS}/
Example: outputs/randomized/dynamite_full/seed_42/20260316_143022/
```

### 2.4 Checkpoint Names

```
ckpt_{step:010d}.pt     →  ckpt_0005000000.pt
latest.pt               →  always symlinked/copied to most recent
best.pt                 →  best by eval reward
```

### 2.5 Eval / Sweep Result Names

```
eval_{task}_step{N}.json    →  eval_randomized_stepbest.json
sweep_{name}.json           →  sweep_push_magnitude.json
```

### 2.6 Figure / Table Names

```
fig_{type}.{fmt}            →  fig_learning_curves.pdf
table_{type}.{fmt}          →  table_main_comparison.tex
```

### 2.7 API Usage

```python
from src.utils.run_naming import (
    make_run_id, make_group_id, parse_run_id,
    make_run_dir, checkpoint_name,
    eval_result_name, sweep_result_name,
    figure_name, table_name,
)

cfg = {...}  # merged config
rid = make_run_id(cfg, variant="no_latent")
# → "20260316_143022_dynamite_randomized_dynamite_no_latent_seed42"

parts = parse_run_id(rid)
# → {"date": "20260316", "time": "143022", "project": "dynamite",
#     "task": "randomized", "model": "dynamite", "variant": "no_latent", "seed": 42}
```

---

## Phase 3 — Directory Structure

### 3.1 Single Run Directory

```
outputs/randomized/dynamite_full/seed_42/20260316_143022/
├── config.yaml                 # Frozen merged config
├── manifest.json               # Experiment manifest (Phase 6)
├── metrics.csv                 # Step-level CSV (Phase 7)
├── tb/                         # TensorBoard logs
│   └── events.out.tfevents.*
├── checkpoints/
│   ├── ckpt_0005000000.pt
│   ├── ckpt_0010000000.pt
│   ├── ...
│   ├── latest.pt
│   └── best.pt
├── eval_metrics.json           # Standard evaluation results
├── sweep_push_magnitude.json   # Robustness sweep results (optional)
├── sweep_friction.json
└── stdout.log                  # Captured stdout (from shell scripts)
```

### 3.2 Group Directory

A group = all seeds for one (task, model, variant) combination:
```
outputs/randomized/dynamite_full/
├── seed_42/
│   └── 20260316_143022/        # Run directory
├── seed_123/
│   └── 20260316_150000/
└── seed_456/
    └── 20260316_160000/
```

### 3.3 Full Project Output Tree

```
outputs/
├── flat/
│   ├── mlp_full/
│   │   ├── seed_42/…
│   │   ├── seed_123/…
│   │   └── seed_456/…
│   ├── lstm_full/
│   ├── transformer_full/
│   └── dynamite_full/
├── push/
│   ├── mlp_full/
│   ├── lstm_full/
│   ├── transformer_full/
│   └── dynamite_full/
├── randomized/
│   ├── mlp_full/
│   ├── lstm_full/
│   ├── transformer_full/
│   ├── dynamite_full/
│   ├── dynamite_seq_len_4/
│   ├── dynamite_seq_len_16/
│   ├── dynamite_no_latent/
│   ├── dynamite_single_latent/
│   ├── dynamite_no_aux_loss/
│   ├── dynamite_depth_1/
│   └── dynamite_depth_4/
├── terrain/
│   ├── mlp_full/
│   ├── lstm_full/
│   ├── transformer_full/
│   └── dynamite_full/
```

### 3.4 Results / Figures / Tables Tree

```
results/
├── aggregated/
│   ├── main_comparison.json
│   ├── ablation_results.json
│   └── main_comparison.csv
├── sweeps/                     # Collected sweep JSONs
│   ├── mlp_push_magnitude.json
│   └── ...
└── tables/
    ├── table_main_comparison.md
    ├── table_main_comparison.tex
    ├── table_ablation.md
    ├── table_ablation.tex
    └── table_efficiency.md

figures/
├── fig_learning_curves_flat.pdf
├── fig_learning_curves_push.pdf
├── fig_learning_curves_randomized.pdf
├── fig_learning_curves_terrain.pdf
├── fig_sweep_push_magnitude.pdf
├── fig_sweep_friction.pdf
├── fig_sweep_motor_strength.pdf
├── fig_sweep_action_delay.pdf
└── fig_ablation_bars.pdf
```

---

## Phase 4 — Training Protocol

### 4.1 Single Training Run

```bash
# Using the shell wrapper (recommended):
bash scripts/run_train.sh \
    --task flat \
    --model mlp \
    --seed 42

# Directly:
python scripts/train.py \
    --task configs/task/flat.yaml \
    --model configs/model/mlp.yaml \
    --seed 42
```

The shell wrapper adds:
- stdout capture to `stdout.log`
- GPU info logging
- exit code tracking
- `--dry-run` mode for verification

### 4.2 Main Comparison: All 48 Runs

**What**: 4 models × 4 tasks × 3 seeds = 48 runs.

```bash
bash scripts/run_all_main.sh [--skip-existing] [--dry-run]
```

**Exact run matrix**:
| | flat | push | randomized | terrain |
|---|---|---|---|---|
| MLP | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 |
| LSTM | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 |
| Transformer | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 |
| DynaMITE | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 |

**Time estimate**: ~48 × 3h = **144 hours** (6 days, sequential, RTX 4060).

**Each run produces**:
```
outputs/{task}/{model}_full/seed_{seed}/{timestamp}/
├── config.yaml
├── manifest.json
├── metrics.csv
├── tb/
├── checkpoints/{ckpt_*.pt, latest.pt, best.pt}
└── stdout.log
```

### 4.3 Ablation Runs: 21 Runs

**What**: 7 ablations × 3 seeds, all on task=randomized, model=dynamite.

```bash
bash scripts/run_ablations_v2.sh [--skip-existing] [--dry-run]
```

**Ablation matrix**:
| Ablation | variant tag | Description |
|---|---|---|
| seq_len_4 | `seq_len_4` | History window = 4 |
| seq_len_16 | `seq_len_16` | History window = 16 |
| no_latent | `no_latent` | Disable latent dynamics head |
| single_latent | `single_latent` | Unfactored single latent |
| no_aux_loss | `no_aux_loss` | Zero auxiliary identification loss |
| depth_1 | `depth_1` | 1 transformer encoder layer |
| depth_4 | `depth_4` | 4 transformer encoder layers |

**Time estimate**: ~21 × 3h = **63 hours** (2.6 days).

**Note**: The full DynaMITE on randomized (seeds 42/123/456) is already in the main comparison matrix. The ablation table references those as the "full" baseline.

### 4.4 Training with CLI Overrides

```bash
# Quick debug run (10k steps, fewer envs)
python scripts/train.py \
    --task configs/task/flat.yaml \
    --model configs/model/mlp.yaml \
    --seed 42 \
    --set train.total_timesteps=10000 task.num_envs=64

# Custom learning rate
python scripts/train.py \
    --task configs/task/randomized.yaml \
    --model configs/model/dynamite.yaml \
    --set train.learning_rate=1e-4
```

### 4.5 Resuming a Crashed Run

```bash
python scripts/train.py \
    --task configs/task/randomized.yaml \
    --model configs/model/dynamite.yaml \
    --seed 42 \
    --resume outputs/randomized/dynamite_full/seed_42/20260316_143022
```

The script loads `checkpoints/latest.pt` and continues from `global_step` stored in the checkpoint.

---

## Phase 5 — Evaluation Protocol

### 5.1 Standard Evaluation (Single Run)

```bash
python scripts/eval.py \
    --run_dir outputs/randomized/dynamite_full/seed_42/20260316_143022/ \
    --num_episodes 100 \
    --seed 42
```

**Output**: `eval_metrics.json` in the run directory.

### 5.2 Evaluation with Explicit Checkpoint

```bash
python scripts/eval.py \
    --checkpoint outputs/randomized/dynamite_full/seed_42/20260316_143022/checkpoints/best.pt \
    --task configs/task/randomized.yaml \
    --num_episodes 100
```

### 5.3 Batch Evaluation (All Runs)

```bash
bash scripts/run_eval.sh --run-dir outputs/ --num-episodes 100 --seed 42
```

Or manually loop:
```bash
for dir in outputs/*/*/seed_*/*/; do
    if [ -f "$dir/checkpoints/best.pt" ] && [ ! -f "$dir/eval_metrics.json" ]; then
        python scripts/eval.py --run_dir "$dir" --num_episodes 100 --seed 42
    fi
done
```

### 5.4 Robustness Sweep Evaluation

Sweeps evaluate a trained model across a range of one domain parameter.

```bash
# Single sweep
python scripts/eval.py \
    --checkpoint outputs/randomized/dynamite_full/seed_42/.../checkpoints/best.pt \
    --task configs/task/randomized.yaml \
    --sweep configs/sweeps/push_magnitude.yaml \
    --num_episodes 100

# All 4 sweeps for all 4 methods (16 evaluations):
bash scripts/run_robustness.sh [--dry-run]
```

**Output**: `sweep_{name}.json` in the run directory or `results/sweeps/`.

### 5.5 Cross-Task Evaluation

To test generalization (not required for the paper, but supported):
```bash
# Train on randomized, evaluate on push
python scripts/eval.py \
    --run_dir outputs/randomized/dynamite_full/seed_42/.../ \
    --task configs/task/push.yaml \
    --output_dir results/cross_task/
```

### 5.6 Eval Output Format

**`eval_metrics.json`**:
```json
{
  "checkpoint": "outputs/randomized/dynamite_full/seed_42/.../checkpoints/best.pt",
  "task": "randomized",
  "model": "dynamite",
  "seed": 42,
  "num_episodes": 100,
  "episode_reward": {
    "mean": 285.3,
    "std": 42.1,
    "min": 120.5,
    "max": 380.2
  },
  "episode_length": {
    "mean": 950.2,
    "std": 85.3
  },
  "success_rate": 0.92,
  "fall_rate": 0.03,
  "lin_vel_tracking_error": {
    "mean": 0.12,
    "std": 0.04
  },
  "wall_time_s": 45.2
}
```

**`sweep_push_magnitude.json`**:
```json
{
  "sweep_name": "push_magnitude",
  "parameter": "task.domain_randomization.push_vel_range",
  "values": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
  "results": [
    {
      "value": 0.0,
      "episode_reward_mean": 305.1,
      "episode_reward_std": 38.2,
      "success_rate": 0.98,
      "fall_rate": 0.01
    },
    {
      "value": 2.0,
      "episode_reward_mean": 290.3,
      "episode_reward_std": 41.0,
      "success_rate": 0.95,
      "fall_rate": 0.02
    }
  ]
}
```

---

## Phase 6 — Manifest System

Every training run creates a manifest that records **everything** needed to know where this run came from and what happened to it.

### 6.1 Manifest JSON Schema

```json
{
  "run_id": "20260316_143022_dynamite_randomized_dynamite_full_seed42",
  "run_dir": "outputs/randomized/dynamite_full/seed_42/20260316_143022",
  "created_at": "2026-03-16T14:30:22.000000",
  "updated_at": "2026-03-16T17:45:10.000000",
  "status": "completed",
  "config": {
    "seed": 42,
    "task": {"name": "randomized", "...": "..."},
    "model": {"name": "dynamite", "...": "..."},
    "train": {"total_timesteps": 50000000, "...": "..."}
  },
  "git": {
    "commit": "a1b2c3d4e5f6...",
    "commit_short": "a1b2c3d",
    "branch": "main",
    "dirty": false,
    "remote_url": "git@github.com:user/dynamite.git",
    "diff_stat": "",
    "recent_commits": ["a1b2c3d msg1", "b2c3d4e msg2", "..."]
  },
  "system": {
    "hostname": "laptop",
    "os": "Linux-6.1.0-...",
    "python": "3.10.12",
    "pytorch": "2.1.0",
    "cuda": "12.1",
    "cudnn": "8.9.2",
    "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "gpu_memory_mb": 8188,
    "nvidia_driver": "535.86.10",
    "isaaclab_version": "...",
    "cpu_count": 12,
    "ram_gb": 32.0
  },
  "env_vars": {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "42"
  },
  "training": {
    "started_at": "2026-03-16T14:30:25",
    "finished_at": "2026-03-16T17:45:10",
    "wall_time_s": 11685,
    "final_step": 50000000,
    "best_eval_reward": 285.3,
    "best_eval_step": 45000000,
    "total_iterations": 1017
  },
  "checkpoints": [
    "checkpoints/ckpt_0005000000.pt",
    "checkpoints/ckpt_0010000000.pt",
    "checkpoints/latest.pt",
    "checkpoints/best.pt"
  ]
}
```

### 6.2 Manifest Lifecycle

| Status | Meaning | When Set |
|---|---|---|
| `started` | Run has begun | At training start (before first update) |
| `completed` | Run finished normally | After final training iteration |
| `failed` | Run crashed (caught) | In exception handler |
| `interrupted` | Run was killed | Detected retroactively (status=started but no recent checkpoint) |

### 6.3 Status Rules

- A run with `status=started` **and** no checkpoint newer than 1 hour: treat as `interrupted`.
- A run with `status=completed`: **never re-run** unless explicitly forced with `--force`.
- A run with `status=failed` or `interrupted`: **eligible for resume**.

### 6.4 Manifest API

```python
from src.utils.manifest import create_manifest, save_manifest, update_manifest, load_manifest

manifest = create_manifest(run_dir, config)
save_manifest(manifest, run_dir)

# Later, after training completes:
update_manifest(run_dir, {
    "status": "completed",
    "training.finished_at": "...",
    "training.final_step": 50000000,
})

# Read back:
m = load_manifest(run_dir)
```

### 6.5 Enhanced System/Git Collection

For richer manifests, use the dedicated collectors:

```python
from src.utils.system_info import collect_system_info
from src.utils.git_info import collect_git_info

sys_info = collect_system_info()   # OS, Python, PyTorch, CUDA, GPU, RAM, etc.
git_info = collect_git_info()      # commit, branch, dirty, diff_stat, recent_commits
```

---

## Phase 7 — Metrics Recording Standard

### 7.1 Step-Level Metrics (Training Curve)

**File**: `metrics.csv` in each run directory.

**Columns** (14 fields):
```
iteration, global_step, wall_time_s, reward_mean, reward_std,
episode_length_mean, policy_loss, value_loss, entropy, approx_kl,
aux_loss, learning_rate, fps, gpu_mem_mb
```

**Frequency**: Every `train.log_interval` iterations (default: 10).

**Example row**:
```csv
100,4915200,1205.3,45.2,12.1,450.3,-0.0234,0.512,0.823,0.0085,0.0012,0.0003,25600,4200
```

**API**:
```python
from src.utils.metrics_io import write_step_header, append_step_row, read_step_metrics

write_step_header("run_dir/metrics.csv")   # Write CSV header
append_step_row("run_dir/metrics.csv", {   # Append one row
    "iteration": 100,
    "global_step": 4915200,
    "wall_time_s": 1205.3,
    "reward_mean": 45.2,
    # ... all 14 fields
})
rows = read_step_metrics("run_dir/metrics.csv")  # List[dict]
```

### 7.2 Episode-Level Metrics (Eval)

**File**: `eval_episodes.csv` (optional, for per-episode data).

**Columns** (8 fields):
```
episode_idx, reward, length, success, fall, lin_vel_error, ang_vel_error, max_torque
```

### 7.3 Summary-Level Metrics

**File**: `eval_metrics.json` — aggregate stats from evaluation.

**File**: `run_summary.json` — optional end-of-training summary.

### 7.4 TensorBoard Logs

**Directory**: `tb/` inside each run directory.

Logged scalars (via `Logger`):
- `reward/mean`, `reward/std`
- `loss/policy`, `loss/value`, `loss/entropy`, `loss/aux`
- `train/lr`, `train/kl`, `train/clip_frac`
- `perf/fps`, `perf/gpu_mem_mb`
- `eval/reward_mean`, `eval/success_rate`

**View**:
```bash
tensorboard --logdir outputs/randomized/ --port 6006
```

### 7.5 Metrics Discovery

```python
from src.utils.metrics_io import discover_run_dirs, is_run_valid, is_run_complete

# Find all DynaMITE runs on the randomized task
runs = discover_run_dirs("outputs/", task="randomized", model="dynamite")

# Check each
for r in runs:
    valid, issues = is_run_valid(r)
    complete = is_run_complete(r)
    print(f"{r}: valid={valid}, complete={complete}, issues={issues}")
```

---

## Phase 8 — Result Aggregation

### 8.1 Aggregation Rule

For every group (task × model × variant):
1. **Discover** all seed runs matching the group.
2. **Skip** any run where `is_run_complete(run_dir) == False`.
3. **Load** `eval_metrics.json` from each valid run.
4. **Compute** across seeds: `mean`, `std`, `min`, `max`, `CI_95`.
5. **Require** ≥ 2 valid seeds per group; warn if < 3.

### 8.2 Aggregation Commands

```bash
# Full pipeline:
bash scripts/aggregate_all.sh

# Main comparison only:
python src/analysis/aggregate_results.py \
    --base-dir outputs/ \
    --output results/aggregated/

# With ablations:
python src/analysis/aggregate_results.py \
    --base-dir outputs/ \
    --output results/aggregated/ \
    --include-ablations

# Export CSV:
python src/analysis/aggregate_results.py \
    --base-dir outputs/ \
    --output results/aggregated/ \
    --include-ablations \
    --csv
```

### 8.3 Aggregate Output Format

**`results/aggregated/main_comparison.json`**:
```json
{
  "flat/mlp_full": {
    "num_seeds": 3,
    "seeds": [42, 123, 456],
    "metrics": {
      "episode_reward/mean": {
        "mean": 245.3,
        "std": 12.1,
        "min": 230.5,
        "max": 258.2,
        "ci_95": [233.2, 257.4]
      },
      "success_rate": {
        "mean": 0.88,
        "std": 0.03
      }
    }
  },
  "flat/lstm_full": { "..." : "..." },
  "randomized/dynamite_full": { "..." : "..." }
}
```

**`results/aggregated/ablation_results.json`**:
```json
{
  "full": {
    "num_seeds": 3,
    "metrics": { "episode_reward/mean": { "mean": 285.3, "std": 8.2 } }
  },
  "no_latent": {
    "num_seeds": 3,
    "metrics": { "episode_reward/mean": { "mean": 240.1, "std": 10.5 } }
  }
}
```

### 8.4 Seed Aggregation per Group

```bash
# Original raw per-seed tool (still available):
python scripts/aggregate_seeds.py \
    --run_dirs outputs/randomized/dynamite_full/seed_42/... \
               outputs/randomized/dynamite_full/seed_123/... \
               outputs/randomized/dynamite_full/seed_456/... \
    --output results/randomized_dynamite_full_aggregated.json
```

### 8.5 Skip Logic

| Condition | Action |
|---|---|
| `manifest.status == "completed"` and `eval_metrics.json` exists | **Include** in aggregation |
| `manifest.status == "completed"` but no `eval_metrics.json` | **Run eval first**, then include |
| `manifest.status == "started"` or `"failed"` | **Skip** (log warning) |
| `manifest.json` does not exist | **Skip** (log error) |
| Fewer than 2 valid seeds for a group | **Skip group** (log error) |

---

## Phase 9 — Figure & Table Regeneration

### 9.1 Figures

All figures are **deterministically regenerable** from the metrics/results files.

#### Learning Curves (Training Progress)

```bash
# All tasks
python src/analysis/plot_learning_curves.py \
    --base-dir outputs/ \
    --output figures/

# Single task
python src/analysis/plot_learning_curves.py \
    --base-dir outputs/ \
    --task randomized \
    --output figures/
```

**Input**: `metrics.csv` from each run directory.
**Output**: `figures/fig_learning_curves_{task}.pdf` (one per task).
**Content**: 4 curves (one per model), x=global_step, y=reward_mean, shaded ±1σ across seeds, smoothed (EMA α=0.6).

#### Robustness Sweep Plots

```bash
python src/analysis/plot_robustness.py \
    --sweep-dir results/sweeps/ \
    --output figures/
```

**Input**: `sweep_*.json` files from `results/sweeps/`.
**Output**: `figures/fig_sweep_{name}.pdf` (one per sweep parameter).
**Content**: One line per model, x=parameter value, y=reward.

#### Ablation Bar Chart

```bash
python scripts/plot_results.py \
    --results-dir results/aggregated/ \
    --plot ablation \
    --output figures/fig_ablation_bars.pdf
```

#### Full Plot Regeneration Pipeline

```bash
# Regenerate ALL figures from existing data:
python src/analysis/plot_learning_curves.py --base-dir outputs/ --output figures/
python src/analysis/plot_robustness.py --sweep-dir results/sweeps/ --output figures/
python scripts/plot_results.py --results-dir results/aggregated/ --output figures/
```

### 9.2 Tables

```bash
# Generate all tables:
python src/analysis/make_tables.py \
    --results-dir results/aggregated/ \
    --output results/tables/
```

**Inputs**: `results/aggregated/main_comparison.json`, `results/aggregated/ablation_results.json`.

**Outputs**:
| File | Format | Content |
|---|---|---|
| `table_main_comparison.md` | Markdown | 4 models × 4 tasks, mean ± std |
| `table_main_comparison.tex` | LaTeX | Same, for paper |
| `table_ablation.md` | Markdown | 8 variants, reward + Δ from full |
| `table_ablation.tex` | LaTeX | Same, for paper |
| `table_efficiency.md` | Markdown | Parameter counts, memory estimates |

#### Example Main Comparison Table (Markdown)

```
| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | 245.3 ± 12.1 | 180.2 ± 15.3 | 200.1 ± 18.4 | 160.5 ± 20.1 |
| LSTM | 260.1 ± 10.5 | 210.3 ± 13.2 | 230.5 ± 14.1 | 190.2 ± 16.8 |
| Transformer | 265.2 ± 11.0 | 220.1 ± 12.8 | 245.3 ± 13.5 | 200.1 ± 15.2 |
| **DynaMITE (ours)** | **270.5 ± 9.2** | **235.4 ± 11.5** | **285.3 ± 8.2** | **215.3 ± 12.1** |
```

#### Example Ablation Table (LaTeX)

```latex
\begin{table}[h]
\centering
\caption{Ablation study on randomized task (mean reward $\pm$ std).}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Variant & Reward & $\Delta$ \\
\midrule
\textbf{DynaMITE (full)} & 285.3 ± 8.2 & — \\
History = 4 & 270.1 ± 10.5 & -15.2 \\
History = 16 & 280.5 ± 9.0 & -4.8 \\
No latent head & 240.1 ± 12.3 & -45.2 \\
Single (unfactored) latent & 260.2 ± 11.0 & -25.1 \\
No auxiliary loss & 255.3 ± 10.8 & -30.0 \\
1 transformer layer & 265.1 ± 11.2 & -20.2 \\
4 transformer layers & 282.0 ± 9.5 & -3.3 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Phase 10 — Master Run Plan

### 10.1 Summary

| Category | Runs | Time (est.) | Script |
|---|---|---|---|
| Main comparison | 48 (4×4×3) | ~144 h | `scripts/run_all_main.sh` |
| Ablations | 21 (7×3) | ~63 h | `scripts/run_ablations_v2.sh` |
| Eval (main) | 48 | ~2 h | `scripts/run_eval.sh` (loop) |
| Eval (ablation) | 21 | ~1 h | `scripts/run_eval.sh` (loop) |
| Robustness sweeps | 16 (4×4) | ~3 h | `scripts/run_robustness.sh` |
| Aggregation | 1 | ~5 min | `scripts/aggregate_all.sh` |
| Figures | 1 | ~2 min | Plot scripts |
| Tables | 1 | ~1 min | `src/analysis/make_tables.py` |
| **Total** | **~157** | **~213 h (~9 days)** | |

### 10.2 Recommended Execution Order

```
1. [Main comparison] bash scripts/run_all_main.sh
   ├── Runs 48 training jobs sequentially
   ├── ~144 hours
   └── Check: python -m src.utils.validate_runs --base-dir outputs/

2. [Ablations] bash scripts/run_ablations_v2.sh
   ├── 21 training jobs (all on randomized/dynamite)
   ├── ~63 hours
   └── Check: python -m src.utils.validate_runs --base-dir outputs/ --task randomized

3. [Evaluate all] Loop eval over all completed runs
   ├── for d in outputs/*/*/seed_*/*/; do
   │       [ -f "$d/checkpoints/best.pt" ] && [ ! -f "$d/eval_metrics.json" ] && \
   │       python scripts/eval.py --run_dir "$d" --num_episodes 100 --seed 42
   │   done
   └── ~3 hours total

4. [Robustness sweeps] bash scripts/run_robustness.sh
   ├── 16 sweep evaluations
   └── ~3 hours

5. [Aggregate] bash scripts/aggregate_all.sh
   ├── Validate → Aggregate → Tables
   └── ~5 minutes

6. [Figures] Regenerate all plots
   ├── python src/analysis/plot_learning_curves.py --base-dir outputs/ --output figures/
   ├── python src/analysis/plot_robustness.py --sweep-dir results/sweeps/ --output figures/
   └── ~2 minutes

7. [Tables] Regenerate all tables
   ├── python src/analysis/make_tables.py --results-dir results/aggregated/ --output results/tables/
   └── ~1 minute
```

### 10.3 Detailed Command List — Main Comparison

Each `bash scripts/run_train.sh` invocation is one run:

```bash
# ── Flat ──
bash scripts/run_train.sh --task flat --model mlp         --seed 42
bash scripts/run_train.sh --task flat --model mlp         --seed 123
bash scripts/run_train.sh --task flat --model mlp         --seed 456
bash scripts/run_train.sh --task flat --model lstm        --seed 42
bash scripts/run_train.sh --task flat --model lstm        --seed 123
bash scripts/run_train.sh --task flat --model lstm        --seed 456
bash scripts/run_train.sh --task flat --model transformer --seed 42
bash scripts/run_train.sh --task flat --model transformer --seed 123
bash scripts/run_train.sh --task flat --model transformer --seed 456
bash scripts/run_train.sh --task flat --model dynamite    --seed 42
bash scripts/run_train.sh --task flat --model dynamite    --seed 123
bash scripts/run_train.sh --task flat --model dynamite    --seed 456

# ── Push ──
bash scripts/run_train.sh --task push --model mlp         --seed 42
bash scripts/run_train.sh --task push --model mlp         --seed 123
bash scripts/run_train.sh --task push --model mlp         --seed 456
bash scripts/run_train.sh --task push --model lstm        --seed 42
bash scripts/run_train.sh --task push --model lstm        --seed 123
bash scripts/run_train.sh --task push --model lstm        --seed 456
bash scripts/run_train.sh --task push --model transformer --seed 42
bash scripts/run_train.sh --task push --model transformer --seed 123
bash scripts/run_train.sh --task push --model transformer --seed 456
bash scripts/run_train.sh --task push --model dynamite    --seed 42
bash scripts/run_train.sh --task push --model dynamite    --seed 123
bash scripts/run_train.sh --task push --model dynamite    --seed 456

# ── Randomized ──
bash scripts/run_train.sh --task randomized --model mlp         --seed 42
bash scripts/run_train.sh --task randomized --model mlp         --seed 123
bash scripts/run_train.sh --task randomized --model mlp         --seed 456
bash scripts/run_train.sh --task randomized --model lstm        --seed 42
bash scripts/run_train.sh --task randomized --model lstm        --seed 123
bash scripts/run_train.sh --task randomized --model lstm        --seed 456
bash scripts/run_train.sh --task randomized --model transformer --seed 42
bash scripts/run_train.sh --task randomized --model transformer --seed 123
bash scripts/run_train.sh --task randomized --model transformer --seed 456
bash scripts/run_train.sh --task randomized --model dynamite    --seed 42
bash scripts/run_train.sh --task randomized --model dynamite    --seed 123
bash scripts/run_train.sh --task randomized --model dynamite    --seed 456

# ── Terrain ──
bash scripts/run_train.sh --task terrain --model mlp         --seed 42
bash scripts/run_train.sh --task terrain --model mlp         --seed 123
bash scripts/run_train.sh --task terrain --model mlp         --seed 456
bash scripts/run_train.sh --task terrain --model lstm        --seed 42
bash scripts/run_train.sh --task terrain --model lstm        --seed 123
bash scripts/run_train.sh --task terrain --model lstm        --seed 456
bash scripts/run_train.sh --task terrain --model transformer --seed 42
bash scripts/run_train.sh --task terrain --model transformer --seed 123
bash scripts/run_train.sh --task terrain --model transformer --seed 456
bash scripts/run_train.sh --task terrain --model dynamite    --seed 42
bash scripts/run_train.sh --task terrain --model dynamite    --seed 123
bash scripts/run_train.sh --task terrain --model dynamite    --seed 456
```

### 10.4 Detailed Command List — Ablations

All use `--task randomized --model dynamite`:

```bash
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_4      --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_4      --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_4      --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_16     --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_16     --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant seq_len_16     --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant no_latent      --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant no_latent      --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant no_latent      --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant single_latent  --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant single_latent  --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant single_latent  --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant no_aux_loss    --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant no_aux_loss    --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant no_aux_loss    --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_1        --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_1        --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_1        --seed 456
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_4        --seed 42
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_4        --seed 123
bash scripts/run_train.sh --task randomized --model dynamite --variant depth_4        --seed 456
```

---

## Phase 11 — Failure & Recovery Protocol

### 11.1 Failure Detection

Run the validator:
```bash
python -m src.utils.validate_runs --base-dir outputs/
```

This checks every run for:
- Missing `config.yaml` or `manifest.json`
- `manifest.status` still set to `"started"` (likely crash/kill)
- No checkpoints
- NaN/Inf in `metrics.csv`
- Missing eval results

### 11.2 Failure Types and Responses

| Failure | Detection | Response |
|---|---|---|
| **OOM crash** | Process killed, `manifest.status="started"` | Reduce `task.num_envs` (try 1024), then resume |
| **NaN divergence** | NaN in `metrics.csv` or loss fields | Delete run, restart with lower LR or smaller grad norm |
| **SIGKILL / power loss** | `manifest.status="started"`, recent `latest.pt` exists | Resume: `--resume <run_dir>` |
| **SIGKILL / power loss** | `manifest.status="started"`, no `latest.pt` | Delete run, restart from scratch |
| **Corrupted checkpoint** | `verify_checkpoint()` returns False | Use previous checkpoint or restart |
| **Slow training** | FPS drops > 50% from start | Check GPU thermals, restart Isaac Sim, reduce envs |
| **Eval failure** | `eval_metrics.json` missing after eval script | Re-run eval: `python scripts/eval.py --run_dir ...` |

### 11.3 Resume Command Template

```bash
# Always use the ORIGINAL task/model/seed args + --resume:
python scripts/train.py \
    --task configs/task/randomized.yaml \
    --model configs/model/dynamite.yaml \
    --seed 42 \
    --resume outputs/randomized/dynamite_full/seed_42/20260316_143022
```

### 11.4 Checkpoint Utilities

```python
from src.utils.checkpoint_utils import (
    verify_checkpoint,
    list_checkpoints,
    get_checkpoint_meta,
    find_best_and_latest,
    cleanup_old_checkpoints,
)

# Check if a checkpoint file is loadable
ok, msg = verify_checkpoint("path/to/ckpt.pt")

# List all checkpoints in a run
ckpts = list_checkpoints("run_dir/checkpoints/")

# Get metadata without loading full tensors
meta = get_checkpoint_meta("path/to/ckpt.pt")
# → {"step": 5000000, "has_model": True, "has_optimizer": True, "file_size_mb": 12.3}

# Find best and latest
best, latest = find_best_and_latest("run_dir/checkpoints/")

# Clean up old checkpoints (keep last 3 + best + latest)
cleanup_old_checkpoints("run_dir/checkpoints/", keep_last=3, dry_run=True)
```

### 11.5 Retry Policy

1. **First failure**: Resume from latest checkpoint.
2. **Second failure (same run)**: Reduce `num_envs` to 1024, resume.
3. **Third failure**: Delete run, restart with `--set train.learning_rate=1e-4`.
4. **If NaN appeared**: Always delete and restart (NaN is non-recoverable).
5. **Maximum retries per run**: 3. After 3 failures, mark as permanently failed and document.

### 11.6 Checkpoint Cleanup

To conserve disk space during long campaigns:
```bash
# Dry run first:
python -c "
from src.utils.checkpoint_utils import cleanup_old_checkpoints
import glob
for d in glob.glob('outputs/*/*/seed_*/*/checkpoints/'):
    cleanup_old_checkpoints(d, keep_last=3, dry_run=True)
"

# Actual cleanup:
python -c "
from src.utils.checkpoint_utils import cleanup_old_checkpoints
import glob
for d in glob.glob('outputs/*/*/seed_*/*/checkpoints/'):
    cleanup_old_checkpoints(d, keep_last=3, dry_run=False)
"
```

---

## Phase 12 — Reproducibility Checklist

### 12.1 Environment Lock

Before ANY experiment run:

```bash
# 1. Export exact Python environment
pip freeze > requirements_frozen.txt
conda env export > environment_frozen.yml

# 2. Record system info
python -c "from src.utils.system_info import collect_system_info; import json; print(json.dumps(collect_system_info(), indent=2))" > reproducibility/system_info.json

# 3. Record git state
python -c "from src.utils.git_info import collect_git_info; import json; print(json.dumps(collect_git_info(), indent=2))" > reproducibility/git_info.json

# 4. Ensure clean git state
git status  # Should show no uncommitted changes
```

### 12.2 Required Environment Variables

Set these before every training session:

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
```

### 12.3 Seed Policy

| Experiment type | Seeds | Rationale |
|---|---|---|
| Main comparison | 42, 123, 456 | 3 seeds for mean±std |
| Ablations | 42, 123, 456 | Same 3 seeds for fair comparison |
| Robustness sweeps | 42 | Single seed (eval only, low variance) |
| Debug / development | 0 | Clearly separate from real experiments |

### 12.4 Determinism Settings

The seed utility (`src/utils/seed.py`) sets:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`

**Note**: Full CUDA determinism (`torch.use_deterministic_algorithms(True)`) is NOT enabled by default because Isaac Lab/PhysX is inherently non-deterministic with multi-GPU/multi-env. Same seed gives **similar** but not bit-identical results. This is expected and acceptable.

### 12.5 Version Requirements

| Component | Minimum Version | Tested Version |
|---|---|---|
| Python | 3.10 | 3.10.12 |
| PyTorch | 2.0 | 2.1.0 |
| CUDA | 11.8 | 12.1 |
| Isaac Lab | — | (record in manifest) |
| Isaac Sim | — | (record in manifest) |

### 12.6 Rerun Policy

- **Never overwrite** a completed run. Always create a new timestamp directory.
- If a run must be discarded, move it to `outputs/_discarded/` rather than deleting.
- To verify reproducibility, run 2 seeds of the same config and check that rewards are within ±10% by iteration 500.

### 12.7 Pre-Flight Validation

Run this before starting a full experiment campaign:

```bash
# Verify all configs parse without error
python -c "
from src.utils.config import load_config
import itertools
tasks = ['flat', 'push', 'randomized', 'terrain']
models = ['mlp', 'lstm', 'transformer', 'dynamite']
for t, m in itertools.product(tasks, models):
    cfg = load_config(
        base_path='configs/base.yaml',
        task_path=f'configs/task/{t}.yaml',
        model_path=f'configs/model/{m}.yaml',
    )
    print(f'  OK: {t}/{m} → seed={cfg[\"seed\"]}, steps={cfg[\"train\"][\"total_timesteps\"]:,}')
print('All configs valid.')
"

# Quick smoke test (10k steps)
python scripts/train.py \
    --task configs/task/flat.yaml \
    --model configs/model/mlp.yaml \
    --seed 0 \
    --set train.total_timesteps=10000 task.num_envs=64
```

---

## Phase 13 — README: Running Experiments

> Copy-paste this section into `README.md` under a `## Running Experiments` heading.

---

### Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate dynamite
pip install -e .

# 2. Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# 3. Smoke test (fast, ~2 minutes)
python scripts/train.py \
    --task configs/task/flat.yaml \
    --model configs/model/mlp.yaml \
    --seed 0 \
    --set train.total_timesteps=10000 task.num_envs=64
```

### Training a Single Model

```bash
# MLP on flat terrain, seed 42
python scripts/train.py \
    --task configs/task/flat.yaml \
    --model configs/model/mlp.yaml \
    --seed 42

# DynaMITE on randomized dynamics, seed 42
python scripts/train.py \
    --task configs/task/randomized.yaml \
    --model configs/model/dynamite.yaml \
    --seed 42
```

### Running All Experiments

```bash
# Main comparison (48 runs, ~144 hours)
bash scripts/run_all_main.sh

# Ablations (21 runs, ~63 hours)
bash scripts/run_ablations_v2.sh

# Use --skip-existing to resume an interrupted campaign:
bash scripts/run_all_main.sh --skip-existing
```

### Evaluating

```bash
# Evaluate a single run
python scripts/eval.py \
    --run_dir outputs/randomized/dynamite_full/seed_42/<timestamp>/ \
    --num_episodes 100 --seed 42

# Robustness sweeps (16 evaluations)
bash scripts/run_robustness.sh
```

### Generating Results

```bash
# Aggregate all results
bash scripts/aggregate_all.sh

# Regenerate figures
python src/analysis/plot_learning_curves.py --base-dir outputs/ --output figures/
python src/analysis/plot_robustness.py --sweep-dir results/sweeps/ --output figures/

# Regenerate tables
python src/analysis/make_tables.py --results-dir results/aggregated/ --output results/tables/
```

### Validating Runs

```bash
# Check all runs for problems
python -m src.utils.validate_runs --base-dir outputs/
```

### Resuming a Crashed Run

```bash
python scripts/train.py \
    --task configs/task/randomized.yaml \
    --model configs/model/dynamite.yaml \
    --seed 42 \
    --resume outputs/randomized/dynamite_full/seed_42/<timestamp>
```

---

## Phase 14 — Files Inventory

### 14.1 Operational Utilities (New)

| File | Purpose |
|---|---|
| `src/utils/system_info.py` | Collect OS, Python, PyTorch, CUDA, GPU, RAM versions |
| `src/utils/git_info.py` | Collect git commit, branch, dirty status, recent commits |
| `src/utils/run_naming.py` | Deterministic run ID, group ID, directory path construction |
| `src/utils/metrics_io.py` | Standardized CSV/JSON I/O with 14-column step metrics |
| `src/utils/checkpoint_utils.py` | Verify, list, clean up checkpoints |
| `src/utils/validate_runs.py` | Detect failed/incomplete/NaN runs, produce health report |

### 14.2 Shell Run Scripts (New)

| File | Purpose |
|---|---|
| `scripts/run_train.sh` | Single run launcher with logging and dry-run |
| `scripts/run_eval.sh` | Evaluation launcher |
| `scripts/run_all_main.sh` | 48-run main comparison campaign |
| `scripts/run_ablations_v2.sh` | 21-run ablation campaign |
| `scripts/run_robustness.sh` | 16-run robustness sweep campaign |
| `scripts/aggregate_all.sh` | Full aggregation pipeline |

### 14.3 Analysis Tools (New)

| File | Purpose |
|---|---|
| `src/analysis/aggregate_results.py` | Production aggregator with CLI |
| `src/analysis/plot_learning_curves.py` | Training curve plots with multi-seed shading |
| `src/analysis/plot_robustness.py` | Robustness sweep line charts |
| `src/analysis/make_tables.py` | Markdown + LaTeX table generator |

### 14.4 Existing Core Files (From Prior Session)

| Category | Files |
|---|---|
| Entrypoints | `scripts/train.py`, `scripts/eval.py` |
| Config | `src/utils/config.py` + all YAML files |
| Training | `src/algos/ppo.py` (PPOTrainer, RolloutBuffer) |
| Models | `src/models/{components,mlp_policy,lstm_policy,transformer_policy,dynamite_policy}.py` |
| Environment | `src/envs/g1_env.py`, `src/envs/reward.py` |
| Utilities | `src/utils/{seed,logger,history_buffer,checkpoint,manifest,metrics}.py` |
| Analysis | `src/analysis/{plotting,tables,latent_analysis}.py` |
| Scripts | `scripts/{plot_results,aggregate_seeds,generate_tables}.py` |
| Tests | `tests/test_*.py` |
| Docs | `docs/{architecture,config_system,experiment_plan}.md` |

### 14.5 This Document

| File | Purpose |
|---|---|
| `docs/experiment_protocol.md` | **This file** — the 15-phase operations protocol |

---

## Phase 15 — Final Audit

### 15.1 Realistic Assessment

**What this system does well:**
- **Naming**: Fully deterministic, parseable run IDs and directory paths. No ambiguity.
- **Recording**: Every run captures config, manifest (git + system + status), step-level CSV, TensorBoard, checkpoints.
- **Failure detection**: Automated validation catches missing files, NaN, interrupted runs.
- **Aggregation**: Automated discovery → skip logic → mean/std/CI → JSON/CSV output.
- **Regeneration**: All figures and tables regenerable from data files with one command.
- **Reproducibility**: Environment locking, seed policy, version recording, determinism settings.
- **Scalability**: `--skip-existing` and `--dry-run` flags on all campaign scripts.

### 15.2 Known Weak Points

| Weakness | Severity | Mitigation |
|---|---|---|
| **Isaac Lab/PhysX non-determinism** | Medium | Same-seed runs produce similar (±5%) but not identical results. Addressed by using 3 seeds and reporting mean±std. |
| **Sequential execution only** | Medium | RTX 4060 can only run 1 training at a time. No parallelism. Total wall time ~9 days. |
| **No automatic retry** | Low | Failed runs require manual resume. `validate_runs.py` identifies them, but the human must act. |
| **Checkpoint disk usage** | Low | Each run saves periodic checkpoints (~15 MB each). With 69 training runs × ~20 checkpoints = ~20 GB. Use `cleanup_old_checkpoints()` to manage. |
| **No live monitoring** | Low | TensorBoard provides live monitoring during a run, but there's no cross-run dashboard. Check with `tensorboard --logdir outputs/`. |
| **No hyperparameter search** | By design | This is a fixed comparison paper, not an AutoML search. All hyperparams are set in config files. |

### 15.3 Minimum Viable Experiment Set

If time-constrained, run this subset first:

| Category | Runs | Time | What You Get |
|---|---|---|---|
| Randomized task only (4 models × 3 seeds) | 12 | ~36 h | Main result: DynaMITE vs baselines on the hardest task |
| Ablations (7 × 1 seed) | 7 | ~21 h | Ablation table (less statistical power) |
| Sweeps (4 methods × 4 params, 1 seed) | 16 eval-only | ~3 h | Robustness plots |
| **Subtotal** | **35** | **~60 h** | Paper core |

Then expand:
- Add seeds 123, 456 for ablations (+14 runs, +42 h)
- Add flat/push/terrain tasks for main comparison (+36 runs, +108 h)

### 15.4 Pre-Campaign Checklist

Before starting the full experiment campaign, verify:

- [ ] `git status` shows clean working tree (or commit all changes)
- [ ] `pip freeze > requirements_frozen.txt` captures exact environment
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] Smoke test passes: `python scripts/train.py --task configs/task/flat.yaml --model configs/model/mlp.yaml --seed 0 --set train.total_timesteps=10000 task.num_envs=64`
- [ ] Config validation passes (see Phase 12.7)
- [ ] `CUDA_VISIBLE_DEVICES`, `CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED` are set
- [ ] Sufficient disk space: `df -h .` shows ≥50 GB free
- [ ] `nvidia-smi` shows expected GPU and driver
- [ ] All shell scripts are executable: `ls -la scripts/*.sh`
- [ ] `outputs/` directory exists and is writable

### 15.5 Post-Campaign Checklist

After all runs complete:

- [ ] `python -m src.utils.validate_runs --base-dir outputs/` shows 0 failed, 0 interrupted
- [ ] All 69 training runs (48 main + 21 ablation) have `manifest.status="completed"`
- [ ] All 69 runs have `eval_metrics.json`
- [ ] All 16 sweep results exist in run directories
- [ ] `bash scripts/aggregate_all.sh` completes without errors
- [ ] `results/aggregated/main_comparison.json` contains all 16 groups (4×4)
- [ ] `results/aggregated/ablation_results.json` contains all 8 variants
- [ ] All figures regenerated in `figures/`
- [ ] All tables regenerated in `results/tables/`
- [ ] `git add` all results, figures, tables; commit with descriptive message
- [ ] Archive `outputs/` to external storage (it's large)

---

*End of protocol. This document should be treated as the single source of truth for all experiment operations in the DynaMITE project.*
