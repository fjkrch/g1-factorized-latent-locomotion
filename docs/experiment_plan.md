# Experiment Plan

## Overview

This document describes the complete experiment plan for the DynaMITE project. All experiments are designed to be executable on a single NVIDIA RTX 4060 Laptop GPU (8 GB VRAM).

---

## Experiment Categories

| Category | Purpose | Runs | Est. Time |
|---|---|---|---|
| Main comparison | Compare 4 methods across 4 tasks | 48 | ~120 h |
| Ablation study | Isolate DynaMITE design decisions | 21 | ~52 h |
| Robustness sweeps | Test OOD generalization | ~48 evals | ~4 h |
| Latent analysis | Verify disentanglement | 1 analysis | ~1 h |
| **Total** | | **~70 train + evals** | **~177 h** |

---

## 1. Main Method Comparison

### Goal

Compare DynaMITE against three baselines (MLP, LSTM, Transformer) across four locomotion tasks of increasing difficulty.

### Experiment Matrix

| | flat | push | randomized | terrain |
|---|---|---|---|---|
| **MLP** | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| **LSTM** | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| **Transformer** | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| **DynaMITE** | 3 seeds | 3 seeds | 3 seeds | 3 seeds |

**Total**: 4 methods × 4 tasks × 3 seeds = **48 runs**

### Seeds

All experiments use seeds: **42, 43, 44**.

### Per-Run Configuration

- **Environments**: 4096 parallel
- **Iterations**: 3000
- **Steps per iteration**: 24
- **Total environment steps**: 3000 × 24 × 4096 = ~295M steps per run
- **Estimated wall time**: ~2.5 hours per run

### Execution Order

1. Run `flat` task for all methods first (sanity check)
2. Run `push` task
3. Run `randomized` task (primary evaluation)
4. Run `terrain` task

Within each task, run methods in order: MLP → LSTM → Transformer → DynaMITE. This ensures simpler methods complete first for early comparison.

### Commands

```bash
# Option 1: Run all sequentially
bash scripts/run_all_baselines.sh

# Option 2: Run one at a time
python scripts/train.py --task flat --model mlp --seed 42
# ... continue for all combinations
```

### Expected Outputs

Per run:
- `config.yaml`, `manifest.json`, `metrics.csv`
- `checkpoints/best.pt`, `checkpoints/latest.pt`
- TensorBoard logs in `tb/`

Aggregated:
- `results/aggregated/main_comparison.json`

### Evaluation Criteria

| Metric | Definition | Primary? |
|---|---|---|
| Mean episodic reward | Average reward over 100 eval episodes | **Yes** |
| Reward std (across seeds) | Standard deviation over 3 seeds | Yes |
| Episode length | Average episode duration (longer = more stable) | Secondary |
| Success rate | Fraction of episodes without falling | Secondary |

### Analysis

1. **Training curves** — reward vs. iteration for all methods (Figure 1)
2. **Eval bar chart** — final reward ± std for all method-task combinations (Figure 2)
3. **Main results table** — tabulated means ± std (Table 1)

---

## 2. Ablation Study

### Goal

Isolate the contribution of each DynaMITE design decision by removing or varying one component at a time.

### Ablation Matrix

All ablations use the `randomized` task (most informative due to full domain randomization).

| Ablation ID | What Changes | Baseline | Hypothesis |
|---|---|---|---|
| `seq_len_4` | history_len: 8 → 4 | DynaMITE | 4 steps too short; ↓ reward |
| `seq_len_16` | history_len: 8 → 16 | DynaMITE | Diminishing returns; ≈ reward, ↑ cost |
| `no_latent` | Disable latent head | DynaMITE | Degrades to vanilla transformer; ↓ reward |
| `single_latent` | Unfactored 24-d latent | DynaMITE | Loses disentanglement; slight ↓ |
| `no_aux_loss` | Disable aux identification loss | DynaMITE | Latent lacks semantics; ↓ reward |
| `depth_1` | num_layers: 2 → 1 | DynaMITE | Insufficient capacity; ↓ reward |
| `depth_4` | num_layers: 2 → 4 | DynaMITE | Overkill; ≈ reward, ↑ cost |

**Total**: 7 ablations × 3 seeds = **21 runs**

### Execution

```bash
bash scripts/run_ablations.sh
```

### Expected Ordering (reward on `randomized`)

```
DynaMITE (full) > single_latent ≈ no_aux_loss > seq_len_4 ≈ depth_1 > no_latent > seq_len_16 (≈ full but slower) > depth_4 (≈ full but slower)
```

### Analysis

1. **Ablation bar chart** — reward comparison (Figure 3)
2. **Ablation table** — tabulated means ± std with Δ from full DynaMITE (Table 2)

---

## 3. Robustness Sweeps

### Goal

Test how each method degrades as dynamics parameters shift beyond the training distribution.

### Sweep Parameters

| Sweep | Parameter | Range | Steps | Training Range |
|---|---|---|---|---|
| Push magnitude | External force | [0, 50, 100, 150, 200, 250, 300] N | 7 | [50, 150] N |
| Friction | Ground friction coefficient | [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5] | 8 | [0.3, 2.0] |
| Action delay | Actuation delay steps | [0, 1, 2, 3, 4, 5, 6] | 7 | [0, 3] |

### Procedure

For each sweep parameter and each method:
1. Load the best checkpoint from `randomized` task (seed 42)
2. Evaluate at each sweep value for 50 episodes
3. Record mean reward at each level

### Commands

```bash
bash scripts/run_sweeps.sh
```

### Expected Results

DynaMITE should show:
- **Slower degradation** as parameters shift OOD
- **Higher reward** at extreme values (e.g., very low friction, high push force)
- The gap between DynaMITE and baselines should **widen** at OOD values

### Analysis

1. **Sweep line plots** — reward vs. parameter value for all methods (Figures 4–6)

---

## 4. Latent Space Analysis

### Goal

Verify that DynaMITE's factored latent space captures meaningful, disentangled dynamics information.

### Methods

1. **t-SNE visualization**: Collect latent vectors z from 10,000 evaluation steps with known GT dynamics parameters. Color by GT parameter value. If disentangled, t-SNE of z_friction should show structure correlated with friction values.

2. **Correlation matrix**: Compute Pearson correlation between each latent factor dimension and each GT dynamics parameter. A disentangled latent should show block-diagonal structure (z_friction correlates with GT friction, not GT mass).

3. **Disentanglement score**: Quantify the degree of factorization using a simple metric: for each GT parameter, the R² of predicting it from the corresponding latent factor vs. from other factors. High score = good factorization.

### Commands

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --task randomized \
  --analyze-latent
```

### Expected Results

- t-SNE clusters should be well-separated and colored smoothly by GT parameter value
- Correlation matrix should show block-diagonal structure
- Disentanglement score > 0.7 (on a 0–1 scale)

### Analysis

1. **t-SNE plots** — one per GT parameter (Figures 7–8)
2. **Correlation heatmap** — latent dims × GT params
3. **Disentanglement score** — single number in Table 3

---

## Experiment Execution Timeline

For a single researcher on RTX 4060 Laptop GPU:

### Week 1–2: Setup & Sanity

| Day | Activity | Time |
|---|---|---|
| 1 | Install Isaac Sim + Isaac Lab | 4 h |
| 2 | Verify environment, run unit tests | 2 h |
| 3 | Smoke test all 4 models on flat task | 2 h |
| 4–5 | Full training: flat task, all methods, seed 42 | 10 h |

### Week 3–5: Main Experiments

| Day | Activity | Time |
|---|---|---|
| 6–8 | Flat task: remaining seeds | 16 h |
| 9–12 | Push task: all methods, 3 seeds | 30 h |
| 13–17 | Randomized task: all methods, 3 seeds | 30 h |
| 18–22 | Terrain task: all methods, 3 seeds | 30 h |

### Week 6–7: Ablations & Sweeps

| Day | Activity | Time |
|---|---|---|
| 23–28 | Ablation study: 21 runs | 52 h |
| 29 | Robustness sweeps | 4 h |
| 30 | Latent analysis | 1 h |

### Week 8: Analysis & Writing

| Day | Activity | Time |
|---|---|---|
| 31 | Aggregate results, generate plots/tables | 2 h |
| 32–35 | Analyze results, iterate on figures | flexible |

**Total calendar time**: ~8 weeks (with runs primarily overnight/unattended)

---

## Minimum Viable Experiment Set

If time is limited, the minimum experiments for a credible submission:

1. **Main comparison on `randomized` task** (4 methods × 3 seeds = 12 runs ≈ 30 h)
2. **One cross-task** validation (flatten or terrain, 4 methods × 1 seed = 4 runs ≈ 10 h)
3. **Top 3 ablations** (no_latent, no_aux_loss, single_latent, 3 seeds each = 9 runs ≈ 22 h)
4. **One robustness sweep** (push magnitude, ~1 h)
5. **Latent t-SNE** (~30 min)

**Minimum total**: ~25 runs, ~64 hours

---

## Data Management

### Storage Requirements

| Data Type | Per Run | Total (all experiments) |
|---|---|---|
| Checkpoints (best + latest) | ~5 MB | ~350 MB |
| Metrics CSV | ~2 MB | ~140 MB |
| TensorBoard logs | ~50 MB | ~3.5 GB |
| Config + manifest | ~5 KB | ~350 KB |
| **Total** | ~57 MB | **~4 GB** |

### Backup Strategy

1. After each batch of runs, copy `config.yaml` and `eval_metrics.json` to `results/`
2. Keep `checkpoints/best.pt` for all runs; delete periodic checkpoints if space is tight
3. Back up `results/aggregated/` — this is the most valuable post-experiment artifact
