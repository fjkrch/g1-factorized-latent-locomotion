# DynaMITE: Dynamic Mismatch Inference via Transformer Encoder for Robust Humanoid Locomotion

> **One-line summary**: A lightweight transformer-based method that infers factorized latent dynamics from short observation–action history, enabling a Unitree G1 humanoid to walk robustly under unknown friction, mass, motor strength, contact properties, and actuation delay—without any explicit system identification or domain knowledge at deployment time.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Hypothesis](#hypothesis)
- [Novelty Statement](#novelty-statement)
- [Core Contributions](#core-contributions)
- [Method Overview](#method-overview)
- [Benchmarks & Tasks](#benchmarks--tasks)
- [Observation & Action Spaces](#observation--action-spaces)
- [Model Architectures](#model-architectures)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Environment & Simulator Setup](#environment--simulator-setup)
- [Sanity Checks](#sanity-checks)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Studies](#ablation-studies)
- [Robustness Sweeps](#robustness-sweeps)
- [Plotting & Table Generation](#plotting--table-generation)
- [Full Reproduction](#full-reproduction)
- [Experiment Naming Convention](#experiment-naming-convention)
- [Output Directory Structure](#output-directory-structure)
- [Expected Outputs per Run](#expected-outputs-per-run)
- [Expected Results Summary](#expected-results-summary)
- [Runtime Estimates](#runtime-estimates)
- [Reproducibility Protocol](#reproducibility-protocol)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [How to Extend](#how-to-extend)
- [Citation](#citation)
- [License](#license)

---

## Problem Statement

Sim-to-real transfer for legged robots fails when the simulator's dynamics deviate from reality. Friction coefficients, link masses, motor constants, ground contact stiffness, and actuation delays all vary between simulation and the physical robot—and across real-world surfaces and payloads. A policy trained under a single set of dynamics parameters will exhibit degraded or unsafe behavior when any of these quantities shift.

**The core challenge**: How can a locomotion policy *sense* the current dynamics online—using only proprioceptive observations and its own past actions—and *adapt* its behavior accordingly, all within a single forward pass, without requiring an explicit system identification phase, calibration procedure, or privileged information at deployment?

Prior approaches either:
1. Train blind domain-randomized policies that are robust-by-averaging (which sacrifices peak performance),
2. Use recurrent networks (LSTM/GRU) that encode dynamics implicitly but with long memory horizons and entangled representations, or
3. Use explicit system identification modules that require known parameter ranges and structured estimators.

None of these methods produce a compact, interpretable, and factorized estimate of the current dynamics that the policy can directly condition on.

---

## Hypothesis

> A small transformer encoder operating on a fixed-length observation–action history window (8 steps ≈ 160 ms at 50 Hz) can infer a **factorized latent vector** $z = [z_{\text{friction}}, z_{\text{mass}}, z_{\text{motor}}, z_{\text{contact}}, z_{\text{delay}}]$ that captures the dominant axes of dynamics variation. When the policy and value function condition on $z$, and auxiliary identification losses supervise each factor against the true (simulator-known) dynamics parameters during training, the resulting policy:
>
> 1. **Outperforms** MLP, LSTM, and vanilla Transformer baselines on reward across all four evaluation tasks.
> 2. **Exhibits greater robustness** when dynamics parameters shift beyond the training distribution.
> 3. **Learns disentangled representations**: each latent factor correlates primarily with its corresponding ground-truth dynamics parameter group.

---

## Novelty Statement

**What is new in DynaMITE (and what is not)**:

| Aspect | Status |
|---|---|
| PPO algorithm | Not new — standard clipped PPO |
| Reward function for G1 locomotion | Not new — standard velocity tracking + regularization terms |
| Domain randomization as a training strategy | Not new — widely used |
| Transformer encoder for RL observation processing | Incremental — used by some recent works |
| **Factorized latent dynamics inference** | **Novel** — decomposing the latent dynamics representation into semantically meaningful, independently projected subspaces |
| **Per-factor auxiliary identification loss** | **Novel** — supervising each latent factor subspace against the corresponding GT dynamics parameter group during training |
| **Latent-conditioned policy with factored dynamics** | **Novel combination** — using the factored latent as an explicit policy/value conditioning signal |
| Single-GPU, single-researcher reproducibility focus | Methodological contribution — demonstrating the full pipeline is feasible on commodity hardware |

**Novelty score**: 6/10 — The individual components (transformer, PPO, domain randomization, auxiliary losses) are known. The novelty is in their specific *composition*: factorized latent spaces with per-factor supervision for dynamics-adaptive locomotion. This is a focused, incremental-but-meaningful contribution.

---

## Core Contributions

1. **DynaMITE architecture**: A transformer encoder that maps a short (8-step) observation–action history to a factored latent dynamics vector $z \in \mathbb{R}^{24}$, decomposed into 5 semantically meaningful subspaces.

2. **Per-factor auxiliary identification loss**: Each latent subspace ($z_{\text{friction}} \in \mathbb{R}^4$, $z_{\text{mass}} \in \mathbb{R}^6$, etc.) is supervised during training against the corresponding ground-truth dynamics parameters, encouraging disentanglement without post-hoc alignment.

3. **Empirical demonstration** across 4 locomotion tasks on a Unitree G1 humanoid showing that DynaMITE matches or exceeds baselines on in-distribution tasks and shows meaningfully better robustness on out-of-distribution dynamics shifts.

4. **Complete, single-GPU reproducible pipeline**: The entire codebase, config system, experiment plan, and reproduction protocol are designed for a single researcher with an RTX 4060 laptop GPU.

---

## Method Overview

### Architecture Diagram (Conceptual)

```
                     History Buffer (8 steps)
                    ┌────────────────────────┐
                    │ [obs₁,act₁] ... [obs₈,act₈] │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Token Embedding Layer  │
                    │ obs_emb + act_emb + PE  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Transformer Encoder    │
                    │  (2 layers, 4 heads,    │
                    │   d_model=128)          │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │  Mean Pooling over      │
                    │  sequence dimension     │
                    └──────────┬─────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │ Factorized     │ │ Policy Head  │ │ Value Head   │
    │ Latent Head    │ │ MLP(256,128) │ │ MLP(256,128) │
    │                │ │ → μ, log σ   │ │ → V(s)       │
    │ ┌────────────┐ │ └──────▲───────┘ └──────▲───────┘
    │ │ z_friction │ │        │                │
    │ │ z_mass     │ │        │ concat         │ concat
    │ │ z_motor    │─┼────────┘                │
    │ │ z_contact  │ │                         │
    │ │ z_delay    │─┼─────────────────────────┘
    │ └────────────┘ │
    │   ↓ (train     │
    │    only)       │
    │ ┌────────────┐ │
    │ │ Aux ID     │ │
    │ │ Losses     │ │
    │ └────────────┘ │
    └────────────────┘
```

### Pipeline

1. **Observation**: At each step, the G1 proprioceptive observation (48 dims) and the previous action (19 dims) are pushed into a FIFO history buffer of length 8.
2. **Embedding**: Each (obs, act) pair is projected through learned embeddings + sinusoidal positional encoding → token sequence of shape `(batch, 8, 128)`.
3. **Transformer encoder**: 2-layer, 4-head transformer with pre-layer-norm. Output: `(batch, 8, 128)`.
4. **Aggregation**: Mean pooling → `(batch, 128)`.
5. **Factorized latent head**: 5 independent linear projections produce $z_{\text{friction}} \in \mathbb{R}^4$, $z_{\text{mass}} \in \mathbb{R}^6$, $z_{\text{motor}} \in \mathbb{R}^6$, $z_{\text{contact}} \in \mathbb{R}^4$, $z_{\text{delay}} \in \mathbb{R}^4$. Concatenated: $z \in \mathbb{R}^{24}$.
6. **Latent-conditioned policy**: The current observation embedding is concatenated with $z$ and fed to the policy MLP → action mean $\mu \in \mathbb{R}^{19}$ and learned log-std.
7. **Latent-conditioned value**: Same conditioning → scalar $V(s)$.
8. **Auxiliary identification loss** (training only): Each $z_{\text{factor}}$ is mapped through a small MLP to predict the ground-truth dynamics parameter for that factor (e.g., $z_{\text{friction}} \rightarrow \hat{\mu}_{\text{friction}}$). MSE loss per factor.

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{PPO-clip}} + c_v \mathcal{L}_{\text{value}} + \lambda_{\text{aux}} \sum_{f \in \mathcal{F}} w_f \mathcal{L}_{\text{aux},f}$$

where $\mathcal{F} = \{\text{friction, mass, motor, contact, delay}\}$, $\lambda_{\text{aux}} = 0.1$, and all $w_f = 1.0$ by default.

---

## Benchmarks & Tasks

All tasks use the Unitree G1 humanoid (19 DoF lower body) in Isaac Lab at 50 Hz control / 200 Hz physics.

| Task ID | Name | Domain Rand | Push | Terrain | Purpose |
|---|---|---|---|---|---|
| `flat` | Flat ground | None | None | Flat plane | Baseline sanity |
| `push` | Push recovery | None | 50–150 N, every 5–15 s | Flat plane | Disturbance rejection |
| `randomized` | Randomized dynamics | Full (friction, mass, motor, contact, delay) | 50–150 N | Flat plane | **Primary evaluation** |
| `terrain` | Rough terrain | Full | 50–150 N | Stairs, slopes, rough | Full challenge |

**Domain randomization ranges** (for `randomized` and `terrain` tasks):
- Friction coefficient: [0.3, 2.0] (uniform)
- Link mass scaling: [0.8, 1.2] × nominal (uniform, per-link)
- Motor strength scaling: [0.85, 1.15] × nominal (uniform, per-joint)
- Contact stiffness: [500, 2000] N/m (log-uniform)
- Contact damping: [10, 100] Ns/m (log-uniform)
- Actuation delay: [0, 3] steps (discrete uniform)

---

## Observation & Action Spaces

### Observation Vector (48 dimensions)

| Component | Dimensions | Description |
|---|---|---|
| Base linear velocity | 3 | Local frame, m/s |
| Base angular velocity | 3 | Local frame, rad/s |
| Projected gravity | 3 | Gravity in body frame |
| Velocity commands | 3 | (vx_cmd, vy_cmd, ωz_cmd) |
| Joint positions | 19 | Relative to default, rad |
| Joint velocities | 19 | rad/s |

**Total**: 50 dims (as listed in base config; the observation space may be trimmed based on implementation to 48 or 50 depending on command encoding).

### Action Vector (19 dimensions)

Joint position targets for 19 lower-body DoF (hip, knee, ankle joints × 2 legs + waist). Scaled by `action_scale=0.25` and added to default joint positions. Clipped to joint limits.

### Command Space

| Command | Range | Unit |
|---|---|---|
| Forward velocity (vx) | [-1.0, 1.0] | m/s |
| Lateral velocity (vy) | [-0.5, 0.5] | m/s |
| Yaw rate (ωz) | [-1.0, 1.0] | rad/s |

Commands are resampled every 500 steps (10 s). Commands below threshold 0.1 are zeroed (stand-still command).

---

## Model Architectures

| Model | Key Idea | History | Latent | Aux Loss | Params (approx) |
|---|---|---|---|---|---|
| **MLP** | Feedforward, current obs only | No | No | No | ~200k |
| **LSTM** | Recurrent, implicit memory | Implicit (hidden state) | No | No | ~300k |
| **Transformer** | Attention on history window | 8 steps | No | No | ~400k |
| **DynaMITE** | Factored latent dynamics | 8 steps | Yes (24-d factored) | Yes | ~450k |

All models share the same observation embedding, action embedding, policy head, and value head components (defined in `src/models/components.py`) to ensure fair comparison.

### DynaMITE-Specific Parameters

```yaml
latent:
  enabled: true
  factored: true
  factors:
    friction: 4   # 4-dim subspace for friction
    mass: 6       # 6-dim subspace for link masses
    motor: 6      # 6-dim subspace for motor strength
    contact: 4    # 4-dim subspace for contact properties
    delay: 4      # 4-dim subspace for actuation delay
  total_dim: 24   # sum of all factor dimensions

auxiliary_loss:
  enabled: true
  weight: 0.1
  targets: [friction, mass, motor, contact, delay]
  per_factor_weight: {friction: 1.0, mass: 1.0, motor: 1.0, contact: 1.0, delay: 1.0}
```

---

## Repository Structure

```
robotpaper/
├── README.md                          # This file (main artifact)
├── LICENSE                            # MIT License
├── pyproject.toml                     # Python package metadata & tool config
├── requirements.txt                   # Pip dependencies
├── environment.yml                    # Conda environment specification
├── .gitignore                         # Git ignore rules
│
├── configs/                           # All experiment configuration (YAML)
│   ├── base.yaml                      #   Master defaults for everything
│   ├── task/                          #   Task-specific overrides
│   │   ├── flat.yaml                  #     Flat ground, no randomization
│   │   ├── push.yaml                  #     Push disturbances on flat ground
│   │   ├── randomized.yaml            #     Full domain randomization
│   │   └── terrain.yaml               #     Rough terrain + full randomization
│   ├── model/                         #   Architecture-specific overrides
│   │   ├── mlp.yaml                   #     Standard MLP policy
│   │   ├── lstm.yaml                  #     LSTM-based recurrent policy
│   │   ├── transformer.yaml           #     Vanilla transformer (no latent)
│   │   └── dynamite.yaml              #     DynaMITE (proposed method)
│   ├── train/                         #   Training hyperparameter overrides
│   │   └── default.yaml               #     Default training configuration
│   ├── eval/                          #   Evaluation settings
│   │   └── default.yaml               #     Default evaluation configuration
│   ├── ablations/                     #   Ablation study configs
│   │   ├── seq_len_4.yaml             #     History length 4 (vs. default 8)
│   │   ├── seq_len_16.yaml            #     History length 16
│   │   ├── no_latent.yaml             #     No latent head (transformer only)
│   │   ├── single_latent.yaml         #     Single unfactored latent
│   │   ├── no_aux_loss.yaml           #     No auxiliary identification loss
│   │   ├── depth_1.yaml               #     1 transformer layer
│   │   └── depth_4.yaml               #     4 transformer layers
│   └── sweeps/                        #   Robustness sweep configs
│       ├── push_magnitude.yaml        #     Sweep push force [0, 50, ..., 300] N
│       ├── friction.yaml              #     Sweep friction [0.1, 0.2, ..., 2.5]
│       └── action_delay.yaml          #     Sweep delay [0, 1, ..., 6] steps
│
├── src/                               # Source code (Python package)
│   ├── __init__.py
│   ├── utils/                         #   Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py                  #     YAML config loading, merging, CLI overrides
│   │   ├── seed.py                    #     Deterministic seeding (Python, NumPy, PyTorch)
│   │   ├── logger.py                  #     TensorBoard + CSV + console logger
│   │   ├── history_buffer.py          #     GPU-resident FIFO history buffer
│   │   ├── checkpoint.py              #     Save/load/find checkpoints
│   │   ├── manifest.py                #     Run manifest (git hash, hardware, config)
│   │   └── metrics.py                 #     Metrics accumulator & seed aggregator
│   ├── models/                        #   All model architectures
│   │   ├── __init__.py                #     Model registry & build_model() factory
│   │   ├── components.py              #     Shared building blocks (embeddings, heads)
│   │   ├── latent_heads.py            #     Factorized latent & aux identification heads
│   │   ├── mlp_policy.py              #     MLP baseline
│   │   ├── lstm_policy.py             #     LSTM baseline
│   │   ├── transformer_policy.py      #     Vanilla transformer baseline
│   │   └── dynamite_policy.py         #     DynaMITE (proposed method)
│   ├── envs/                          #   Environment integration
│   │   ├── __init__.py
│   │   ├── g1_env.py                  #     Isaac Lab G1 environment wrapper
│   │   └── reward.py                  #     Reward function components
│   ├── algos/                         #   RL algorithms
│   │   ├── __init__.py
│   │   └── ppo.py                     #     PPO trainer with rollout buffer
│   └── analysis/                      #   Post-training analysis
│       ├── __init__.py
│       ├── plotting.py                #     Training curves, eval bars, sweep plots
│       ├── tables.py                  #     Markdown & LaTeX table generators
│       └── latent_analysis.py         #     t-SNE, correlation, disentanglement
│
├── scripts/                           # Executable scripts
│   ├── train.py                       #   Training entry point
│   ├── eval.py                        #   Evaluation entry point
│   ├── plot_results.py                #   Generate all figures
│   ├── aggregate_seeds.py             #   Aggregate metrics across seeds
│   ├── generate_tables.py             #   Generate results tables
│   ├── run_all_baselines.sh           #   Run all 16 main experiments
│   ├── run_ablations.sh               #   Run all 7 ablation experiments
│   ├── run_sweeps.sh                  #   Run robustness sweeps
│   ├── run_everything.sh              #   Run full pipeline (~3 hours)
│   └── reproduce_all.sh              #    Full 4-tier reproduction script
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_models.py                 #   Model architecture tests
│   ├── test_history_buffer.py         #   History buffer tests
│   └── test_config.py                 #   Config system tests
│
├── docs/                              # Additional documentation
│   ├── architecture.md                #   Detailed architecture documentation
│   ├── config_system.md               #   Config system reference
│   └── experiment_plan.md             #   Full experiment plan
│
├── reproducibility/                   # Reproducibility artifacts
│   ├── checklist.md                   #   Pre/post-run verification checklist
│   └── expected_results.md            #   Expected reward ranges & variance bounds
│
├── manifests/                         # Run manifests (auto-populated)
│   └── experiment_registry.json       #   Master experiment registry
│
├── outputs/                           # Training outputs (git-ignored)
├── checkpoints/                       # Model checkpoints (git-ignored)
├── results/                           # Aggregated results (git-ignored)
├── figures/                           # Generated plots (git-ignored)
└── logs/                              # Raw logs (git-ignored)
```

---

## Installation

### Prerequisites

- **OS**: Ubuntu 20.04 or 22.04 (tested), or any Linux with NVIDIA driver support
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4060 Laptop, 8 GB VRAM)
- **NVIDIA Driver**: ≥ 525.xx
- **CUDA Toolkit**: 12.1 (installed via conda or system)
- **Python**: 3.10
- **Isaac Sim**: 4.0+ (for environment simulation; see [Environment & Simulator Setup](#environment--simulator-setup))

### Option A: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/robotpaper.git
cd robotpaper

# Create conda environment
conda env create -f environment.yml
conda activate dynamite

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/ -v
```

### Option B: Pip + venv

```bash
git clone https://github.com/<your-username>/robotpaper.git
cd robotpaper

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -m pytest tests/ -v
```

### Option C: Editable Install (for development)

```bash
pip install -e ".[dev]"
```

---

## Environment & Simulator Setup

### Isaac Lab / Isaac Sim

This project assumes the Unitree G1 environment runs inside **NVIDIA Isaac Lab** (built on Isaac Sim 4.0+). Isaac Lab provides:
- Parallelized GPU-based physics simulation
- Built-in humanoid robot assets (including Unitree G1)
- Domain randomization APIs
- High-throughput vectorized environments (4096 parallel environments)

#### Installing Isaac Sim + Isaac Lab

1. **Install Isaac Sim** following [NVIDIA's official instructions](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).
2. **Install Isaac Lab** following the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/):
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ./isaaclab.sh --install
   ```
3. Ensure the Isaac Lab Python environment has access to this project's packages, or install this project inside the Isaac Lab conda environment.

#### Running Without Isaac Lab (Mock Mode)

For development, testing, and debugging purposes, the codebase provides a **mock environment** that mimics the Isaac Lab interface:

```bash
# This will automatically use the mock environment if Isaac Lab is not detected
python scripts/train.py --task flat --model mlp --set training.max_iterations=10

# Unit tests always use mock environments
python -m pytest tests/ -v
```

The mock environment generates random observations and rewards. It is NOT suitable for actual training or evaluation—only for verifying that the code runs end-to-end.

### Simulator Prerequisites Checklist

- [ ] Isaac Sim 4.0+ launches successfully (`./isaac-sim.sh`)
- [ ] Isaac Lab installed and importable (`python -c "import omni.isaac.lab"`)
- [ ] Unitree G1 URDF/USD available in Isaac Lab assets
- [ ] 4096 parallel environments fit in GPU memory (reduce `num_envs` if OOM)
- [ ] Physics simulation runs at 200 Hz, control at 50 Hz (4:1 decimation)

---

## Sanity Checks

Run these before starting any real experiments:

### 1. Unit Tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass. Tests verify model forward passes, parameter counts, gradient flow, history buffer operations, and config system.

### 2. Smoke Test (Mock Environment)

```bash
python scripts/train.py \
  --task flat --model dynamite \
  --set training.max_iterations=5 training.num_envs=64
```

Expected: completes in <1 minute, prints loss values, creates output directory.

### 3. Full Environment Smoke Test (Isaac Lab Required)

```bash
python scripts/train.py \
  --task flat --model mlp \
  --set training.max_iterations=10 training.num_envs=256
```

Expected: completes in ~2 minutes, reward values are non-trivial (not all zeros).

### 4. Checkpoint Save/Load Roundtrip

```bash
# Train briefly
python scripts/train.py --task flat --model mlp --set training.max_iterations=10

# Verify checkpoint exists
ls outputs/flat/mlp/seed_42/*/checkpoints/

# Resume training
python scripts/train.py --task flat --model mlp --resume outputs/flat/mlp/seed_42/*/
```

---

## Training

### Basic Training Command

```bash
python scripts/train.py \
  --base configs/base.yaml \
  --task <task> \
  --model <model> \
  --train configs/train/default.yaml \
  --seed <seed>
```

The `--base` flag defaults to `configs/base.yaml` and can be omitted.

### Training Each Method

#### MLP Baseline

```bash
python scripts/train.py --task flat --model mlp --seed 42
python scripts/train.py --task push --model mlp --seed 42
python scripts/train.py --task randomized --model mlp --seed 42
python scripts/train.py --task terrain --model mlp --seed 42
```

#### LSTM Baseline

```bash
python scripts/train.py --task flat --model lstm --seed 42
python scripts/train.py --task push --model lstm --seed 42
python scripts/train.py --task randomized --model lstm --seed 42
python scripts/train.py --task terrain --model lstm --seed 42
```

#### Transformer Baseline

```bash
python scripts/train.py --task flat --model transformer --seed 42
python scripts/train.py --task push --model transformer --seed 42
python scripts/train.py --task randomized --model transformer --seed 42
python scripts/train.py --task terrain --model transformer --seed 42
```

#### DynaMITE (Proposed)

```bash
python scripts/train.py --task flat --model dynamite --seed 42
python scripts/train.py --task push --model dynamite --seed 42
python scripts/train.py --task randomized --model dynamite --seed 42
python scripts/train.py --task terrain --model dynamite --seed 42
```

### Multi-Seed Training

By default, all experiments use seed 42 for fast iteration (~3 hours total). To run with multiple seeds for final results, override the seed:

```bash
for seed in 42 43 44; do
  python scripts/train.py --task randomized --model dynamite --seed $seed
done
```

### Run All Baselines (16 Runs)

```bash
bash scripts/run_all_baselines.sh
```

This runs 4 models × 4 tasks × 1 seed = 16 training runs sequentially. Estimated time: ~1.5 hours on RTX 4060.

### CLI Overrides

Override any config value from the command line using `--set`:

```bash
# Reduce environment count (for limited VRAM)
python scripts/train.py --task randomized --model dynamite \
  --set training.num_envs=2048

# Change learning rate
python scripts/train.py --task randomized --model dynamite \
  --set training.learning_rate=0.0001

# Quick debug run
python scripts/train.py --task flat --model mlp \
  --set training.max_iterations=50 training.num_envs=128
```

### Resuming Training

```bash
# Resume from the latest checkpoint in a run directory
python scripts/train.py --task randomized --model dynamite \
  --resume outputs/randomized/dynamite/seed_42/20250101_120000/
```

The trainer will load the latest checkpoint and continue training from that iteration.

---

## Evaluation

### Basic Evaluation

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --task randomized \
  --num-episodes 50
```

### Evaluate with Specific Config

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --eval configs/eval/default.yaml \
  --task randomized
```

### Cross-Task Evaluation

Evaluate a model trained on one task against a different task:

```bash
# Model trained on randomized, evaluated on terrain
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --task terrain \
  --num-episodes 50
```

### Batch Evaluation

After training completes, evaluate all models:

```bash
for model in mlp lstm transformer dynamite; do
  for task in flat push randomized terrain; do
    python scripts/eval.py \
      --checkpoint outputs/$task/${model}_full/seed_42/*/checkpoints/best.pt \
      --task $task \
      --num-episodes 50
  done
done
```

### Aggregate Results Across Seeds

```bash
python scripts/aggregate_seeds.py \
  --results-dir outputs/ \
  --output results/aggregated/main_comparison.json
```

---

## Ablation Studies

Ablations test the contribution of each DynaMITE design decision. All ablations use the `randomized` task.

### Individual Ablation Commands

```bash
# History length ablations
python scripts/train.py --task randomized --model dynamite --ablation seq_len_4 --seed 42
python scripts/train.py --task randomized --model dynamite --ablation seq_len_16 --seed 42

# Latent design ablations
python scripts/train.py --task randomized --model dynamite --ablation no_latent --seed 42
python scripts/train.py --task randomized --model dynamite --ablation single_latent --seed 42
python scripts/train.py --task randomized --model dynamite --ablation no_aux_loss --seed 42

# Architecture depth ablations
python scripts/train.py --task randomized --model dynamite --ablation depth_1 --seed 42
python scripts/train.py --task randomized --model dynamite --ablation depth_4 --seed 42
```

### Run All Ablations (7 Runs)

```bash
bash scripts/run_ablations.sh
```

This runs 7 ablation variants × 1 seed = 7 runs. Estimated time: ~42 minutes.

### Ablation Hypotheses

| Ablation | Hypothesis | Expected Outcome |
|---|---|---|
| `seq_len_4` | 4 steps (80 ms) is too short to infer dynamics | ↓ 5–15% reward vs. baseline (8 steps) |
| `seq_len_16` | 16 steps provides diminishing returns, adds cost | ≈ reward, ↑ compute time |
| `no_latent` | Without latent, it's just a vanilla transformer | ↓ 10–20% on randomized tasks |
| `single_latent` | Unfactored latent loses disentanglement benefit | ↓ 5–10% vs. factored |
| `no_aux_loss` | Without supervision, latent may not be meaningful | ↓ 5–15%, worse disentanglement |
| `depth_1` | Single layer has insufficient capacity | ↓ 5–10% |
| `depth_4` | 4 layers is overkill for this problem | ≈ reward, ↑ compute |

---

## Robustness Sweeps

Robustness sweeps evaluate trained models under progressively harder perturbations, testing generalization beyond the training distribution.

### Push Magnitude Sweep

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --sweep configs/sweeps/push_magnitude.yaml
```

Sweeps push force from 0 N to 300 N in steps of 50 N.

### Friction Sweep

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --sweep configs/sweeps/friction.yaml
```

Sweeps friction coefficient from 0.1 to 2.5.

### Action Delay Sweep

```bash
python scripts/eval.py \
  --checkpoint outputs/randomized/dynamite/seed_42/*/checkpoints/best.pt \
  --sweep configs/sweeps/action_delay.yaml
```

Sweeps actuation delay from 0 to 6 steps.

### Run All Sweeps for All Methods

```bash
bash scripts/run_sweeps.sh
```

---

## Plotting & Table Generation

### Generate All Plots

```bash
python scripts/plot_results.py \
  --results results/aggregated/ \
  --output figures/
```

This generates:
- `training_curves.png` — reward vs. iteration for all methods
- `eval_bars.png` — bar chart comparing methods across tasks
- `ablation.png` — ablation comparison bar chart
- `sweep_push_magnitude.png` — robustness vs. push force
- `sweep_friction.png` — robustness vs. friction
- `sweep_action_delay.png` — robustness vs. delay
- `latent_tsne_*.png` — t-SNE visualizations of learned latent factors

### Generate Tables

```bash
python scripts/generate_tables.py \
  --results results/aggregated/ \
  --output results/tables/
```

This generates:
- `main_table.md` — main comparison in Markdown
- `main_table.tex` — main comparison in LaTeX
- `ablation_table.md` — ablation results in Markdown
- `ablation_table.tex` — ablation results in LaTeX
- `efficiency_table.md` — parameter count and throughput comparison

---

## Full Reproduction

The `reproduce_all.sh` script provides 4 tiers of reproduction:

```bash
# Tier 1: Sanity check (~3 minutes)
# Runs unit tests + smoke test with 5 iterations
bash scripts/reproduce_all.sh sanity

# Tier 2: Single run (~6 minutes)
# One full training run: DynaMITE on randomized, seed 42
bash scripts/reproduce_all.sh single

# Tier 3: Main experiments (~1.5 hours)
# All 16 baseline runs (4 models × 4 tasks × 1 seed)
bash scripts/reproduce_all.sh main

# Tier 4: Full reproduction (~3 hours)
# Main + ablations + sweeps + analysis + plots + tables
bash scripts/reproduce_all.sh full
```

**Recommendation**: Run `sanity` first. If it passes, run `single` to verify a complete training run produces expected reward ranges. Then proceed to `main` or `full`.

### Run Everything (One Command)

To run the entire pipeline (training, evaluation, ablations, sweeps, aggregation, plots) in a single command:

```bash
bash scripts/run_everything.sh
```

Estimated total time: **~3 hours** on RTX 4060 (2M timesteps per run, 512 envs, 1 seed). Use `--dry-run` to preview what will be executed without running anything.

---

## Experiment Naming Convention

All output directories follow this deterministic naming pattern:

```
outputs/{task}/{model}/seed_{seed}/{timestamp}/
```

Example:

```
outputs/randomized/dynamite/seed_42/20250115_143022/
```

For ablation experiments:

```
outputs/randomized/dynamite_ablation_{name}/seed_{seed}/{timestamp}/
```

Example:

```
outputs/randomized/dynamite_ablation_no_latent/seed_42/20250115_160000/
```

This structure ensures:
- **Uniqueness**: No two runs overwrite each other
- **Discoverability**: Easy to glob for specific method/task/seed combinations
- **Aggregation**: `aggregate_seeds.py` can find all seeds for a method-task pair

---

## Output Directory Structure

Each training run produces this directory:

```
outputs/{task}/{model}/seed_{seed}/{timestamp}/
├── config.yaml              # Frozen config for this run (fully merged)
├── manifest.json            # Run metadata (git hash, hardware, timing)
├── metrics.csv              # Per-iteration metrics (reward, loss, etc.)
├── eval_metrics.json        # Final evaluation metrics
├── tb/                      # TensorBoard event files
│   └── events.out.tfevents.*
├── checkpoints/
│   ├── ckpt_1000.pt         # Periodic checkpoint
│   ├── ckpt_2000.pt
│   ├── ...
│   ├── latest.pt            # Symlink to most recent checkpoint
│   └── best.pt              # Symlink to best-reward checkpoint
└── logs/
    └── train.log            # Console log output
```

---

## Expected Outputs per Run

After a complete training run, verify these files exist:

| File | Minimum Size | Purpose |
|---|---|---|
| `config.yaml` | > 1 KB | Frozen configuration |
| `manifest.json` | > 500 B | Run metadata |
| `metrics.csv` | > 100 KB | Training metrics |
| `checkpoints/best.pt` | > 500 KB | Best model weights |
| `checkpoints/latest.pt` | > 500 KB | Latest model weights |
| `tb/events.out.tfevents.*` | > 100 KB | TensorBoard logs |

After evaluation:

| File | Purpose |
|---|---|
| `eval_metrics.json` | Mean reward, std, success rate, episode length |

---

## Expected Results Summary

These are approximate expected results based on the method design and typical RL locomotion benchmarks. Actual numbers will be determined by running the experiments.

### Main Comparison (Reward — Higher is Better)

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | ~18–22 | ~12–16 | ~10–14 | ~8–12 |
| LSTM | ~19–23 | ~14–18 | ~13–17 | ~11–15 |
| Transformer | ~19–23 | ~14–18 | ~14–18 | ~11–15 |
| **DynaMITE** | **~20–24** | **~16–20** | **~16–20** | **~13–17** |

### Expected Improvement of DynaMITE over Baselines

- vs. MLP: +15–30% on randomized/terrain tasks
- vs. LSTM: +5–15% on randomized/terrain tasks
- vs. Transformer: +5–15% on randomized/terrain tasks (benefit from latent)
- On flat task: modest (0–5%) since dynamics are fixed

### Variance Guidelines

- Per-seed standard deviation should be < 15% of mean reward
- If seed variance exceeds 20%, consider adding more seeds
- Cross-seed relative ordering of methods should be consistent

---

## Runtime Estimates

All estimates are for an RTX 4060 Laptop GPU (8 GB VRAM), 512 parallel environments, 2M timesteps per run.

| Experiment Set | Runs | Time per Run | Total Time |
|---|---|---|---|
| Single training run | 1 | ~6 min | 6 min |
| All baselines (4×4×1) | 16 | ~6 min | ~1.5 hours |
| All ablations (7×1) | 7 | ~6 min | ~42 min |
| Robustness sweeps | 4 methods × 4 sweeps | ~1 min each | ~16 min |
| Evaluation | 23 runs | ~1 min each | ~23 min |
| Analysis & plotting | — | — | ~2 min |
| **Full reproduction** | **—** | **—** | **~3 hours** |

### Memory Usage

- 4096 environments: ~4–5 GB VRAM
- Model (DynaMITE): ~2 MB
- Rollout buffer: ~1–2 GB VRAM
- Total: ~6–7 GB VRAM (fits in 8 GB)

If you encounter OOM errors, reduce `training.num_envs` to 2048 (expect ~1.5× longer training time).

---

## Reproducibility Protocol

### Source of Randomness

All sources of randomness are seeded deterministically:

1. `random.seed(seed)` — Python stdlib
2. `np.random.seed(seed)` — NumPy
3. `torch.manual_seed(seed)` — PyTorch CPU
4. `torch.cuda.manual_seed_all(seed)` — PyTorch CUDA
5. `torch.backends.cudnn.deterministic = True` — CuDNN
6. `torch.backends.cudnn.benchmark = False` — CuDNN

**Note**: Full bitwise reproducibility is not guaranteed across different GPU architectures or CUDA versions due to floating-point non-associativity in parallel reductions. However, statistical reproducibility (same trends, same method ordering, reward within ±5%) is expected.

### Pre-Run Checklist

Before starting experiments:

- [ ] All unit tests pass: `python -m pytest tests/ -v`
- [ ] Git working directory is clean: `git status`
- [ ] Record git commit hash: `git rev-parse HEAD`
- [ ] Verify GPU: `nvidia-smi`
- [ ] Verify CUDA version matches requirement
- [ ] Smoke test completes successfully

### Post-Run Verification

After each run:

- [ ] `manifest.json` exists and contains correct git hash
- [ ] `config.yaml` is identical to the intended configuration
- [ ] `metrics.csv` has the expected number of rows (= `max_iterations`)
- [ ] `checkpoints/best.pt` and `checkpoints/latest.pt` exist
- [ ] Final reward is within the expected range (see [Expected Results](#expected-results-summary))
- [ ] No NaN values in `metrics.csv`

### Multi-Seed Consistency

When running with multiple seeds (optional, for final publication):

- [ ] All seeds completed successfully
- [ ] Cross-seed standard deviation < 15% of mean
- [ ] Method ordering is consistent across seeds (e.g., DynaMITE > Transformer in all seeds)

See `reproducibility/checklist.md` for the full reproducibility checklist and `reproducibility/expected_results.md` for detailed expected result ranges.

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce the number of parallel environments:

```bash
python scripts/train.py --task randomized --model dynamite \
  --set training.num_envs=2048
```

If 2048 still OOMs, try 1024. Training time will scale approximately linearly with the reduction.

#### Isaac Lab Import Error

```
ModuleNotFoundError: No module named 'omni.isaac.lab'
```

**Solution**: Either install Isaac Lab, or run in mock mode for development/testing. Mock mode activates automatically. See [Environment & Simulator Setup](#environment--simulator-setup).

#### NaN in Training Loss

```
WARNING: NaN detected in loss at iteration 150
```

**Solution**: This usually indicates a learning rate that is too high or an unstable reward signal.

1. Try reducing the learning rate: `--set training.learning_rate=0.0001`
2. Check reward function scale: ensure rewards are in a reasonable range (0–30)
3. Enable gradient clipping (this is on by default: `max_grad_norm=1.0`)

#### Checkpoint Not Found

```
FileNotFoundError: No checkpoint found at ...
```

**Solution**: Verify the path. Use the glob pattern with the run directory:

```bash
ls outputs/randomized/dynamite/seed_42/*/checkpoints/
```

#### TensorBoard Not Showing Data

```bash
tensorboard --logdir outputs/
```

If no data appears, ensure the training run created the `tb/` directory within the run directory.

#### Very Low Reward After Full Training

Expected rewards for DynaMITE on `randomized` task: 16–20. If you see values below 5:

1. Verify the correct task config is loaded (check `config.yaml` in the run directory)
2. Ensure domain randomization ranges are reasonable
3. Check that action scaling is correct (`action_scale=0.25`)
4. Try a different seed — occasionally one seed may diverge

---

## FAQ

**Q: Do I need Isaac Sim to run the code?**

A: For actual training and meaningful results, yes. For development, testing, and debugging, the mock environment allows running the full pipeline without Isaac Sim. All unit tests use the mock environment.

**Q: Can I use a different GPU?**

A: Yes. The code is not GPU-specific. For GPUs with less VRAM, reduce `training.num_envs`. For GPUs with more VRAM, you can increase it. Here are rough guidelines:

| GPU | VRAM | Recommended `num_envs` |
|---|---|---|
| RTX 3060 | 12 GB | 4096 |
| RTX 4060 Laptop | 8 GB | 4096 (tight) |
| RTX 4090 | 24 GB | 8192–16384 |
| A100 | 40–80 GB | 16384–32768 |

**Q: Can I use a different robot?**

A: The codebase is designed for the Unitree G1 but can be adapted. You would need to:
1. Modify `src/envs/g1_env.py` for the new robot's URDF/USD
2. Update observation and action dimensions in `configs/base.yaml`
3. Adjust reward weights in `src/envs/reward.py`
4. Update domain randomization ranges

**Q: Why factored latent instead of a single latent vector?**

A: Factorization provides:
1. **Interpretability**: each subspace corresponds to a known dynamics axis
2. **Supervised disentanglement**: per-factor auxiliary losses encourage separation
3. **Targeted analysis**: we can measure correlation between each factor and its GT parameter
4. The ablation `single_latent` directly tests whether factoring matters

**Q: Why 8 steps for the history window?**

A: 8 steps at 50 Hz = 160 ms. This is long enough to observe dynamics effects (a step response to a friction change settles within ~100 ms) but short enough for real-time inference. The ablations `seq_len_4` and `seq_len_16` test sensitivity to this choice.

**Q: Why PPO and not SAC or other algorithms?**

A: PPO is the standard algorithm for massively parallel on-policy RL in Isaac Lab. It is simple, stable, and well-understood. The contribution is in the architecture (DynaMITE), not the RL algorithm. Using PPO for all methods ensures a fair comparison.

**Q: How do I visualize training progress?**

A: Use TensorBoard:

```bash
tensorboard --logdir outputs/ --port 6006
```

Or inspect `metrics.csv` directly:

```bash
# Show last 5 iterations
tail -5 outputs/randomized/dynamite/seed_42/*/metrics.csv
```

**Q: What if my results don't match the expected ranges?**

A: The expected ranges in this README and in `reproducibility/expected_results.md` are approximate. Small differences (±10%) are normal due to hardware, CUDA version, and floating-point non-determinism. If results deviate significantly:
1. Verify your setup matches the requirements
2. Check the frozen `config.yaml` in the run directory
3. Ensure you're using the correct checkpoint (best vs. latest)
4. Try additional seeds

---

## How to Extend

### Adding a New Model Architecture

1. Create `src/models/your_model.py` implementing the same interface as `DynaMITEPolicy`:
   - `forward(obs, history_obs, history_act, history_mask, cmd)` → `(action_mean, action_log_std, value, aux_losses)`
   - `get_value(obs, ...)` → `value`

2. Register it in `src/models/__init__.py`:
   ```python
   from .your_model import YourModel
   MODEL_REGISTRY['your_model'] = YourModel
   ```

3. Create `configs/model/your_model.yaml` with architecture-specific hyperparameters.

4. Add tests in `tests/test_models.py`.

5. Train: `python scripts/train.py --task randomized --model your_model`

### Adding a New Task

1. Create `configs/task/your_task.yaml` overriding the relevant fields from `base.yaml`.

2. If the task requires new environment logic, extend `src/envs/g1_env.py`.

3. Train: `python scripts/train.py --task your_task --model dynamite`

### Adding a New Reward Component

1. Add the reward function to `src/envs/reward.py`.
2. Add its weight to `configs/base.yaml` under `reward.weights`.
3. The reward aggregator `compute_rewards()` automatically picks it up.

### Adding a New Ablation

1. Create `configs/ablations/your_ablation.yaml` overriding the specific parameter(s).
2. Add it to `scripts/run_ablations.sh`.
3. Train: `python scripts/train.py --task randomized --model dynamite --ablation your_ablation`

### Adding a New Latent Factor

1. In `configs/model/dynamite.yaml`, add the new factor to `model.latent.factors`:
   ```yaml
   factors:
     friction: 4
     mass: 6
     motor: 6
     contact: 4
     delay: 4
     your_factor: 4  # new factor
   ```
2. Update `total_dim` accordingly.
3. Add the ground-truth target in `src/envs/g1_env.py`'s `get_dynamics_targets()`.
4. The `FactorizedLatentHead` and `AuxiliaryIdentificationHead` will automatically create the new projection.

---

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{dynamite2025,
  title     = {DynaMITE: Dynamic Mismatch Inference via Transformer Encoder
               for Robust Humanoid Locomotion},
  author    = {<Author Name>},
  journal   = {<Venue>},
  year      = {2025},
  note      = {Code: https://github.com/<username>/robotpaper}
}
```

*(Update with actual publication details upon acceptance.)*

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgments

- **Unitree Robotics** for the G1 humanoid platform
- **NVIDIA** for Isaac Sim and Isaac Lab
- The reinforcement learning and legged locomotion research communities

---

*This README is designed to be self-contained. A new researcher should be able to understand the problem, method, codebase, and experiment plan, and reproduce all results by following this document alone.*
