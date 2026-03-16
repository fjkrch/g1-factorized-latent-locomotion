# Configuration System Reference

## Overview

The DynaMITE project uses a **layered YAML configuration system** that provides:

1. **Single source of truth** — `configs/base.yaml` defines every parameter with its default value
2. **Composable overrides** — task, model, training, eval, and ablation configs override specific subsets
3. **CLI overrides** — `--set key=value` for one-off changes
4. **Frozen configs** — every run saves its fully-merged config as `config.yaml` in the output directory

---

## Config Merge Order

Configs are merged left-to-right, with later values overriding earlier ones:

```
base.yaml → task/{task}.yaml → model/{model}.yaml → train/{train}.yaml → eval/{eval}.yaml → ablations/{ablation}.yaml → CLI --set overrides
```

### Merge Semantics

- **Scalars**: later value replaces earlier value
- **Dicts**: deep-merged recursively (nested keys are merged, not replaced)
- **Lists**: later value replaces entire list (no list merging)

### Example

```yaml
# base.yaml
training:
  learning_rate: 0.0003
  num_envs: 4096
  max_iterations: 3000

# configs/model/dynamite.yaml
model:
  type: dynamite
  latent:
    enabled: true

# CLI: --set training.learning_rate=0.0001
```

Result:
```yaml
training:
  learning_rate: 0.0001     # CLI override
  num_envs: 4096             # from base
  max_iterations: 3000       # from base
model:
  type: dynamite             # from model config
  latent:
    enabled: true            # from model config
```

---

## Config File Reference

### `configs/base.yaml` — Master Defaults

This is the single source of truth. Every configurable parameter appears here with its default value.

#### Top-Level Sections

| Section | Purpose |
|---|---|
| `task` | Task name, robot, simulator settings |
| `observation` | Observation space definition |
| `action` | Action space definition |
| `command` | Velocity command ranges and resampling |
| `reward` | Reward component weights |
| `termination` | Episode termination conditions |
| `domain_randomization` | DR parameter ranges |
| `training` | PPO hyperparameters, env count, iterations |
| `model` | Architecture selection and hyperparameters |
| `evaluation` | Eval frequency, episode count |
| `output` | Output directory conventions |
| `seed` | Default seed value |

#### Key Parameters

```yaml
# Core training
training.learning_rate: 0.0003       # Adam LR
training.num_envs: 4096              # Parallel environments
training.max_iterations: 3000        # Total training iterations
training.steps_per_iteration: 24     # Steps per env per PPO update
training.num_epochs: 5               # PPO epochs per update
training.mini_batch_size: 4096       # Mini-batch size
training.gamma: 0.99                 # Discount factor
training.gae_lambda: 0.95            # GAE lambda
training.clip_ratio: 0.2             # PPO clip parameter
training.value_loss_coef: 1.0        # Value loss weight
training.entropy_coef: 0.01          # Entropy bonus weight
training.max_grad_norm: 1.0          # Gradient clipping

# Model
model.type: mlp                      # Architecture: mlp|lstm|transformer|dynamite
model.obs_embed_dim: 64              # Observation embedding dimension
model.act_embed_dim: 64              # Action embedding dimension
model.hidden_dims: [256, 128]        # Policy/value head hidden layers

# History (for transformer/dynamite)
model.history_len: 8                 # History buffer length
model.d_model: 128                   # Transformer model dimension
model.nhead: 4                       # Attention heads
model.num_layers: 2                  # Transformer layers
model.dim_feedforward: 256           # FFN dimension

# Latent (for dynamite)
model.latent.enabled: false          # Latent head on/off
model.latent.factored: false         # Factored vs. single latent

# Evaluation
evaluation.num_episodes: 100         # Episodes per evaluation
evaluation.eval_interval: 100        # Evaluate every N iterations
```

### `configs/task/*.yaml` — Task Overrides

Each task config sets:
- `task.name` — task identifier
- `domain_randomization.*` — which DR is enabled and ranges
- `task.push.*` — push disturbance settings
- `task.terrain.*` — terrain type

**Hierarchy**: `flat` (simplest) → `push` → `randomized` → `terrain` (hardest)

### `configs/model/*.yaml` — Model Overrides

Each model config sets:
- `model.type` — architecture name
- `model.history_len` — history buffer length (0 for MLP)
- Model-specific parameters (LSTM hidden size, transformer heads/layers, latent config)

### `configs/ablations/*.yaml` — Ablation Overrides

Each ablation config overrides **one or two** parameters to isolate a design decision:

| Ablation | Parameter Changed |
|---|---|
| `seq_len_4` | `model.history_len: 4` |
| `seq_len_16` | `model.history_len: 16` |
| `no_latent` | `model.latent.enabled: false` |
| `single_latent` | `model.latent.factored: false` |
| `no_aux_loss` | `model.auxiliary_loss.enabled: false` |
| `depth_1` | `model.num_layers: 1` |
| `depth_4` | `model.num_layers: 4` |

### `configs/sweeps/*.yaml` — Robustness Sweep Configs

Each sweep config defines:
- `sweep.parameter` — which parameter to sweep
- `sweep.values` — list of values to evaluate at
- `sweep.override_path` — which config key to override

---

## CLI Override Syntax

The `--set` flag allows overriding any config value from the command line:

```bash
python scripts/train.py --task randomized --model dynamite \
  --set training.learning_rate=0.0001 \
  --set training.num_envs=2048 \
  --set model.num_layers=3
```

### Supported Types

- **Integers**: `--set training.num_envs=2048` → `int`
- **Floats**: `--set training.learning_rate=0.0001` → `float`
- **Booleans**: `--set model.latent.enabled=true` → `bool`
- **Strings**: `--set model.type=dynamite` → `str`

Nested keys use dot notation: `training.learning_rate` → `config['training']['learning_rate']`.

---

## Config Loading API (`src/utils/config.py`)

### `load_config(path: str) -> dict`

Load a single YAML file and return as a dictionary.

### `deep_merge(base: dict, override: dict) -> dict`

Recursively merge `override` into `base`. Returns a new dict (does not mutate inputs).

### `apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict`

Apply `key=value` overrides from the CLI. Automatically converts values to appropriate Python types.

### `make_run_dir(cfg: dict) -> str`

Create and return the output directory path following the naming convention:

```
outputs/{task}/{model}/seed_{seed}/{YYYYMMDD_HHMMSS}/
```

### `save_config(cfg: dict, path: str)`

Save the fully-merged config as YAML to the specified path. This is the "frozen config" for reproducibility.

---

## Best Practices

1. **Never modify `base.yaml` for a specific experiment** — create an override config instead
2. **Always check the frozen `config.yaml`** in the output directory to verify what was actually used
3. **Use `--set` for quick experiments** but create a proper config file for anything you plan to reference later
4. **Keep ablation configs minimal** — override only what you're testing, rely on defaults for everything else
5. **Document new parameters** — if you add a parameter to `base.yaml`, add a comment explaining it
