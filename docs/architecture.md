# DynaMITE Architecture Documentation

## Overview

DynaMITE (Dynamic Mismatch Inference via Transformer Encoder) is a policy architecture for robust humanoid locomotion under unknown dynamics. It extends a standard transformer encoder policy with two novel components:

1. **Factorized Latent Head** — projects the transformer's aggregated representation into semantically meaningful latent subspaces
2. **Auxiliary Identification Head** — supervises each latent subspace against ground-truth dynamics parameters during training

This document provides a detailed description of each architectural component, tensor shapes, and design rationale.

---

## Shared Components (`src/models/components.py`)

All four model architectures share these building blocks to ensure fair comparison.

### ObsEmbedding

Projects raw observations into a shared embedding space.

```
Input:  obs ∈ ℝ^(B, obs_dim)           # B = batch, obs_dim = 48
Output: obs_emb ∈ ℝ^(B, embed_dim)     # embed_dim = 64
```

Architecture: `Linear(obs_dim, embed_dim) → ELU`.

### ActEmbedding

Projects raw actions into the same embedding space.

```
Input:  act ∈ ℝ^(B, act_dim)           # act_dim = 19
Output: act_emb ∈ ℝ^(B, embed_dim)     # embed_dim = 64
```

Architecture: `Linear(act_dim, embed_dim) → ELU`.

### CmdEmbedding

Projects velocity commands into a command embedding.

```
Input:  cmd ∈ ℝ^(B, cmd_dim)           # cmd_dim = 3
Output: cmd_emb ∈ ℝ^(B, embed_dim)     # embed_dim = 64
```

Architecture: `Linear(cmd_dim, embed_dim) → ELU`.

### PolicyHead

Maps a conditioning vector to action mean and learned log-std.

```
Input:  x ∈ ℝ^(B, input_dim)
Output: mean ∈ ℝ^(B, act_dim), log_std ∈ ℝ^(B, act_dim)
```

Architecture: `MLP(input_dim → 256 → 128) → Linear(128, act_dim)`.
Log-std is a **learned parameter** (not input-dependent), clamped to [-5, 2].

### ValueHead

Maps a conditioning vector to a scalar value estimate.

```
Input:  x ∈ ℝ^(B, input_dim)
Output: value ∈ ℝ^(B, 1)
```

Architecture: `MLP(input_dim → 256 → 128) → Linear(128, 1)`.

### SinusoidalPE

Standard sinusoidal positional encoding (Vaswani et al., 2017).

```
Input:  seq_len (int), d_model (int)
Output: pe ∈ ℝ^(1, seq_len, d_model)    # registered buffer, not learned
```

### RunningNormalizer

Online running mean/variance normalizer for observations.

```
Input:  x ∈ ℝ^(B, dim)
Output: normalized_x ∈ ℝ^(B, dim)
```

Uses Welford's online algorithm. Clips normalized values to [-5, 5].

---

## MLP Policy (`src/models/mlp_policy.py`)

The simplest baseline. Processes only the current observation (no history).

### Forward Pass

```
obs ∈ ℝ^(B, 48)
    ↓ ObsEmbedding
obs_emb ∈ ℝ^(B, 64)
    ↓ CmdEmbedding (cmd → ℝ^(B, 64))
    ↓ concat([obs_emb, cmd_emb])
x ∈ ℝ^(B, 128)
    ↓ MLP(128 → 256 → 256 → 128)
h ∈ ℝ^(B, 128)
    ├── PolicyHead → mean ∈ ℝ^(B, 19), log_std ∈ ℝ^(B, 19)
    └── ValueHead  → value ∈ ℝ^(B, 1)
```

**Parameters**: ~200k
**History**: None
**Latent**: None

---

## LSTM Policy (`src/models/lstm_policy.py`)

Recurrent baseline with implicit memory through hidden state.

### Forward Pass (Single Step)

```
obs ∈ ℝ^(B, 48), hidden ∈ (h ∈ ℝ^(1, B, 128), c ∈ ℝ^(1, B, 128))
    ↓ ObsEmbedding
obs_emb ∈ ℝ^(B, 64)
    ↓ CmdEmbedding
    ↓ concat([obs_emb, cmd_emb])
x ∈ ℝ^(B, 128) → unsqueeze(1) → x ∈ ℝ^(B, 1, 128)
    ↓ LSTM(input_size=128, hidden_size=128, num_layers=1)
lstm_out ∈ ℝ^(B, 1, 128), new_hidden
    ↓ squeeze(1)
h ∈ ℝ^(B, 128)
    ├── PolicyHead → mean, log_std
    └── ValueHead  → value
```

**Parameters**: ~300k
**History**: Implicit via hidden state (theoretically infinite, practically limited)
**Latent**: None
**Note**: Hidden state must be reset at episode boundaries.

---

## Transformer Policy (`src/models/transformer_policy.py`)

Vanilla transformer encoder processing an explicit observation–action history window.

### Forward Pass

```
history_obs ∈ ℝ^(B, T, 48), history_act ∈ ℝ^(B, T, 19)    # T = history_len = 8
    ↓ ObsEmbedding (per timestep)
obs_embs ∈ ℝ^(B, T, 64)
    ↓ ActEmbedding (per timestep)
act_embs ∈ ℝ^(B, T, 64)
    ↓ concat([obs_embs, act_embs], dim=-1)
tokens ∈ ℝ^(B, T, 128)
    ↓ + SinusoidalPE(T, 128)
tokens ∈ ℝ^(B, T, 128)
    ↓ TransformerEncoder(d_model=128, nhead=4, num_layers=2, dim_ff=256)
enc_out ∈ ℝ^(B, T, 128)
    ↓ Mean pooling (with mask for padding)
h ∈ ℝ^(B, 128)
    ↓ concat([h, obs_emb_current, cmd_emb])
x ∈ ℝ^(B, 128+64+64) = ℝ^(B, 256)
    ├── PolicyHead(256 → ...) → mean, log_std
    └── ValueHead(256 → ...)  → value
```

**Parameters**: ~400k
**History**: Explicit, 8 steps
**Latent**: None

---

## DynaMITE Policy (`src/models/dynamite_policy.py`)

The proposed method. Extends the Transformer Policy with factorized latent dynamics inference.

### Forward Pass

```
history_obs ∈ ℝ^(B, T, 48), history_act ∈ ℝ^(B, T, 19)
    ↓ [Same as Transformer Policy up to mean pooling]
h ∈ ℝ^(B, 128)
    ↓ FactorizedLatentHead
z ∈ ℝ^(B, 24), z_factors = {
    'friction': ℝ^(B, 4),
    'mass':     ℝ^(B, 6),
    'motor':    ℝ^(B, 6),
    'contact':  ℝ^(B, 4),
    'delay':    ℝ^(B, 4)
}
    ↓ concat([obs_emb_current, cmd_emb, z])
x ∈ ℝ^(B, 64+64+24) = ℝ^(B, 152)
    ├── PolicyHead(152 → ...) → mean ∈ ℝ^(B, 19), log_std ∈ ℝ^(B, 19)
    └── ValueHead(152 → ...)  → value ∈ ℝ^(B, 1)

    ↓ [Training only] AuxiliaryIdentificationHead
aux_losses = {
    'friction': MSE(predict(z_friction), gt_friction),
    'mass':     MSE(predict(z_mass),     gt_mass),
    'motor':    MSE(predict(z_motor),    gt_motor),
    'contact':  MSE(predict(z_contact),  gt_contact),
    'delay':    MSE(predict(z_delay),    gt_delay)
}
```

**Parameters**: ~450k
**History**: Explicit, 8 steps
**Latent**: Factored, 24-dimensional (5 subspaces)
**Aux Loss**: Per-factor MSE identification loss

### Factorized Latent Head (`src/models/latent_heads.py`)

```python
class FactorizedLatentHead(nn.Module):
    # For each factor f with dimension d_f:
    #   z_f = Linear(input_dim, d_f)(h)
    # z = concat([z_f for f in factors])
```

Each factor has its own independent linear projection. This ensures the latent subspaces are structurally separated (though the auxiliary loss is needed to give them semantic meaning).

### Auxiliary Identification Head (`src/models/latent_heads.py`)

```python
class AuxiliaryIdentificationHead(nn.Module):
    # For each factor f:
    #   predict_f = MLP(d_f → 64 → gt_dim_f)
    #   loss_f = MSE(predict_f(z_f), gt_f)
```

Ground-truth dynamics parameters are available from the simulator during training but NOT at deployment. The auxiliary loss provides a training-time supervisory signal that encourages each latent subspace to encode information about its corresponding dynamics axis.

---

## History Buffer (`src/utils/history_buffer.py`)

GPU-resident FIFO buffer that stores the last T (obs, action) pairs per environment.

```
history_obs:  ℝ^(num_envs, history_len, obs_dim)
history_act:  ℝ^(num_envs, history_len, act_dim)
history_mask: bool^(num_envs, history_len)      # True = valid entry
```

Operations:
- `insert(obs, act)`: pushes new (obs, act) to front, shifts buffer, updates mask
- `get()`: returns `(history_obs, history_act, history_mask)`
- `reset_envs(env_ids)`: clears buffer and mask for specified environments

At episode start, the buffer is empty (mask all False). The transformer handles padding via the mask.

---

## Design Decisions & Rationale

### Why Pre-LayerNorm Transformer?

Pre-LayerNorm (applying LayerNorm before attention/FFN) is more stable for training than post-LayerNorm, especially with small models and without warmup schedules.

### Why Mean Pooling (not CLS token)?

Mean pooling aggregates information from all timesteps equally. A CLS token would add a learnable parameter but provides minimal benefit for short sequences (T=8). The ablation could test this.

### Why Learned Log-Std (not State-Dependent)?

A state-dependent std network doubles the policy head size and can cause instability in locomotion tasks. A learned (but action-dimension-specific) log-std parameter is simpler and standard in Isaac Lab PPO implementations.

### Why Separate Policy and Value Conditioning?

Both the policy and value function receive the latent z, but through separate heads. This follows standard actor-critic practice where the value function may benefit from different feature processing than the policy.

### Why MSE for Auxiliary Loss (not Contrastive)?

MSE is simplest and directly supervises the regression from latent factors to GT parameters. Contrastive losses could work but add complexity (negative sampling, temperature tuning) without clear benefit for this factored design.
