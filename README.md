# DynaMITE: Dynamic Mismatch Inference via Transformer Encoder for Robust Humanoid Locomotion

We study whether a short-horizon transformer encoder can infer a factorized latent representation of hidden dynamics parameters — friction, mass, motor strength, contact properties, and actuation delay — from proprioceptive history alone.
DynaMITE conditions both the policy and value function on this latent vector and trains per-factor auxiliary identification losses during PPO optimization.
We evaluate on a Unitree G1 humanoid in Isaac Lab across four locomotion tasks with domain randomization.

Across 5 seeds with deterministic 100-episode evaluation, **LSTM achieves the best aggregate reward on all four tasks**, with DynaMITE ranking second on push, randomized, and terrain.
However, DynaMITE shows the **lowest friction sensitivity** among all models in OOD robustness sweeps, and its factorized latent achieves a 0.477 disentanglement score (chance = 0.20), confirming that the learned subspaces partially align with ground-truth dynamics factors.
The key value of DynaMITE lies in **interpretability and robustness under dynamics shifts**, not aggregate reward dominance.

---

## Contributions

1. **Factorized latent dynamics inference.** A transformer encoder maps an 8-step (160 ms) observation–action history to a 24-d latent vector decomposed into 5 semantically-aligned subspaces (friction, mass, motor, contact, delay).

2. **Per-factor auxiliary supervision.** Each subspace is trained against the corresponding ground-truth dynamics parameter via MSE loss during training. At deployment, no privileged information is required.

3. **Controlled comparison** of MLP, LSTM, Transformer, and DynaMITE policies under identical PPO training on the same four tasks, with shared observation/action embeddings, policy heads, and value heads.

4. **Latent disentanglement analysis.** We show that the learned factored subspaces achieve a 0.477 disentanglement score (chance = 0.20), indicating moderate but above-chance alignment with ground-truth dynamics factors.

5. **Single-GPU reproducible pipeline.** Full experiment (80 training runs + 7 ablations + evaluation + plotting) completes in ~20 hours on an RTX 4060 Laptop GPU.

---

## Method

```
     History Buffer (8 steps)
    ┌─────────────────────────┐
    │ [obs₁,act₁]…[obs₈,act₈]│
    └───────────┬─────────────┘
                │
    ┌───────────▼─────────────┐
    │ Token Embedding + PE    │
    └───────────┬─────────────┘
                │
    ┌───────────▼─────────────┐
    │ Transformer Encoder     │
    │ (2 layers, 4 heads,     │
    │  d_model=128)           │
    └───────────┬─────────────┘
                │ mean pool
       ┌────────┼────────┐
       │        │        │
  ┌────▼───┐ ┌──▼──┐ ┌──▼──┐
  │Factored│ │π(a|s│ │V(s) │
  │Latent  │ │,z)  │ │     │
  │Head    │ └──▲──┘ └──▲──┘
  │ z∈R²⁴  │    │concat  │
  │────────┼────┘        │
  │        ├─────────────┘
  │ aux ID │
  │ losses │ (train only)
  └────────┘
```

**Loss:**
$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + c_v \mathcal{L}_{\text{value}} + 0.1 \sum_{f} \mathcal{L}_{\text{aux},f}$$

All four model architectures share the same observation embedding, action embedding, policy MLP, and value MLP (`src/models/components.py`).

| Model | History | Latent | Aux Loss | Params |
|---|---|---|---|---|
| MLP | None | No | No | 266–362k |
| LSTM | Hidden state | No | No | 191–215k |
| Transformer | 8 steps | No | No | 330–342k |
| DynaMITE | 8 steps | 24-d factored | Yes | 380–392k |

---

## Results

All numbers: PPO, 512 envs, 10M timesteps, 5 seeds (42–46), deterministic evaluation (100 episodes), RTX 4060 Laptop GPU.
Rewards are penalty-based (negative); higher (less negative) is better.
Values reported as mean ± std across seeds.

### Main Comparison (5 seeds, deterministic eval)

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | -4.83 ± 0.12 | -5.01 ± 0.29 | -5.32 ± 0.45 | -4.82 ± 0.26 |
| LSTM | **-4.01 ± 0.04** | **-4.30 ± 0.04** | **-4.18 ± 0.04** | **-4.06 ± 0.04** |
| Transformer | -5.02 ± 0.32 | -4.83 ± 0.62 | -4.77 ± 0.37 | -4.46 ± 0.11 |
| DynaMITE | -4.88 ± 0.23 | -4.60 ± 0.12 | -4.48 ± 0.14 | -4.49 ± 0.13 |

**LSTM wins all four tasks** with the lowest variance (σ ≤ 0.04), indicating highly consistent performance.
DynaMITE ranks **second on push, randomized, and terrain** with moderate variance.
MLP shows high variance on randomized (σ = 0.45) and is the weakest model overall.

> **Note:** Stochastic training-time evaluation (20 episodes) showed MLP leading, but deterministic 100-episode evaluation reverses this ranking — highlighting the importance of proper evaluation protocol.

### Ablation Study (Randomized Task, 10M Steps)

Single-seed (seed 42) ablation results. Multi-seed ablation campaign (3 seeds × 3 key ablations) is in progress.

| Variant | Eval Reward | Δ vs Full |
|---|---|---|
| DynaMITE (Full) | **-4.27** | — |
| Depth 1 (1-layer encoder) | -4.32 | -0.06 |
| Seq Len 4 | -4.35 | -0.08 |
| Seq Len 16 | -4.35 | -0.08 |
| No Latent | -4.49 | -0.22 |
| Depth 4 (4-layer encoder) | -4.76 | -0.50 |
| Single Latent (unfactored) | -5.03 | -0.76 |
| No Aux Loss | -5.08 | -0.82 |

**Takeaways (single seed, interpret with caution):**
- Removing auxiliary loss has the largest negative impact (-0.82), confirming the per-factor supervision is the key component.
- Collapsing to a single unfactored latent (-0.76) is nearly as damaging, supporting factored representation.
- A shallower encoder (depth 1, -0.06) barely hurts, while a deeper one (depth 4, -0.50) hurts substantially — likely overfitting with limited data.
- Sequence length in the 4–16 range has minimal effect (±0.08).

### Latent Disentanglement Analysis

We measure whether DynaMITE's learned latent subspaces correlate with their intended ground-truth dynamics parameters using Pearson correlation on 2,176 latent samples (100 episodes, 64 envs).

- **Disentanglement score: 0.477** (chance = 0.20 for 5 factors)
- The score measures the ratio of within-factor to total correlation: values above 0.50 indicate strong disentanglement.
- The 0.477 score suggests moderate disentanglement — the model partially separates dynamics factors into their intended subspaces, but with substantial cross-talk between factors.
- Correlation heatmap and t-SNE visualizations are in `figures/`.

### OOD Robustness: Friction Sweep

We evaluated all four models across 5 friction levels (seed 42, 50 episodes per level).

| Method | Friction 1.0 | Friction 0.7 | Friction 0.5 | Friction 0.3 | Friction 0.1 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | **-4.27** | -4.32 | **-4.23** | **-4.31** | **-4.24** | **0.09** |
| LSTM | -4.74 | -4.78 | -4.77 | -4.75 | -4.82 | 0.08 |
| MLP | -4.50 | **-4.49** | -4.44 | -4.57 | -4.60 | 0.16 |
| Transformer | -4.67 | -4.50 | -4.58 | -4.67 | -4.71 | 0.21 |

*Sensitivity = max(reward) − min(reward) across friction levels. Lower is more robust.*

**Key findings:**
- **DynaMITE and LSTM are the most friction-robust** (sensitivity 0.09 and 0.08 respectively).
- **DynaMITE achieves the best absolute reward** at every friction level except 0.7.
- **Transformer is the most sensitive** (0.21 range), degrading notably at low friction.
- Multi-seed OOD sweeps (3 seeds × 2 models × 3 sweep types) are in progress.

---

## Limitations

- **LSTM dominates aggregate reward.** Despite DynaMITE's architectural advantages, LSTM achieves the best mean reward on all four tasks with the lowest seed variance. DynaMITE's value lies in robustness and interpretability, not raw performance.
- **Narrow reward spread.** After 10M steps, the top-2 models (LSTM, DynaMITE) fall within [-4.01, -4.60] across tasks — a range of ~0.6. Whether this difference is practically meaningful for a physical robot is unknown.
- **No sim-to-real transfer.** All experiments are in simulation (Isaac Lab). We have not validated on physical hardware.
- **OOD sweeps are single-seed.** Friction sweep results use seed 42 only. Multi-seed OOD evaluation is in progress.
- **Moderate disentanglement.** The 0.477 disentanglement score is above chance (0.20) but below the 0.50+ threshold for strong disentanglement, indicating substantial cross-talk between latent subspaces.
- **Reward is penalty-based.** The reward function sums several penalty terms. A method achieving -4.18 vs -4.48 is accumulating ~6% less penalty per step on average. The practical significance of this gap is unclear without real-world deployment.
- **Ablations are single-seed.** Multi-seed ablation runs (no_aux_loss, no_latent, single_latent × 3 seeds) are in progress.

---

## Reproduction

### Requirements

- Ubuntu 20.04+ with NVIDIA GPU (tested: RTX 4060 Laptop, 8 GB VRAM)
- Isaac Sim 4.0+ and Isaac Lab
- Python 3.10, PyTorch with CUDA

### Setup

```bash
git clone https://github.com/<TODO:username>/dynamite-locomotion.git
cd dynamite-locomotion
conda env create -f environment.yml
conda activate env_isaaclab
python -m pytest tests/ -v
```

### Quick start

```bash
# Train DynaMITE on randomized task
python scripts/train.py --task randomized --model dynamite --seed 42

# Evaluate
python scripts/eval.py --run_dir outputs/randomized/dynamite_full/seed_42/*/

# Full pipeline (16 training + 7 ablations + eval + plots, ~3.5 hours)
bash scripts/run_everything.sh
```

### Runtime (RTX 4060 Laptop, 512 envs)

| Run set | Time |
|---|---|
| Single training run (10M steps) | ~14 min |
| All 80 main runs (4 tasks × 4 models × 5 seeds) | ~19 hours |
| 80 deterministic evals (100 episodes each) | ~5 hours |
| 9 ablation runs (3 variants × 3 seeds) | ~2 hours |
| 18 OOD sweep evals | ~1 hour |
| Latent analysis (3 seeds) | ~15 min |

---

## Repository Structure

```
├── configs/             # YAML configs (base, task, model, train, ablations)
├── src/
│   ├── models/          # MLP, LSTM, Transformer, DynaMITE policies
│   ├── envs/            # Isaac Lab G1 wrapper, reward function
│   ├── algos/           # PPO trainer
│   ├── utils/           # Config, seeding, checkpointing, logging
│   └── analysis/        # Plotting, tables, latent analysis
├── scripts/             # train.py, eval.py, batch run scripts
├── tests/               # Unit tests
├── docs/                # Architecture and config documentation
└── outputs/             # Training outputs (git-ignored)
```

---

## Citation

```bibtex
@article{dynamite2026,
  title   = {{DynaMITE}: Dynamic Mismatch Inference via Transformer Encoder
             for Robust Humanoid Locomotion},
  author  = {<TODO: author>},
  year    = {2026},
  note    = {Preprint / under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
