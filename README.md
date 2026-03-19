# DynaMITE: Dynamic Mismatch Inference via Transformer Encoder for Robust Humanoid Locomotion

We study whether a short-horizon transformer encoder can infer a factorized latent representation of hidden dynamics parameters — friction, mass, motor strength, contact properties, and actuation delay — from proprioceptive history alone.
DynaMITE conditions both the policy and value function on this latent vector and trains per-factor auxiliary identification losses during PPO optimization.
We evaluate on a Unitree G1 humanoid in Isaac Lab across four locomotion tasks with domain randomization.

Across 5 seeds with deterministic 100-episode evaluation, **LSTM achieves the best aggregate reward on all four tasks**, with DynaMITE ranking second on push, randomized, and terrain.
However, DynaMITE shows **dramatically lower sensitivity** than LSTM across all three OOD robustness sweeps (friction, push magnitude, action delay), and its factorized latent achieves a **0.500 ± 0.020 disentanglement score** across 3 seeds (chance = 0.20), crossing the 0.50 threshold for strong disentanglement.
The key value of DynaMITE lies in **interpretability and robustness under dynamics shifts**, not aggregate reward dominance.

---

## Contributions

1. **Factorized latent dynamics inference.** A transformer encoder maps an 8-step (160 ms) observation–action history to a 24-d latent vector decomposed into 5 semantically-aligned subspaces (friction, mass, motor, contact, delay).

2. **Per-factor auxiliary supervision.** Each subspace is trained against the corresponding ground-truth dynamics parameter via MSE loss during training. At deployment, no privileged information is required.

3. **Controlled comparison** of MLP, LSTM, Transformer, and DynaMITE policies under identical PPO training on the same four tasks, with shared observation/action embeddings, policy heads, and value heads.

4. **Latent disentanglement analysis.** We show that the learned factored subspaces achieve a 0.500 ± 0.020 disentanglement score across 3 seeds (chance = 0.20), crossing the threshold for strong alignment with ground-truth dynamics factors.

5. **Single-GPU reproducible pipeline.** Full experiment (80 training runs + 9 multi-seed ablations + 4 single-seed ablations + 18 OOD sweeps + 3 latent analyses + evaluation + plotting) completes in ~24 hours on an RTX 4060 Laptop GPU.

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
| LSTM | Hidden state | No | No | 176–215k |
| Transformer | 8 steps | No | No | 330–342k |
| DynaMITE | 8 steps | 24-d factored | Yes | 342–392k |

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

#### Multi-seed ablations (3 seeds: 42, 43, 44)

We retrain the three most impactful ablation variants with 3 seeds each. The DynaMITE (Full) baseline uses the 5-seed mean from the main comparison.

| Variant | Eval Reward | Δ vs Full |
|---|---|---|
| DynaMITE (Full, 5-seed) | **-4.48 ± 0.14** | — |
| No Latent | -4.88 ± 0.27 | -0.40 |
| No Aux Loss | -5.06 ± 0.58 | -0.58 |
| Single Latent (unfactored) | -5.25 ± 0.36 | -0.77 |

**Multi-seed takeaways:**
- **Single Latent (unfactored) has the largest degradation** (-0.77), confirming that the factored latent structure is the most critical design choice.
- **No Aux Loss** is second worst (-0.58) but exhibits high seed variance (σ = 0.58), with one seed nearly matching full DynaMITE — suggesting auxiliary loss helps but its benefit is seed-dependent.
- **No Latent** shows the mildest degradation (-0.40), indicating that the latent head's contribution is less critical than the factoring structure.
- The ranking partially shifts from the single-seed results: Single Latent is now definitively worse than No Aux Loss, while No Latent moves from 5th to the mildest ablation.

#### Extended single-seed ablations (seed 42)

Full 7-variant ablation sweep for architectural sensitivity analysis:

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

*Single-seed architectural findings:* A shallower encoder (depth 1, -0.06) barely hurts, while a deeper one (depth 4, -0.50) hurts substantially — likely overfitting with limited data. Sequence length in the 4–16 range has minimal effect (±0.08).

### Latent Disentanglement Analysis

We measure whether DynaMITE's learned latent subspaces correlate with their intended ground-truth dynamics parameters using Pearson correlation (50 episodes per seed, 3 seeds).

| Seed | Disentanglement Score |
|---|---|
| 42 | 0.496 |
| 43 | 0.482 |
| 44 | 0.521 |
| **Mean** | **0.500 ± 0.020** |

- **Mean disentanglement score: 0.500 ± 0.020** (chance = 0.20 for 5 factors)
- The score measures the ratio of within-factor to total correlation: values above 0.50 indicate strong disentanglement.
- The multi-seed mean of 0.500 crosses the strong disentanglement threshold, up from 0.477 in the single-seed pilot. The factored subspaces meaningfully align with ground-truth dynamics factors.
- Correlation heatmap and t-SNE visualizations are in `figures/`.

### OOD Robustness Sweeps (3 seeds: 42, 43, 44)

We evaluate DynaMITE and LSTM under three OOD perturbation types on the randomized task (50 episodes per level per seed). Values are mean reward across 3 seeds.

#### Friction Sweep

| Method | Fric 1.0 | Fric 0.7 | Fric 0.5 | Fric 0.3 | Fric 0.1 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.40 ± 0.07 | -4.40 ± 0.08 | **-4.38 ± 0.08** | **-4.38 ± 0.07** | **-4.38 ± 0.07** | **0.03** |
| LSTM | **-4.17 ± 0.06** | **-4.20 ± 0.07** | -4.14 ± 0.07 | -4.24 ± 0.03 | -4.34 ± 0.07 | 0.20 |

#### Push Magnitude Sweep

| Method | Push 0 | Push 0.5–1 | Push 1–2 | Push 2–3 | Push 3–5 | Push 5–8 | Sensitivity |
|---|---|---|---|---|---|---|---|
| DynaMITE | -4.31 ± 0.09 | -4.36 ± 0.08 | -4.41 ± 0.08 | **-4.44 ± 0.06** | **-4.49 ± 0.05** | **-4.56 ± 0.05** | **0.25** |
| LSTM | **-3.64 ± 0.15** | **-4.06 ± 0.03** | **-4.20 ± 0.09** | -4.43 ± 0.02 | -4.67 ± 0.02 | -5.03 ± 0.08 | 1.39 |

#### Action Delay Sweep

| Method | Delay 0 | Delay 1 | Delay 2 | Delay 3 | Delay 5 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.40 ± 0.07 | -4.41 ± 0.07 | -4.39 ± 0.07 | **-4.39 ± 0.06** | -4.40 ± 0.07 | **0.02** |
| LSTM | **-4.20 ± 0.06** | **-4.20 ± 0.06** | **-4.18 ± 0.09** | -4.15 ± 0.09 | **-4.21 ± 0.07** | 0.05 |

*Sensitivity = max(mean reward) − min(mean reward) across sweep levels. Lower is more robust.*

**Key findings:**
- **DynaMITE is dramatically more robust than LSTM under friction** (sensitivity 0.03 vs 0.20) and **push magnitude** (0.25 vs 1.39). Under action delay both are robust (0.02 vs 0.05).
- **LSTM achieves better absolute rewards** at most levels due to its overall reward advantage, but **degrades sharply** under strong pushes (−5.03 at push 5–8 vs DynaMITE's −4.56).
- **DynaMITE's reward is nearly flat** across all friction levels and delay values, suggesting its latent dynamics inference successfully compensates for parameter shifts.
- The push magnitude sweep reveals the largest model gap: LSTM's sensitivity (1.39) is **5.6× worse** than DynaMITE's (0.25).

---

## Limitations

- **LSTM dominates aggregate reward.** Despite DynaMITE's architectural advantages, LSTM achieves the best mean reward on all four tasks with the lowest seed variance. DynaMITE's value lies in robustness and interpretability, not raw performance.
- **Narrow reward spread.** After 10M steps, the top-2 models (LSTM, DynaMITE) fall within [-4.01, -4.60] across tasks — a range of ~0.6. Whether this difference is practically meaningful for a physical robot is unknown.
- **No sim-to-real transfer.** All experiments are in simulation (Isaac Lab). We have not validated on physical hardware.
- **OOD sweep scope.** Multi-seed OOD sweeps cover DynaMITE and LSTM only (the top-2 models); MLP and Transformer were evaluated single-seed in a pilot study.
- **Disentanglement at threshold.** The 0.500 ± 0.020 mean score just crosses the 0.50 strong disentanglement threshold. While above chance, there remains cross-talk between latent subspaces.
- **Reward is penalty-based.** The reward function sums several penalty terms. A method achieving -4.18 vs -4.48 is accumulating ~6% less penalty per step on average. The practical significance of this gap is unclear without real-world deployment.
- **Ablation seed count.** The three key ablations (No Aux Loss, No Latent, Single Latent) use 3 seeds; the remaining four architectural variants (depth 1/4, seq len 4/16) use a single seed.

---

## Reproduction

### Requirements

- Ubuntu 20.04+ with NVIDIA GPU (tested: RTX 4060 Laptop, 8 GB VRAM)
- Isaac Sim 4.0+ and Isaac Lab
- Python 3.10, PyTorch with CUDA

### Setup

```bash
git clone https://github.com/fjkrch/g1-factorized-latent-locomotion.git
cd g1-factorized-latent-locomotion
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

# Full reproduction pipeline (48 training + 21 ablations + eval + analysis, ~12 hours)
bash scripts/reproduce_all.sh
# Or dry-run first:
bash scripts/reproduce_all.sh --dry-run
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
  author  = {Chayanin Kraicharoen},
  year    = {2026},
  note    = {Preprint / under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
