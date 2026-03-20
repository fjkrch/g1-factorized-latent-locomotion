# DynaMITE: Dynamic Mismatch Inference via Transformer Encoder for Humanoid Locomotion

We study whether a short-horizon transformer encoder can infer a factorized latent representation of hidden dynamics parameters — friction, mass, motor strength, contact properties, and actuation delay — from proprioceptive history alone.
DynaMITE conditions both the policy and value function on this latent vector and trains per-factor auxiliary identification losses during PPO optimization.
We evaluate on a Unitree G1 humanoid in Isaac Lab across four locomotion tasks with domain randomization.

Across 5 seeds with deterministic 100-episode evaluation, **LSTM achieves the best aggregate reward on all four tasks** (p < 0.03 on all, paired t-test), with DynaMITE ranking second on push, randomized, and terrain.
DynaMITE shows **the lowest OOD sensitivity** across all four models on friction and push magnitude sweeps (n = 5 seeds, 4 models × 3 perturbation axes), and its factored latent achieves a **0.500 ± 0.020 score on a custom within-factor correlation metric** across 3 seeds (chance = 0.20).
DynaMITE did not outperform LSTM on nominal reward. Its potential value is a **tradeoff: worse in-distribution performance for lower OOD sensitivity and partial latent factor alignment**.

---

## Contributions

1. **Factorized latent dynamics inference.** A transformer encoder maps an 8-step (160 ms) observation–action history to a 24-d latent vector decomposed into 5 subspaces intended to align with ground-truth dynamics parameters (friction, mass, motor, contact, delay).

2. **Per-factor auxiliary supervision.** Each subspace is trained against the corresponding ground-truth dynamics parameter via MSE loss during training. At deployment, no privileged information is required.

3. **Controlled comparison** of MLP, LSTM, Transformer, and DynaMITE policies under identical PPO training on the same four tasks, with shared observation/action embeddings, policy heads, and value heads. LSTM outperformed all other methods on aggregate reward.

4. **Latent factor alignment analysis.** Under a custom within-factor correlation metric (see [Evaluation Protocol](#evaluation-protocol)), the learned subspaces achieve a mean score of 0.500 ± 0.020 across 3 seeds (chance = 0.20 for 5 factors). This indicates partial alignment but is not measured using standard disentanglement benchmarks.

5. **Single-GPU reproducible pipeline.** `reproduce_all.sh` trains 69 runs (3 seeds) + evaluates + runs OOD sweeps + latent analysis + generates tables/figures in ~12 hours on an RTX 4060 Laptop GPU. The full 5-seed experiment set reported below took ~24 hours.

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

| Model | History | Latent | Aux Loss | Params\* |
|---|---|---|---|---|
| MLP | None | No | No | 266–362k |
| LSTM | Hidden state | No | No | 176–215k |
| Transformer | 8 steps | No | No | 330–342k |
| DynaMITE | 8 steps | 24-d factored | Yes | 342–392k |

\*Parameter counts vary by task due to different observation dimensions (flat: smaller obs → lower end; terrain: obs + height-map features → higher end). See `docs/architecture.md` for tensor shape details.

---

## Evaluation Protocol

All results below follow this protocol, locked before running the main experiment campaign.

### Training Protocol

| Setting | Value |
|---|---|
| Algorithm | PPO (clipped objective, GAE) |
| Parallel envs | 512 (Isaac Lab vectorized) |
| Total timesteps | 10 M per run |
| Timestep (dt) | 20 ms (50 Hz control) |
| Checkpoint interval | Every 614,400 steps (~every 60 s) |
| Checkpoint selection | **Best** checkpoint by training-time stochastic eval reward |
| Training seeds | Unique per run; controls env randomization, network init, and PPO sampling |

### Evaluation Protocol

| Setting | Value |
|---|---|
| Eval mode | **Deterministic** — action = distribution mean, no sampling |
| Episodes per eval | 100 (main comparison, ablations); 50 (OOD sweeps) |
| Env reset | Full reset between episodes (randomized initial joint positions + domain parameters) |
| Eval env seed | Fixed at 42 for all models within a task (independent of training seed) |
| Episode termination | Fixed-length rollout (no early termination) |
| Reward aggregation | Mean of per-episode cumulative reward across all eval episodes |
| Main comparison | 5 training seeds (42, 43, 44, 45, 46) × 4 tasks × 4 models = 80 evals |
| Multi-seed ablations | 5 training seeds (42, 43, 44, 45, 46) × 3 variants = 15 evals |
| OOD sweeps | 5 training seeds (42–46) × 4 models × 5 sweep types × 3 tasks = 140 evals |
| Latent analysis | 3 training seeds (42, 43, 44) × 50 episodes each |
| Latent intervention | 3 training seeds (42, 43, 44) × 5 factors × 3 DR levels = 90 evaluations |

> **Deterministic vs stochastic eval.** During training, PPO uses stochastic policy evaluation (sampled actions, 20 episodes) for checkpoint selection. All numbers reported in this README use **deterministic evaluation** (mean action, 100 or 50 episodes) run after training completes. An initial pilot used stochastic 20-episode evaluation, which produced a different model ranking (MLP appeared best). The deterministic protocol is standard in Isaac Lab and eliminates action-sampling variance.

> **Same protocol for all models.** All four architectures (MLP, LSTM, Transformer, DynaMITE) share the same env wrapper, observation/action spaces, reward function, eval seed, episode count, and deterministic eval mode. The only difference is the policy network and whether auxiliary losses are active during training.

### Metrics

**Reward.** Penalty-based (always negative). Higher (less negative) = better. A method achieving −4.18 vs −4.48 accumulates ~6% less penalty per step on average.

**Factor alignment metric (custom).** We compute Pearson correlation between each learned latent subspace and each ground-truth dynamics parameter over 50 episodes. The "factor alignment score" is the ratio of mean within-factor correlation to mean total correlation across all factor–subspace pairs. This is a **custom metric** — not MIG, DCI, or SAP — and should not be directly compared to standard disentanglement benchmarks. It measures correlation, not causal alignment or independence. Chance level for 5 factors is 0.20. See `src/analysis/latent_analysis.py` for implementation.

**OOD sensitivity metric.** Sensitivity = max(mean reward) − min(mean reward) across sweep levels. This is a simple range metric; it does not capture curve shape or monotonicity. Lower = more robust. We also report severe-level mean (worst 2 levels), worst-case score, AUC (area under reward vs perturbation curve), and tracking error (‖v_actual − v_cmd‖).

### Statistical Reporting

For the **main comparison** (n = 5 seeds), we report:
- Mean ± sample standard deviation
- 95% confidence intervals (CI) via the t-distribution: $\text{CI} = \bar{x} \pm t_{0.025,\,n-1} \cdot s / \sqrt{n}$
- Paired t-tests (two-sided) for LSTM vs DynaMITE on matched seeds

For **ablations** (n = 5 seeds), we additionally report paired t-tests (Full vs variant). For **OOD sweeps** (n = 5 seeds, 4 models), we report mean ± std, sensitivity, severe-level mean, worst-case, AUC, Cohen's d effect size for DynaMITE vs each baseline, and Holm-Bonferroni corrected p-values.

---

## Results

### Main Comparison (5 seeds, deterministic eval)

<p align="center">
  <img src="figures/eval_bars.png" width="700" alt="Main comparison bar chart">
</p>

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | -4.83 ± 0.14 | -5.01 ± 0.32 | -5.32 ± 0.50 | -4.82 ± 0.29 |
| LSTM | **-4.01 ± 0.04** | **-4.30 ± 0.04** | **-4.18 ± 0.05** | **-4.06 ± 0.04** |
| Transformer | -5.02 ± 0.36 | -4.83 ± 0.69 | -4.77 ± 0.41 | -4.46 ± 0.12 |
| DynaMITE | -4.88 ± 0.26 | -4.60 ± 0.13 | -4.48 ± 0.16 | -4.49 ± 0.15 |

Values are mean ± std across 5 seeds. Per-seed eval rewards and 95% CIs are in the collapsed section below.

<details>
<summary><strong>95% Confidence Intervals and Paired Tests</strong></summary>

#### 95% CIs (t-distribution, n = 5)

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | [-5.00, -4.66] | [-5.41, -4.61] | [-5.94, -4.70] | [-5.18, -4.46] |
| LSTM | [-4.07, -3.96] | [-4.35, -4.25] | [-4.24, -4.12] | [-4.11, -4.01] |
| Transformer | [-5.46, -4.57] | [-5.69, -3.97] | [-5.29, -4.26] | [-4.61, -4.31] |
| DynaMITE | [-5.21, -4.56] | [-4.77, -4.44] | [-4.67, -4.29] | [-4.68, -4.31] |

#### Paired t-tests: LSTM vs DynaMITE (matched training seeds)

| Task | Mean Diff (LSTM − DynaMITE) | Paired t | p-value |
|---|---|---|---|
| Flat | +0.870 ± 0.229 | 8.49 | 0.0011 |
| Push | +0.305 ± 0.145 | 4.69 | 0.0094 |
| Randomized | +0.303 ± 0.203 | 3.34 | 0.029 |
| Terrain | +0.435 ± 0.114 | 8.55 | 0.0010 |

LSTM is significantly better than DynaMITE on all four tasks (p < 0.05, paired t-test, n = 5). The largest gap is on flat (+0.87); the smallest on push and randomized (~+0.30). These are two-sided tests without multiple-comparison correction.

</details>

LSTM achieves the best mean reward on all four tasks with the lowest variance (σ ≤ 0.05). All four paired t-tests (LSTM vs DynaMITE) are significant at p < 0.03.
DynaMITE ranks second on push, randomized, and terrain with moderate variance.
MLP shows high variance on randomized (σ = 0.50) and is the weakest model overall.

### Training Curves (Randomized Task, 5-Seed Average)

<p align="center">
  <img src="figures/training_curves.png" width="700" alt="Training curves — randomized task, 5-seed mean ± std">
</p>

### Ablation Study (Randomized Task, 10M Steps)

#### Multi-seed ablations (5 seeds: 42–46, deterministic eval)

We train the three most impactful ablation variants with 5 seeds each (matching the main comparison) and evaluate with deterministic 100-episode evaluation.

| Variant | Deterministic Eval Reward | Δ vs Full | 95% CI |
|---|---|---|---|
| DynaMITE (Full) | **-4.48 ± 0.16** | — | [-4.68, -4.29] |
| No Aux Loss | -4.56 ± 0.27 | -0.08 | [-4.89, -4.23] |
| No Latent | -4.77 ± 0.41 | -0.29 | [-5.29, -4.26] |
| Single Latent (unfactored) | -4.67 ± 0.11 | -0.19 | [-4.80, -4.54] |

<details>
<summary><strong>Paired t-tests and per-seed data (n = 5)</strong></summary>

#### Paired t-tests: Full vs ablation variants

| Variant | Delta | Paired t | p-value |
|---|---|---|---|
| No Aux Loss | -0.08 | 0.52 | 0.629 |
| No Latent | -0.29 | 1.65 | 0.174 |
| Single Latent | -0.19 | 2.55 | 0.063 |

No ablation variant reaches p < 0.05 with n = 5, though Single Latent approaches significance (p = 0.063). All three variants show consistent degradation in mean reward.

#### Per-seed rewards

| Seed | Full | No Aux Loss | No Latent | Single Latent |
|---|---|---|---|---|
| 42 | -4.39 | -5.01 | -4.39 | -4.70 |
| 43 | -4.46 | -4.32 | -4.47 | -4.79 |
| 44 | -4.34 | -4.43 | -5.28 | -4.64 |
| 45 | -4.74 | -4.52 | -5.16 | -4.72 |
| 46 | -4.47 | -4.51 | -4.57 | -4.51 |

</details>

**Observations:**
- All three ablation variants show consistent directional degradation compared to the full model (Δ = 0.08–0.29), but no variant reaches statistical significance at p < 0.05 with n = 5.
- **No Latent** shows the largest mean degradation (Δ = -0.29, p = 0.174) and highest variance (σ = 0.41). The direction is consistent (4/5 seeds degrade) but the effect is not statistically reliable at this sample size.
- **Single Latent (unfactored)** shows Δ = -0.19, p = 0.063. This does not reach significance. All 5 seeds show degradation, suggesting a consistent but small effect that would require more seeds to confirm.
- **No Aux Loss** shows the smallest mean degradation (Δ = -0.08, p = 0.629) with high variance (σ = 0.27) and inconsistent direction (2/5 seeds improve). The auxiliary loss may not contribute meaningfully to nominal reward.
- These results use the same deterministic 100-episode protocol as the main comparison.

### Latent Factor Alignment Analysis

We measure whether DynaMITE's learned latent subspaces correlate with their intended ground-truth dynamics parameters using Pearson correlation (50 episodes per seed, 3 seeds).

<p align="center">
  <img src="figures/latent_correlation_heatmap.png" width="500" alt="Latent correlation heatmap">
</p>

| Seed | Within-Factor Correlation Score |
|---|---|
| 42 | 0.496 |
| 43 | 0.482 |
| 44 | 0.521 |
| **Mean** | **0.500 ± 0.020** |

- **Mean score: 0.500 ± 0.020** under our custom within-factor correlation metric (chance = 0.20 for 5 factors). This is not a standard disentanglement measure (see [Evaluation Protocol](#evaluation-protocol)).
- The score exceeds chance, indicating partial factor alignment. However, the metric measures correlation only — not independence, causal alignment, or invariance. Cross-talk between subspaces remains, and whether 0.50 constitutes meaningful alignment depends on the application.
- No intervention experiments (clamping or perturbing individual latent dimensions) have been performed, so the correlation evidence does not establish causal factor alignment.
- Full analysis in `figures/latent_correlation_full.png`.

### OOD Sensitivity Sweeps v2 (5 seeds, 4 models, full behavioral metrics)

We evaluate all four models under OOD perturbations with four metrics: reward (higher = better), failure rate (fraction of episodes ending in fall), tracking error (||v_actual − v_cmd||, lower = better), and completion rate (fraction of episodes surviving to time limit). 50 episodes per level per seed, 5 seeds per model. Total: 140 sweep evaluations across 5 sweep types × 3 tasks.

**Note on failure rate:** All episodes on rough terrain end in falls (failure rate = 1.0 for all models at all levels). Isaac Lab's `base_contact` termination triggers within 20–35 steps — well before the 2000-step episode limit. This is expected for the rough terrain G1 environment and means failure rate does not differentiate models here. The key differentiators are reward, tracking error, and how steeply they degrade under perturbation.

<p align="center">
  <img src="figures/sweep_robustness_combined.png" width="700" alt="OOD robustness sweep comparison — 4 models, 5 seeds">
</p>

#### Friction Sweep (Randomized Task)

**Reward:**

| Method | 1.0 | 0.7 | 0.5 | 0.3 | 0.1 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.47 ± 0.13 | -4.46 ± 0.13 | -4.45 ± 0.13 | -4.44 ± 0.11 | -4.43 ± 0.11 | **0.04** |
| LSTM | **-4.18 ± 0.03** | **-4.17 ± 0.04** | **-4.17 ± 0.07** | **-4.19 ± 0.08** | **-4.30 ± 0.12** | 0.13 |
| Transformer | -4.77 ± 0.41 | -4.72 ± 0.37 | -4.67 ± 0.35 | -4.65 ± 0.33 | -4.61 ± 0.32 | 0.16 |
| MLP | -5.77 ± 0.68 | -5.63 ± 0.59 | -5.51 ± 0.52 | -5.44 ± 0.51 | -5.42 ± 0.53 | 0.35 |

**Tracking error:**

| Method | 1.0 | 0.7 | 0.5 | 0.3 | 0.1 |
|---|---|---|---|---|---|
| DynaMITE | 2.58 ± 0.17 | 2.46 ± 0.11 | 2.53 ± 0.19 | 2.54 ± 0.13 | 2.59 ± 0.08 |
| LSTM | **2.01 ± 0.03** | **2.03 ± 0.07** | **2.07 ± 0.03** | **2.12 ± 0.11** | **2.03 ± 0.08** |
| Transformer | 2.50 ± 0.16 | 2.49 ± 0.14 | 2.52 ± 0.18 | 2.55 ± 0.06 | 2.55 ± 0.08 |
| MLP | 2.59 ± 0.08 | 2.61 ± 0.08 | 2.60 ± 0.04 | 2.67 ± 0.08 | 2.67 ± 0.13 |

#### Push Magnitude Sweep (Randomized Task)

**Reward:**

| Method | 0 | 0.5–1 | 1–2 | 2–3 | 3–5 | 5–8 | Sensitivity |
|---|---|---|---|---|---|---|---|
| DynaMITE | -4.37 ± 0.13 | -4.44 ± 0.16 | -4.50 ± 0.17 | -4.56 ± 0.20 | **-4.63 ± 0.24** | **-4.78 ± 0.32** | 0.41 |
| LSTM | **-3.58 ± 0.13** | **-4.05 ± 0.04** | **-4.23 ± 0.04** | **-4.45 ± 0.03** | -4.70 ± 0.10 | -5.09 ± 0.11 | 1.52 |
| Transformer | -4.65 ± 0.39 | -4.72 ± 0.40 | -4.76 ± 0.38 | -4.81 ± 0.39 | -4.94 ± 0.41 | -5.08 ± 0.41 | 0.42 |
| MLP | -5.65 ± 0.64 | -5.64 ± 0.68 | -5.79 ± 0.77 | -5.75 ± 0.67 | -5.89 ± 0.77 | -5.97 ± 0.68 | **0.33** |

**Tracking error:**

| Method | 0 | 0.5–1 | 1–2 | 2–3 | 3–5 | 5–8 |
|---|---|---|---|---|---|---|
| DynaMITE | 2.15 ± 0.16 | 2.27 ± 0.23 | 2.57 ± 0.25 | 3.21 ± 0.17 | 4.26 ± 0.21 | 6.15 ± 0.17 |
| LSTM | **1.33 ± 0.01** | **1.69 ± 0.05** | **2.15 ± 0.05** | **3.04 ± 0.11** | 4.27 ± 0.07 | 6.34 ± 0.21 |
| Transformer | 2.09 ± 0.19 | 2.22 ± 0.21 | 2.60 ± 0.10 | 3.15 ± 0.20 | **4.17 ± 0.11** | **5.93 ± 0.10** |
| MLP | 2.23 ± 0.12 | 2.40 ± 0.11 | 2.69 ± 0.12 | 3.32 ± 0.14 | 4.48 ± 0.21 | 6.46 ± 0.28 |

#### Action Delay Sweep (Randomized Task)

**Reward:**

| Method | 0 | 1 | 2 | 3 | 5 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.47 ± 0.13 | -4.49 ± 0.16 | -4.47 ± 0.13 | -4.48 ± 0.17 | -4.50 ± 0.18 | 0.02 |
| LSTM | **-4.19 ± 0.05** | **-4.17 ± 0.03** | **-4.17 ± 0.06** | **-4.16 ± 0.05** | **-4.17 ± 0.06** | 0.04 |
| Transformer | -4.76 ± 0.40 | -4.75 ± 0.39 | -4.76 ± 0.40 | -4.77 ± 0.40 | -4.75 ± 0.40 | **0.02** |
| MLP | -5.77 ± 0.69 | -5.77 ± 0.70 | -5.81 ± 0.78 | -5.73 ± 0.72 | -5.71 ± 0.66 | 0.10 |

**Tracking error:**

| Method | 0 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| DynaMITE | 2.56 ± 0.18 | 2.46 ± 0.08 | 2.55 ± 0.16 | 2.48 ± 0.17 | 2.53 ± 0.22 |
| LSTM | **2.04 ± 0.06** | **2.00 ± 0.12** | **2.01 ± 0.05** | **2.01 ± 0.04** | **2.00 ± 0.05** |
| Transformer | 2.52 ± 0.15 | 2.50 ± 0.15 | 2.50 ± 0.20 | 2.44 ± 0.13 | 2.48 ± 0.16 |
| MLP | 2.62 ± 0.07 | 2.62 ± 0.11 | 2.59 ± 0.10 | 2.60 ± 0.08 | 2.58 ± 0.06 |

### Cross-Task OOD (Push + Terrain)

Push magnitude sweep evaluated on push-task and terrain-task checkpoints to test whether robustness patterns generalize beyond the randomized task.

#### Push Task — Push Magnitude Reward

| Method | 0 | 0.5–1 | 1–2 | 2–3 | 3–5 | 5–8 | Sensitivity |
|---|---|---|---|---|---|---|---|
| DynaMITE | -4.48 ± 0.11 | -4.53 ± 0.10 | -4.58 ± 0.11 | -4.62 ± 0.12 | -4.74 ± 0.21 | **-4.87 ± 0.22** | **0.39** |
| LSTM | **-3.61 ± 0.08** | **-4.00 ± 0.08** | **-4.23 ± 0.05** | **-4.43 ± 0.04** | **-4.64 ± 0.06** | -5.01 ± 0.09 | 1.41 |
| Transformer | -4.73 ± 0.77 | -4.91 ± 1.02 | -4.82 ± 0.74 | -4.95 ± 0.93 | -5.04 ± 0.91 | -5.16 ± 0.81 | 0.42 |
| MLP | -5.29 ± 0.50 | -5.26 ± 0.47 | -5.33 ± 0.41 | -5.46 ± 0.36 | -5.43 ± 0.29 | -5.66 ± 0.44 | 0.39 |

#### Terrain Task — Push Magnitude Reward

| Method | 0 | 0.5–1 | 1–2 | 2–3 | 3–5 | 5–8 | Sensitivity |
|---|---|---|---|---|---|---|---|
| DynaMITE | -4.39 ± 0.11 | -4.46 ± 0.13 | -4.54 ± 0.19 | -4.59 ± 0.21 | -4.73 ± 0.30 | -4.84 ± 0.35 | 0.46 |
| LSTM | **-3.58 ± 0.12** | **-3.97 ± 0.07** | **-4.22 ± 0.05** | **-4.43 ± 0.04** | -4.68 ± 0.04 | -5.19 ± 0.09 | 1.61 |
| Transformer | -4.40 ± 0.13 | -4.44 ± 0.12 | -4.53 ± 0.14 | -4.58 ± 0.15 | **-4.64 ± 0.12** | **-4.82 ± 0.18** | 0.42 |
| MLP | -5.06 ± 0.49 | -5.10 ± 0.54 | -5.16 ± 0.50 | -5.16 ± 0.50 | -5.30 ± 0.54 | -5.44 ± 0.59 | **0.37** |

LSTM's sensitivity pattern is consistent across all three tasks: randomized (1.52), push (1.41), terrain (1.61). DynaMITE maintains low sensitivity across tasks (0.39–0.46).

### Unseen-Range and Combined-Shift Stress Tests

#### Unseen Action Delay (Training range: [0, 3]; tested up to delay = 10)

| Method | 0 | 3 | 5 | 7 | 10 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.47 ± 0.13 | -4.49 ± 0.16 | -4.47 ± 0.13 | -4.48 ± 0.17 | -4.50 ± 0.18 | 0.02 |
| LSTM | **-4.19 ± 0.05** | **-4.17 ± 0.03** | **-4.17 ± 0.06** | **-4.16 ± 0.05** | **-4.17 ± 0.06** | 0.04 |
| Transformer | -4.76 ± 0.40 | -4.75 ± 0.39 | -4.76 ± 0.40 | -4.77 ± 0.40 | -4.75 ± 0.40 | **0.02** |
| MLP | -5.77 ± 0.69 | -5.77 ± 0.70 | -5.81 ± 0.78 | -5.73 ± 0.72 | -5.71 ± 0.66 | 0.10 |

All models are insensitive to action delay even at 3.3× the training range. This suggests the G1 policy step is slow enough relative to sim.dt that delay has negligible impact.

#### Combined Shift (Friction + Push + Delay shifted simultaneously)

| Method | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | -4.38 ± 0.14 | -4.47 ± 0.15 | -4.50 ± 0.17 | **-4.54 ± 0.18** | **-4.63 ± 0.23** | 0.25 |
| LSTM | **-3.56 ± 0.15** | **-4.23 ± 0.05** | **-4.40 ± 0.04** | -4.63 ± 0.07 | -5.12 ± 0.10 | 1.57 |
| Transformer | -4.67 ± 0.40 | -4.73 ± 0.38 | -4.72 ± 0.34 | -4.75 ± 0.35 | -4.86 ± 0.37 | 0.20 |
| MLP | -5.68 ± 0.69 | -5.61 ± 0.66 | -5.55 ± 0.55 | -5.57 ± 0.53 | -5.64 ± 0.63 | **0.13** |

**Combined shift tracking error:**

| Method | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|---|
| DynaMITE | 2.15 ± 0.13 | 2.56 ± 0.22 | 3.18 ± 0.16 | 4.23 ± 0.15 | 6.13 ± 0.17 |
| LSTM | **1.31 ± 0.02** | **2.19 ± 0.09** | **3.01 ± 0.19** | 4.59 ± 0.14 | 6.94 ± 0.15 |
| Transformer | 2.08 ± 0.21 | 2.58 ± 0.13 | 3.17 ± 0.15 | 4.35 ± 0.07 | 6.48 ± 0.22 |
| MLP | 2.24 ± 0.08 | 2.76 ± 0.13 | 3.40 ± 0.14 | **4.47 ± 0.22** | **6.30 ± 0.29** |

Combined-shift is the most informative stress test. LSTM's sensitivity (1.57) is 6.3× DynaMITE's (0.25). Notably LSTM's tracking error at level 4 (6.94) exceeds all other models, and it crosses DynaMITE at level 3 — the LSTM reward advantage inverts under strong multi-axis perturbation.

### Latent Intervention Analysis

We perform factor-subspace clamping: for each of the 5 latent factor groups (friction dims 0–3, mass 4–9, motor 10–15, contact 16–19, delay 20–23), we clamp those dimensions to their mean activation while leaving the rest free. If a factor subspace causally drives policy behavior, clamping it should degrade reward.

| Factor | Avg |Δ Reward| | Interpretation |
|---|---|---|
| Friction (dims 0–3) | 0.007 | Negligible |
| Mass (dims 4–9) | 0.012 | Negligible |
| Motor (dims 10–15) | 0.021 | Negligible |
| Contact (dims 16–19) | 0.020 | Negligible |
| Delay (dims 20–23) | 0.020 | Negligible |

All reward deltas are < 0.05 across 3 seeds × 5 factors × 3 DR levels. **The factored latent is correlational, not causally driving policy behavior** in a cleanly separable way. Possible interpretations: (1) the policy head is robust to latent perturbations, (2) the encoding is distributed across the full latent rather than cleanly factored, or (3) the auxiliary loss teaches useful representations that are redundant with information available in the observation history.

<p align="center">
  <img src="figures/latent_intervention_results.png" width="600" alt="Latent intervention: reward delta by factor">
</p>

### Overall OOD Robustness Summary

| Model | Avg Reward | Worst Reward | Max Sensitivity | Mean Track Err |
|---|---|---|---|---|
| DynaMITE | -4.53 | -4.87 | 0.46 | 3.08 |
| LSTM | **-4.28** | -5.19 | 1.61 | **2.72** |
| Transformer | -4.75 | -5.16 | **0.42** | 3.09 |
| MLP | -5.58 | -5.97 | 0.39 | 3.21 |

#### Pairwise Statistical Comparisons (Holm-Bonferroni corrected)

Only comparisons reaching significance (adjusted p < 0.05) after Holm-Bonferroni correction across all pairwise tests:

| Comparison | Sweep | Cohen's d | p (adj) |
|---|---|---|---|
| DynaMITE vs MLP | action_delay | 2.65 | 0.044 |
| DynaMITE vs LSTM | push_magnitude (push task) | -5.59 | 0.019 |
| DynaMITE vs MLP | push_magnitude (push task) | 1.41 | 0.022 |

Most pairwise comparisons do **not** reach significance after correction, reflecting tight confidence intervals and modest effect sizes with n = 5 seeds.

**Observations:**
- **Push magnitude remains the most discriminating perturbation** across all three tasks. LSTM's sensitivity (1.41–1.61) is consistently 3–4× DynaMITE's (0.39–0.46).
- **Combined shift is the strongest stress test.** LSTM degrades from -3.56 (level 0) to -5.12 (level 4), a 1.57 reward drop. DynaMITE stays within 0.25 of its baseline. LSTM's tracking error at level 4 (6.94) is the worst of all models.
- **Friction and action delay show small sensitivity for all models.** Action delay is insensitive even at 3.3× the training range (delay = 10), suggesting it is not a meaningful perturbation axis for this environment.
- **LSTM wins absolute reward at low-to-moderate perturbations** but its advantage erodes and inverts under strong combined perturbation (level 3–4).
- **Failure rate = 1.0 universally.** All episodes end in falls on rough terrain within 20–35 steps. This is an environment property, not a model deficiency.
- **Tracking error differentiates models at low perturbation** (LSTM ≈ 1.3, DynaMITE ≈ 2.1) but converges for all models at high push magnitude (all ≈ 6.0–6.5).
- **Latent intervention shows no causal factor separability.** The factored latent does not individually drive policy behavior despite showing correlational alignment. The auxiliary loss may teach useful but distributed representations.
- **Few comparisons reach statistical significance** after Holm-Bonferroni correction (3 of 42 pairwise tests). This limits the strength of ranking claims.

---

## When to Use DynaMITE vs LSTM

| Scenario | Recommended |
|---|---|
| Maximize nominal in-distribution reward | **LSTM** — wins all 4 tasks significantly |
| Moderate dynamics mismatch at deployment (single-axis shift) | **LSTM** — retains absolute reward advantage under friction, delay, and moderate push |
| Severe or multi-axis dynamics mismatch (e.g., combined sim-to-real gap) | **DynaMITE** — sensitivity 0.25 vs LSTM's 1.57 under combined shift; LSTM advantage inverts at level 3–4 |
| Need to inspect latent dynamics estimates | **DynaMITE** — factored latent shows correlational factor alignment (not causal per intervention analysis) |
| Training budget is tight | **LSTM** — fewer parameters, no auxiliary loss overhead |
| Deployment environment is well-characterized | **LSTM** — DynaMITE's sensitivity advantage is unnecessary |

---

## Limitations

- **LSTM dominates aggregate reward.** LSTM achieves the best mean reward on all four tasks with the lowest seed variance (p < 0.03 on all four, paired t-test). DynaMITE's potential value is a tradeoff — lower OOD sensitivity at the cost of worse nominal performance — not overall superiority.
- **LSTM advantage inverts only under strong combined perturbation** (level 3–4 of the combined shift). Under single-axis perturbations and moderate shifts, LSTM remains superior.
- **Narrow reward spread.** The top-2 models (LSTM, DynaMITE) fall within [-4.01, -4.60] across tasks — a range of ~0.6. Whether this is practically meaningful for a physical robot is unknown.
- **No sim-to-real transfer.** All experiments are in simulation (Isaac Lab). Not validated on physical hardware.
- **Failure rate is uninformative.** All episodes end in falls on rough terrain (failure rate = 1.0 for all models). The `base_contact` termination triggers within 20–35 steps. This environment property means failure rate cannot differentiate models.
- **OOD sweep scope.** OOD sweeps now cover 5 perturbation types (friction, push magnitude, action delay, unseen delay, combined shift) across 3 tasks = 140 evaluations. Mass, contact stiffness, and observation noise remain untested.
- **Action delay is not a meaningful perturbation axis.** All models show near-zero sensitivity even at 3.3× the training range (delay = 10), limiting its value as a robustness discriminator.
- **Custom factor alignment metric.** The 0.500 ± 0.020 score uses a within-factor correlation ratio, not a standard disentanglement metric (MIG, DCI, SAP). Direct comparison to other work is not possible.
- **Latent is correlational, not causal.** Factor-subspace clamping (all |Δ reward| < 0.05) shows the factored latent does not individually drive policy behavior. The auxiliary loss may teach useful but distributed or redundant representations.
- **Reward is penalty-based.** A method achieving -4.18 vs -4.48 accumulates ~6% less penalty per step on average. Practical significance is unclear without real-world deployment.
- **Ablation significance.** No ablation variant reaches statistical significance at p < 0.05 with n = 5, though all three show consistent directional degradation (Δ = 0.08–0.29).
- **Few OOD comparisons reach significance.** Only 3 of 42 pairwise tests survive Holm-Bonferroni correction (p_adj < 0.05). Most model ranking claims are directional, not statistically confirmed.

---

## Future Work

- **Standard disentanglement metrics.** Supplement the custom within-factor correlation metric with established measures (MIG, DCI, SAP) for comparability with the representation learning literature.
- **Why is the latent not causal?** Factor-subspace clamping showed negligible reward impact. Investigate whether this is due to distributed encoding (information spread across all 24 dimensions), policy head robustness, or redundancy with the observation history. Gradient-based attribution or ablation of the full latent (not just subspaces) could help.
- **Cross-factor leakage analysis.** Quantify off-diagonal leakage in the correlation matrix and report it alongside the disentanglement score.
- **Latent response curves.** Plot per-subspace latent activations as a function of individual ground-truth parameters (friction, mass, delay) varied one-at-a-time, to visualize monotonicity and sensitivity.
- **Bootstrap / permutation CIs.** Replace or supplement t-distribution CIs with non-parametric bootstrap intervals, especially given n = 5.
- **Increase seed count.** Many pairwise OOD comparisons fail to reach significance with n = 5 after Holm-Bonferroni correction. n = 10–20 seeds would provide better statistical power.
- **More informative environments.** The rough terrain G1 environment produces 100% failure rate (all episodes end in falls), making failure rate non-discriminative. Testing on flat or easier terrain variants where some models survive full episodes would make behavioral metrics more informative.
- **Additional perturbation axes.** Mass, contact stiffness, and observation noise remain untested. These may interact differently with the latent inference mechanism.
- **Sim-to-real transfer.** The combined-shift stress test suggests DynaMITE may be more robust under real-world dynamics mismatch, but this claim requires physical hardware validation.

---

## Reproduction

### Requirements

| Requirement | Tested Version |
|---|---|
| OS | Ubuntu 20.04+ |
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | 2.2 |
| Isaac Sim | 4.0+ |
| Isaac Lab | Compatible with Isaac Sim 4.0 |
| Disk space | ~15 GB (training outputs) |
| RAM | 14 GB+ |

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

# Full 3-seed reproduction (69 training runs + eval + OOD + latent + tables/figures, ~12 hours)
bash scripts/reproduce_all.sh
# Or dry-run first:
bash scripts/reproduce_all.sh --dry-run
```

> **Note:** `reproduce_all.sh` uses 3 seeds (42–44). The 5-seed main comparison in the Results section above used additional campaign scripts (`run_all_main.sh` with seeds 42–46, ~19 hours). No pre-trained checkpoints are provided; all models must be trained from scratch.

### Runtime (RTX 4060 Laptop, 512 envs)

| Run set | Time |
|---|---|
| Single training run (10M steps) | ~14 min |
| `reproduce_all.sh` (3-seed, 69 training + eval + analysis) | ~12 hours |
| All 80 main runs (4 tasks × 4 models × 5 seeds) | ~19 hours |
| 80 deterministic evals (100 episodes each) | ~5 hours |
| 15 ablation runs (3 variants × 5 seeds) | ~3.5 hours |
| 60 OOD sweep evals (original 3 sweeps) | ~2.5 hours |
| 140 OOD sweep evals v2 (5 sweeps × 3 tasks) | ~6 hours |
| Latent intervention (3 seeds × 5 factors × 3 levels) | ~45 min |
| Latent analysis (3 seeds) | ~15 min |
| **Full 5-seed experiment set (including OOD v2)** | **~30 hours** |

### Artifact Mapping

| README Section | Script | Output |
|---|---|---|
| Main Comparison table | `scripts/eval.py` → `scripts/aggregate_seeds.py` | `results/aggregated/` |
| Main Comparison figure | `scripts/plot_results.py` | `figures/eval_bars.png` |
| Training Curves figure | `scripts/plot_results.py` | `figures/training_curves.png` |
| Ablation table | `scripts/generate_tables.py` | `figures/ablation_table.md` |
| Factor alignment table | `scripts/run_latent_analysis.py` | `results/latent_analysis/` |
| Factor alignment heatmap | `scripts/run_latent_analysis.py` | `figures/latent_correlation_heatmap.png` |
| OOD Sweep tables (v2) | `scripts/analyze_ood_v2.py` | `results/aggregated/ood_analysis_v2.json` |
| OOD Sweep figures (v2) | `scripts/plot_ood_v2.py` | `figures/ood_v2_*.png`, `figures/ood_v2_heatmap.png` |
| Cross-task / stress test tables | `scripts/analyze_ood_v2.py` | `results/aggregated/ood_analysis_v2.json` |
| Latent intervention table | `scripts/latent_intervention.py` | `results/latent_intervention/` |
| Latent intervention figure | `scripts/latent_intervention.py` | `figures/latent_intervention_results.png` |

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
├── reproducibility/     # Checklist and expected results reference
└── outputs/             # Training outputs (git-ignored)
```

---

## Citation

```bibtex
@article{dynamite2026,

  title   = {{DynaMITE}: Dynamic Mismatch Inference via Transformer Encoder
             for Humanoid Locomotion},
  author  = {Chayanin Kraicharoen},
  year    = {2026},
  note    = {Preprint / under review}
}
```

## License

MIT. See [LICENSE](LICENSE).
