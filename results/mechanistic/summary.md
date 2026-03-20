# Mechanistic Analysis: Summary of Results

**Date:** 2026-03-20
**Seeds:** 42–46 (geometry/MINE), 42–44 (gradient flow)

---

## 1. Gradient Flow Analysis

**Question:** Do auxiliary losses align with or oppose the PPO objective?

**Method:** Retrained DynaMITE (3 seeds, 10M steps each) with gradient
instrumentation.  Logged per-component gradient norms and cosine similarity
between PPO and each auxiliary loss gradient every 10 iterations (~81 data
points per seed).

### Key findings

| Metric | Value |
|--------|-------|
| Mean cosine(PPO, aux_friction) | −0.0014 ± 0.0045 |
| Mean cosine(PPO, aux_mass) | −0.0024 ± 0.0096 |
| Mean cosine(PPO, aux_motor) | −0.0048 ± 0.0161 |
| Mean cosine(PPO, aux_contact) | −0.0077 ± 0.0261 |
| Mean cosine(PPO, aux_delay) | −0.0017 ± 0.0087 |

**All cosine similarities are indistinguishable from zero** (mean < 0.01,
std ≈ 0.01–0.03 across training).  The auxiliary gradients are *orthogonal*
to the PPO gradient throughout training, not aligned with it.

### Gradient magnitude at convergence (last 10 log points, 3 seeds)

| Component | Norm |
|-----------|------|
| PPO (policy + value − entropy) | 2.15 ± 0.94 |
| aux_friction | 0.69 ± 0.12 |
| aux_delay | 0.47 ± 0.36 |
| aux_mass | 0.16 ± 0.11 |
| aux_motor | 0.12 ± 0.10 |
| aux_contact | 0.02 ± 0.01 |
| **Total** | **2.36 ± 0.90** |

The auxiliary gradients contribute ~20–40% of the total gradient norm
(depending on seed), but in directions orthogonal to the PPO objective.
This is consistent with the **gradient regularization hypothesis**: the
aux losses impose additional structure on the learned representation
without directly improving the RL objective.

**Figures:** `figures/gradient_norms.png`, `figures/cosine_similarity.png`

---

## 2. Representation Geometry

**Question:** How does DynaMITE's 24-d factored latent differ in structure
from LSTM's 128-d hidden state?

**Method:** Collected ~36,000 representations per model/seed (200 episodes
× 32 envs) from trained checkpoints.  Computed SVD-based geometry metrics.

### Key findings

| Metric | DynaMITE (24-d) | LSTM (128-d) |
|--------|:---------------:|:------------:|
| Effective rank | **4.78 ± 0.72** | 32.20 ± 3.85 |
| Participation ratio | **2.27 ± 0.15** | 4.96 ± 1.52 |
| Condition number | 315.9 ± 204.1 | **108.4 ± 19.9** |

**DynaMITE's latent is much lower-dimensional in practice** (effective
rank ~5 out of 24 possible dimensions), while **LSTM uses more of its
capacity** (effective rank ~32 out of 128).

The low effective rank of DynaMITE's latent means the tanh-bottlenecked
representaton concentrates information in very few dimensions.  This is
consistent with a *compression* regime forced by the auxiliary losses
rather than a rich dynamics-encoding regime.

The high condition number for DynaMITE (316 vs 108) indicates the latent
has a highly anisotropic structure—a few dimensions dominate while most
are nearly unused.

**Figure:** `figures/geometry_comparison.png`

---

## 3. Mutual Information (MINE + KNN Fallback)

**Question:** Is there *any* nonlinear information about dynamics
parameters in the learned representations, beyond what linear probes
(R² ≈ 0) can detect?

**Method:** MINE with 3-layer MLP critic (128 hidden, 5000 steps), EMA
baseline for variance reduction, KNN fallback when MINE is unstable.
All MINE runs fell back to KNN (sklearn mutual_info_regression with
n_neighbors=5), indicating the signal is weak enough that MINE's gradient
estimator cannot reliably converge.

### Key findings

| Factor | DynaMITE MI (nats) | LSTM MI (nats) |
|--------|:------------------:|:--------------:|
| overall | **0.233 ± 0.052** | 0.028 ± 0.033 |
| friction | **0.054 ± 0.021** | 0.024 ± 0.020 |
| mass | **0.091 ± 0.010** | 0.011 ± 0.013 |
| motor | **0.037 ± 0.031** | 0.013 ± 0.015 |
| contact | **0.026 ± 0.011** | 0.010 ± 0.015 |
| delay | 0.007 ± 0.004 | 0.005 ± 0.003 |

**DynaMITE's latent contains ~8× more mutual information with dynamics
parameters** than LSTM's hidden state (0.233 vs 0.028 nats overall).
The per-factor pattern shows:

- **Mass** has the strongest signal (0.091 nats)—reasonable since mass
  affects balance dynamics the most
- **Friction** and **motor** are moderate
- **Delay** is negligible for both models

However, **all MI values are very small in absolute terms** (< 0.25 nats).
For context, if the latent perfectly encoded the 8 target dimensions with
uniform marginals over the DR ranges, MI would be on the order of several
nats.  The measured values suggest the latent captures at most a few bits
of dynamics information, mixed with substantial task-relevant information.

**Important caveat:** All MINE runs fell back to KNN, suggesting the MI
signal is near the noise floor of the MINE estimator.  The KNN estimates
are more robust but potentially biased downward for high-dimensional inputs.

**Figure:** `figures/mi_comparison.png`

---

## 4. Mechanistic Hypothesis

Combining all three analyses:

1. **Gradient orthogonality** → The aux losses don't help the RL objective
   *directly*.  They push the gradient in orthogonal directions.

2. **Low effective rank** → The bottleneck forces severe compression.  The
   latent uses ~5 effective dimensions (of 24), creating a low-dimensional
   information highway between the history encoder and the policy head.

3. **Weak MI** → The compressed representation contains *some* dynamics
   information (8× more than LSTM), but far less than would be needed for
   genuine system identification.

**Revised mechanistic story:** The auxiliary losses act as a
*representation regularizer* that forces the latent bottleneck to compress
adaptive-control-relevant features into a low-dimensional subspace.  This
compression is beneficial not because it enables dynamics identification,
but because it prevents the history encoder from collapsing to a trivial
representation (as LSTM's higher effective rank but lower MI suggests LSTM
spreads information more diffusely).  The improved OOD robustness likely
comes from this forced compression creating a more structured, less
overfittable representation—not from inferring dynamics parameters.

This is consistent with the paper's main negative finding: **the auxiliary
losses regularize but do not identify dynamics**.

---

## Reproduction

```bash
# Full pipeline (gradient flow ~45 min, geometry+MINE ~25 min)
bash scripts/run_mechanistic.sh

# Skip gradient retraining (just geometry + MINE from existing checkpoints)
bash scripts/run_mechanistic.sh --skip-gradient

# Aggregate and plot only
python scripts/representation_analysis.py --aggregate
python scripts/plot_mechanistic.py
```
