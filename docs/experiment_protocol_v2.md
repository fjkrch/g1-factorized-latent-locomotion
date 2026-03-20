# Experiment Protocol v2 — Post-Batch-2 Methodology Overhaul

**Effective from:** Batch 3 onwards  
**Motivation:** Batches 1–2 both screened on single seed 42 and confirmed on only 3 seeds. Both were REJECTED after confirmation revealed seed 42 was atypically optimistic. This protocol addresses 15 specific methodology failures identified in the post-Batch-2 review.

---

## 1. Screening Phase: 2-Seed Minimum

### Problem (Batches 1–2)
Single-seed (seed 42) screening produced false positives:
- Batch 1: ckpt_5529600 showed −35% sensitivity on seed 42; 3-seed confirmation showed +72% worsening.
- Batch 2: V2 showed −21.7% sensitivity on seed 42; seeds 43/44 showed +40% worsening.

### New Protocol
- **Minimum 2 seeds** for screening (seeds 42, 43).
- A variant passes screening only if **both seeds** show improvement or at most one shows marginal degradation (Δ < +5%).
- Compute cost: 2 trains (~38 min) + 2 push sweeps (~40s) per variant.
- **Max 3 variants** per screen batch (6 training runs total).

### Rationale
2-seed screening catches the most common failure mode (single-seed fluke) at 2× cost. The marginal cost (~19 min/seed) is far cheaper than false promotion to 5-seed confirmation.

---

## 2. Confirmation Phase: 5-Seed Minimum

### Problem
3-seed confirmation has wide CIs and low statistical power (p ≤ 0.05 requires very large effects).

### New Protocol
- **5 seeds minimum** (seeds 42–46) for confirmation.
- Seeds chosen to match main experiment seed set for direct pairing.
- Report paired t-test with per-seed deltas as primary evidence.
- **8 seeds** if initial 5-seed p ∈ [0.05, 0.15] (add seeds 47–49).
- Compute cost: 5 trains (~95 min) + 5×3 sweeps (~5 min) per confirmed variant.

### Decision Criteria (5-seed)
| Criterion | Threshold | Weight |
|-----------|-----------|--------|
| Worst-case improvement | Δ ≥ +0.10 | Required |
| Sensitivity reduction | ≥ 10% decrease | Required |
| Nominal degradation | Δ ≤ −0.15 | Hard veto |
| RobustScore gain | Δ > 0 | Required |
| Paired t-test | p ≤ 0.10 | Required for ACCEPT |

---

## 3. Single-Factor Intervention Design

### Problem (Batch 2)
V2 (wider_all) changed push range + friction range + action delay simultaneously.  
V3 (aggressive_push) changed push range + push_steps (temporal structure).  
Confounding made it impossible to attribute effects.

### New Protocol
- **One factor per variant.** Each variant changes exactly one training parameter relative to baseline.
- If a combination is hypothesized to help, test each factor individually first, then test the additive combination only if both factors show independent improvement.
- Document the **causal claim** for each variant before training.

### Example (Batch 3 — Aux Loss Tuning)
- A1: `aux_weight = 1.0` (vs baseline 0.5). Single factor: aux loss weight.
- A2: `aux_weight = 2.0`. Single factor: aux loss weight.
- No other hyperparameters changed. ✓ Clean design.

---

## 4. Broader Intervention Types

### Problem
Batches 1–2 only tried "move training closer to test distribution" approaches (checkpoint selection, wider DR). These assume the gap is insufficient coverage, but Batch 2 disproved this — V3 covered full eval range and still performed worse.

### Intervention Categories to Explore
1. **Curriculum widening** — gradual DR expansion during training (not fixed wide from start)
2. **Mixture sampling** — train on mix of nominal + extreme (e.g., 70/30 split) instead of uniform random
3. **Tail upweighting** — sample harder conditions more frequently but keep nominal in the mix
4. **Adversarial sampling** — use worst-case performance to guide the DR sampling
5. **Auxiliary loss modifications** — change what the latent is trained to encode (Batch 3)
6. **Latent architecture changes** — number of latents, information bottleneck width (Batch 4)
7. **Inference quality** — KL penalty, reconstruction targets, temporal context (Batch 5)

---

## 5. Separating Coverage from Optimization Damage

### Problem
Wider DR ranges hurt because they:
1. Reduce time spent on learnable conditions (wasted coverage on too-hard cases)
2. Increase gradient variance across the distribution
3. Push the optimizer toward compromise policies

### New Protocol
When testing DR changes, always include a **constant-range control** that matches the new sampling distribution's shape without extending the range:
- Experimental: push_vel_range [0.5, 5.0], uniform
- Control A: push_vel_range [0.5, 2.0], uniform (baseline)
- Control B: push_vel_range [0.5, 2.0], same distribution shape (e.g., if exp uses log-uniform, so does control B)

This isolates "wider range" effect from "distribution shape" effect.

---

## 6. Training Stability Metrics

### Problem
Batch 2 V2 had 4.3× wider CI than baseline but this was only discovered post-hoc.

### New Protocol — Log and Report
| Metric | How | Purpose |
|--------|-----|---------|
| Cross-seed reward variance | std(best_reward) across seeds | Training stability |
| Collapse rate | % seeds where best_reward > mean + 2×std | Detects outlier seeds |
| Best checkpoint timing | mean ± std of best_step across seeds | Convergence consistency |
| Return trajectory std | std of moving-average reward across training | Learning curve stability |
| Final 1M step slope | Linear fit slope of last 1M steps | Converged vs still declining |

Report these in every batch report alongside eval metrics.

---

## 7. Multi-Axis OOD Evaluation

### Problem
Screening used only push_magnitude sweep. A variant might improve push robustness but worsen friction or delay robustness.

### New Protocol
- **Screen phase**: push_magnitude sweep only (cheapest, most discriminative).
- **Confirmation phase**: ALL 3 sweep axes (push_magnitude, friction, motor_strength) + action_delay if variant modifies temporal processing.
- **RobustScore** computed from all axes, not just push.
- Add **combined perturbation** eval: simultaneous push + friction at moderate levels.

---

## 8. Nominal Performance Retention

### Problem
Post-hoc rejection when nominal degrades is wasteful. Better to build retention into the training objective.

### Approaches (for future batches)
- **Mixture training**: Dedicate fixed fraction (e.g., 30%) of episodes to nominal conditions
- **Constrained optimization**: Add penalty term if nominal performance drops below threshold
- **Curriculum**: Start with nominal-only, gradually add perturbations

---

## 9. Checkpoint Selection Standardization

### Problem
Using `best.pt` (highest single-evaluation reward during training) introduces selection bias toward lucky checkpoints.

### New Protocol
- Primary: `best.pt` from training (for comparability with main results).
- Secondary: Average of last 3 checkpoints' eval rewards. Report if significantly different from best.pt.
- If best.pt step differs by >30% across seeds, flag as unstable training.

---

## 10. Paired Per-Seed Analysis

### Problem
Mean ± CI hides per-seed structure. A variant might help one seed and hurt two.

### New Protocol
- **Always report per-seed table** with paired deltas.
- **Win/loss count**: How many seeds improved? (e.g., 4/5 or 2/5)
- **Effect direction consistency**: Sign of Δ for each seed. If mixed, report as "inconsistent effect, N+/N−".
- **Worst-seed check**: If worst individual seed's Δ < −0.20, flag as "unstable improvement".

---

## 11. Compute Allocation Strategy

### Problem
Batch 2 tested 3 variants × 1 seed (screen) → 1 variant × 3 seeds (confirm).
The 3 V1/V3 training runs were wasted when they failed screen.

### New Protocol
- **2 variants × 2 seeds** beats **4 variants × 1 seed** for the same 4 training runs.
- Max 3 variants per batch to keep compute ≤ 6 training runs for screening.
- Budget per batch: ≤ 6 screen trains + ≤ 5 confirm trains = ≤ 11 total (~3.5 hours).
- If none pass 2-seed screen, save compute — don't force confirmation.

---

## 12. Negative Controls

### Problem
No baseline for pipeline noise. Small RobustScore changes might be within the measurement floor.

### New Protocol
- **Negative control**: Retrain baseline with identical config but different random seed initialization order (e.g., seeds 50, 51). Compute RobustScore delta.
- Expected delta ≈ 0 (pipeline noise estimate).
- Any experimental variant must exceed the negative control's |Δ| to be considered real.
- Run once and reuse across batches.

---

## 13. Direct Adaptation Measurement

### Problem
RobustScore only captures reward outcomes. It doesn't measure whether the model is actually *adapting* to disturbances vs. being passively robust.

### New Metrics (implement in eval pipeline)
| Metric | Definition | Purpose |
|--------|-----------|---------|
| Recovery time | Steps to return to 90% of pre-push reward after push onset | Speed of adaptation |
| Latent shift magnitude | ‖z_post_push − z_pre_push‖₂ | Whether latent responds to disturbance |
| Latent consistency | std(z) within same condition across episodes | Whether latent encoding is stable |
| Action variance post-push | std(action) in 50 steps after push vs. steady state | Active compensation signal |
| State estimation error | If ground truth available, ‖z − z*‖ | Inference quality |

Implement these as optional eval flags for detailed analysis on promoted variants only (saves compute).

---

## 14. Revised Decision Tree

```
Batch N Start
    │
    ├── Design: 2-3 single-factor variants + causal claim per variant
    │
    ├── Screen (2 seeds × 2-3 variants)
    │   ├── Both seeds improve (or 1 marginal) → PROMOTE
    │   ├── Mixed results (1 good, 1 bad) → REJECT
    │   └── Both seeds worse → REJECT
    │
    ├── Confirm (5 seeds × promoted variant, all sweep axes)
    │   ├── RobustScore Δ > 0, p ≤ 0.10, ≥4/5 seeds improve → ACCEPT
    │   ├── p ∈ (0.10, 0.15), ≥3/5 seeds improve → EXTEND to 8 seeds
    │   └── Otherwise → REJECT
    │
    ├── If ACCEPT → Adaptation Diagnostics
    │   ├── Recovery time, latent shift, action variance
    │   └── Document mechanism of improvement
    │
    └── Report: per-seed table, training stability, mechanism analysis
```

---

## 15. Batch Report Template

Every batch report must include:

1. **Hypothesis** — What single factor is being tested and why?
2. **Causal claim** — "Changing X should improve Y because Z"
3. **Confound check** — What else changes? (Must be "nothing" for clean design)
4. **Training stability** — Cross-seed variance, convergence timing, collapse rate
5. **Screen results** — 2-seed table with per-seed deltas and decision
6. **Confirmation results** (if promoted) — 5+-seed table, paired t-test, per-seed win/loss count
7. **Multi-axis evaluation** — All sweep types with per-axis breakdown
8. **Adaptation metrics** (if accepted) — Recovery time, latent shift, etc.
9. **Decision** — ACCEPT/REJECT with explicit criteria evaluation
10. **Lessons** — What was learned, implications for future batches

---

## Application to Remaining Batches

### Batch 3 (Aux Loss Tuning) — Currently In Progress
- A1 (aux_weight=1.0) and A2 (aux_weight=2.0): Training done, seed 42 only.
- **Transition plan**: These were designed before v2 protocol. Train seed 43 for both variants, then apply 2-seed screening.
- Clean single-factor design ✓ (only aux_weight changes)

### Batch 4 (Latent Interventions) — Planned
- Factor options: latent_dim, num_latents, KL weight, reconstruction target
- Design under v2: 2 single-factor variants, 2-seed screen
- Must NOT conflate latent dim with num_latents

### Batch 5 (Architecture Refinements) — Planned
- Factor options: temporal context length, attention mechanism, information bottleneck
- Design under v2: 2 single-factor variants, 2-seed screen

### Negative Control — Run Once Before Batch 4
- Retrain baseline (identical config) with seeds 50, 51
- Evaluate on all sweeps, compute RobustScore
- Establish pipeline noise floor
