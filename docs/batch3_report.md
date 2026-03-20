# Batch 3 Report: Auxiliary Loss Tuning

**Protocol:** v2 (2-seed screening, single-factor design)  
**Date:** 2026-03-20

## Hypothesis
Increasing the auxiliary loss weight forces the latent to encode environment
dynamics more accurately, improving disturbance identification and adaptation
under OOD pushes.

## Causal Claim
Changing `model.auxiliary.loss_weight` from 0.5 (baseline) to 1.0 or 2.0
should improve latent quality → better adaptation → lower push sensitivity.

## Confound Check
Only `loss_weight` changes. All other hyperparameters identical to baseline. ✓ Clean single-factor design.

## Variants

| Variant | aux loss_weight | Factor change |
|---------|----------------|---------------|
| Baseline | 0.5 | — |
| A1 | 1.0 | 2× baseline |
| A2 | 2.0 | 4× baseline |

## Training (seeds 42, 43)

| Variant | Seed | Peak Train Reward | Best Step |
|---------|------|-------------------|-----------|
| A1 (aux=1.0) | 42 | ~-4.95 | ~5.3M |
| A1 (aux=1.0) | 43 | ~-4.94 | — |
| A2 (aux=2.0) | 42 | ~-4.84 | ~8.7M |
| A2 (aux=2.0) | 43 | ~-5.18 | — |

### Training Stability
- A1: Consistent convergence across both seeds, moderate reward levels.
- A2: Higher variance (seed 43 final reward worse at -5.18), later convergence.

## Screen Results (2-seed, push_magnitude sweep, 100 episodes per level)

| Config | Seed | Nominal [0,0] | Worst [5,8] | Sensitivity | Δ Nominal | Δ Sens% |
|--------|------|---------------|-------------|-------------|-----------|---------|
| Baseline | 42 | -4.280 | -4.532 | 0.252 | — | — |
| Baseline | 43 | -4.408 | -4.608 | 0.200 | — | — |
| A1 (aux=1.0) | 42 | -4.346 | -4.517 | 0.171 | -0.066 | **-32.1%** |
| A1 (aux=1.0) | 43 | -4.373 | -4.609 | 0.236 | +0.035 | **+18.0%** |
| A2 (aux=2.0) | 42 | -4.440 | -4.797 | 0.357 | -0.160 | +41.7% |
| A2 (aux=2.0) | 43 | -4.323 | -4.718 | 0.396 | +0.085 | +97.9% |

## Screen Decisions (Protocol v2)

### A1 (aux_weight = 1.0): **REJECT**
- Seed 42: sensitivity -32.1% ✓ improves, nominal -0.066 ✓ within threshold
- Seed 43: sensitivity +18.0% ✗ degrades
- **Mixed result → REJECT** (one seed improves, one degrades beyond +5% threshold)
- Note: This is exactly the Batch 1/2 failure mode — seed 42 shows promise but seed 43 doesn't replicate.

### A2 (aux_weight = 2.0): **REJECT**
- Seed 42: sensitivity +41.7% ✗ degrades, nominal -0.160 ✗ exceeds threshold
- Seed 43: sensitivity +97.9% ✗ degrades
- **Both seeds worse → clear REJECT**
- Higher aux weight significantly worsens OOD robustness, likely by biasing the latent toward auxiliary prediction at the expense of policy quality.

## **DECISION: REJECT BOTH**

Neither A1 nor A2 passes the 2-seed screen. No confirmation phase needed.

## Key Findings

1. **Stronger auxiliary loss does not improve OOD robustness.** A2 (4× baseline weight) dramatically worsens sensitivity (+42–98%). Even A1 (2× baseline) fails to replicate its seed-42 improvement on seed 43.

2. **Seed 42 is atypically optimistic (again).** A1 seed 42 showed -32.1% sensitivity improvement; seed 43 showed +18.0% worsening. This is the third batch where seed 42 gives misleading positive signals.

3. **Protocol v2 catches the false positive.** Under the old single-seed protocol, A1 would have been promoted to 3-seed confirmation (wasting ~57 min of compute). The 2-seed screen correctly identified the inconsistency at a cost of only ~19 min (one extra training run).

4. **Auxiliary loss weight is not the bottleneck.** The baseline aux_weight=0.5 appears to be at or near optimal. The latent already encodes sufficient dynamics information — the problem is not *what* the latent encodes but *how the policy uses it* under distribution shift.

## Implications for Future Batches

The first three batches have systematically eliminated several hypotheses:
- **Batch 1** (checkpoint selection): Robustness is correlated with nominal performance; no checkpoint tradeoff exists.
- **Batch 2** (DR widening): More extreme training doesn't improve adaptation; it degrades nominal without helping worst-case.
- **Batch 3** (aux loss): Stronger latent supervision doesn't improve adaptation; the latent already encodes sufficient dynamics information.

The remaining viable directions are:
- **Batch 4 (Latent architecture):** Change the latent's *capacity* or *structure* (e.g., larger latent dim, additional latent heads for push-specific features).
- **Batch 5 (Information bottleneck / inference quality):** Change how the latent is *inferred* at test time (e.g., temporal context, KL regularization, explicit uncertainty quantification).

## Compute Cost
- 2 training runs (seed 43 for A1, A2) × ~19 min = ~38 min
- 4 sweep evaluations × ~1 min = ~4 min
- Total: ~42 min
- Total Batch 3 including seed 42 (from previous session): ~80 min (~1.3 hours)
