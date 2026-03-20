# Batch 2 Report: Domain Randomization Strengthening

## Hypothesis
Widening the training DR range (especially push velocity) to better cover the
OOD evaluation range [5.0, 8.0] will reduce push sensitivity and improve
worst-case performance.

## Root Cause Analysis
The baseline DynaMITE trains with `push_vel_range: [0.5, 2.0]` but the OOD
push_magnitude sweep evaluates up to `[5.0, 8.0]` — a **4× extrapolation gap**.
This was identified as the primary driver of push sensitivity (0.248 baseline).

## Variants Tested

| Variant | Push Range | Other Changes | Training Peaks |
|---------|-----------|---------------|----------------|
| V1 wider_push | [0.5, 5.0] | None | -4.894 @ 7.1M |
| V2 wider_all | [0.5, 5.0] | friction [0.1,2.5], delay [0,5] | -4.842 @ 8.7M |
| V3 aggressive_push | [0.5, 8.0] | push_steps [3,5,8,10,15] | -5.571 @ 8.7M |

## Screen Phase (Seed 42 only, push_magnitude sweep)

| Metric | Baseline | V1 | V2 | V3 |
|--------|----------|-----|-----|-----|
| Nominal [0,0] | -4.280 | -4.402 | **-4.379** | -4.418 |
| Worst [5,8] | -4.532 | -4.719 | **-4.576** | -4.763 |
| Sensitivity | 0.252 | 0.317 | **0.197** | 0.345 |
| Δ nominal | — | -0.122 | **-0.099** | -0.138 |
| Δ sensitivity | — | +25.8% | **-21.7%** | +37.0% |

### Screen Decisions
- **V1 REJECT**: All metrics worse
- **V2 PROMOTE**: 21.7% sensitivity reduction, nominal loss 0.099 (≤0.10)
- **V3 REJECT**: All metrics worse

## Confirmation Phase (V2 wider_all: 3 seeds × 3 sweeps)

### Per-Seed Push Magnitude Results

| Seed | BL Nom | BL Worst | BL Sens | V2 Nom | V2 Worst | V2 Sens | Δ Sens% |
|------|--------|----------|---------|--------|----------|---------|---------|
| 42 | -4.280 | -4.532 | 0.252 | -4.379 | -4.576 | 0.197 | -21.6% |
| 43 | -4.408 | -4.608 | 0.200 | -4.318 | -4.601 | 0.284 | +41.8% |
| 44 | -4.236 | -4.529 | 0.293 | -4.668 | -5.075 | 0.407 | +38.8% |

### Aggregated RobustScore

| Configuration | RobustScore | 95% CI |
|--------------|-------------|--------|
| Baseline DynaMITE | **-3.086** | ±0.098 |
| V2 wider_all | -3.216 | ±0.421 |

- **Δ RobustScore: -0.130** (WORSE)
- Paired t-test: t=-1.14, **p=0.372** (NOT significant)

### All Decision Criteria FAILED
- ✗ Worst-case: -0.194 worse (need ≥+0.15 better)
- ✗ Sensitivity: +19.2% increase (need ≥15% decrease)
- ✗ Nominal loss: 0.147 (need ≤0.10)
- ✗ RobustScore declined

## **DECISION: REJECT**

## Key Findings

1. **Wider DR hurts more than it helps.** Training on harder distributions
   degrades nominal performance without systematically improving worst-case.

2. **Massive seed variance with wider DR.** V2's 95% CI (±0.421) is 4.3×
   wider than baseline (±0.098), indicating less stable training.

3. **Single-seed screens are unreliable (again).** Seed 42 showed -21.7%
   sensitivity improvement; seeds 43/44 showed +40% worsening. This mirrors
   the Batch 1 finding.

4. **The gap is not about training coverage.** Even V3 which trains up to
   [0.5, 8.0] (covering the full eval range) performs worse, not better.

## Implications for Future Batches
The problem is not insufficient DR range. The model needs to **better identify
and adapt to** disturbances, not just be trained on more extreme ones. This
points toward:
- Auxiliary loss modifications (Batch 3)
- Latent inference quality improvements (Batch 4)
- Architecture refinements (Batch 5)

## Compute Cost
- 5 training runs × ~19 min = ~95 min
- 12 evaluations × ~3 min = ~36 min
- Total: ~131 min (~2.2 hours)
