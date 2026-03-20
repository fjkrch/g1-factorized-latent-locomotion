# Batch 1: Checkpoint Selection — Experiment Report

## Hypothesis
The best-nominal checkpoint (`best.pt`, ~6.6M steps) may not be optimal for OOD robustness. Earlier checkpoints might trade nominal performance for better worst-case and sensitivity.

## Design
- **Screen (Phase 1)**: Evaluate 5 checkpoints from seed 42 on push_magnitude sweep
- **Confirm (Phase 2)**: Promote checkpoints passing thresholds to 3-seed evaluation across all 3 sweep types

### Checkpoints Evaluated
| Checkpoint | Training Step | % of Training | Training Reward |
|---|---|---|---|
| ckpt_4300800 | 4.3M | 35% | −5.07 |
| ckpt_5529600 | 5.5M | 45% | −4.79 |
| best.pt | ~6.6M | 54% | −4.73 (peak) |
| ckpt_7372800 | 7.4M | 60% | −4.77 |
| ckpt_9216000 | 9.2M | 75% | −4.90 |

## Screen Results (Seed 42 Only)

| Checkpoint | Nominal | Worst | Sensitivity | Decision |
|---|---|---|---|---|
| ckpt_4300800 | −4.580 | −4.798 | 0.218 | Reject (nominal −0.30) |
| **ckpt_5529600** | −4.373 | −4.536 | 0.163 | **Promote** (sens −35%) |
| best.pt | −4.280 | −4.532 | 0.252 | Baseline |
| **ckpt_7372800** | −4.331 | −4.538 | 0.207 | **Promote** (sens −18%) |
| ckpt_9216000 | −4.475 | −4.666 | 0.190 | Reject (nominal −0.20) |

## 3-Seed Confirmation Results

### Per-Sweep Comparison (3-seed mean)

**Friction:**
| Checkpoint | Nominal | Worst | Sensitivity |
|---|---|---|---|
| best.pt | −4.402 | −4.402 | 0.000 |
| ckpt_5529600 | −4.449 | −4.449 | 0.000 |
| ckpt_7372800 | −4.476 | −4.476 | 0.000 |

**Push Magnitude:**
| Checkpoint | Nominal | Worst | Sensitivity |
|---|---|---|---|
| best.pt | −4.308 | −4.556 | 0.248 |
| ckpt_5529600 | −4.368 | −4.795 | 0.427 |
| ckpt_7372800 | −4.387 | −4.808 | 0.422 |

**Action Delay:**
| Checkpoint | Nominal | Worst | Sensitivity |
|---|---|---|---|
| best.pt | −4.405 | −4.409 | 0.004 |
| ckpt_5529600 | −4.464 | −4.485 | 0.022 |
| ckpt_7372800 | −4.481 | −4.484 | 0.003 |

### RobustScore Comparison

| Checkpoint | RobustScore | Δ vs Baseline |
|---|---|---|
| best.pt | **−3.119** | — |
| ckpt_5529600 | −3.177 | −0.058 |
| ckpt_7372800 | −3.187 | −0.068 |

### Paired t-tests

| Comparison | Mean Diff | t | p | Significant? |
|---|---|---|---|---|
| ckpt_5529600 vs best.pt | −0.111 | −1.11 | 0.383 | No |
| ckpt_7372800 vs best.pt | −0.122 | −1.67 | 0.236 | No |

## Decision: **REJECT**

Both promoted checkpoints are worse than best.pt on RobustScore when evaluated across 3 seeds. The screen phase was misleading because seed 42 was atypical.

**Key Finding:** The nominal-best checkpoint is simultaneously the OOD-best checkpoint. DynaMITE's robustness is well-correlated with nominal performance — there is no robustness/nominal tradeoff in checkpoint space.

**Lesson:** Multi-seed confirmation is essential. Single-seed screens can identify false positives due to per-seed variability.

## Proceed To
**Batch 2: Randomization Strengthening** — the next prescribed intervention, which involves training new policies with wider domain randomization ranges.
