# Expected Results Reference

## Disclaimer

These are approximate expected values based on typical sim-to-sim transfer
performance for humanoid locomotion. Actual values depend on Isaac Sim/Lab
version, GPU, and exact randomization seed behavior.

## Main Comparison (Mean Reward, higher = better)

| Method       | Flat     | Push      | Randomized | Terrain  |
|-------------|----------|-----------|------------|----------|
| PPO + MLP   | ~150-200 | ~80-120   | ~60-100    | ~90-130  |
| PPO + LSTM  | ~160-220 | ~100-140  | ~80-130    | ~100-150 |
| PPO + Trans | ~170-230 | ~110-150  | ~90-140    | ~110-160 |
| DynaMITE    | ~180-240 | ~130-170  | ~110-160   | ~130-180 |

## Expected Improvements (DynaMITE vs baselines)

- vs MLP on randomized: ~30-60% improvement
- vs LSTM on randomized: ~15-30% improvement
- vs Transformer on randomized: ~10-25% improvement
- vs MLP on flat: ~5-15% (smaller gap expected)

## Expected Training Curves

- All methods should reach >80% of final performance by 25M steps
- DynaMITE may learn slightly slower initially (latent warming up)
- MLP converges fastest on flat terrain (simpler model, simpler task)
- Gap between methods widens as task difficulty increases

## Ablation Expected Ordering (randomized task)

1. DynaMITE (full) — best
2. Single latent (no factorization) — slight drop (~5-10%)
3. Factorized, no aux loss — moderate drop (~10-15%)
4. No latent (= vanilla transformer) — larger drop (~15-25%)

## Acceptable Variance

- Std across 3 seeds: typically 5-15% of mean reward
- If std > 30% of mean, something may be wrong
- One outlier seed out of 3 is acceptable but should be noted

## Robustness Sweeps

- All methods degrade with increasing perturbation
- DynaMITE should degrade most gracefully (slower drop-off curve)
- MLP should degrade fastest
- Key comparison: friction=0.3 and push_vel=5.0 (challenging conditions)
