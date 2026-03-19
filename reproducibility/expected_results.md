# Expected Results Reference

## Disclaimer

These are the actual results measured across our full experiment campaign.
Exact values depend on Isaac Sim/Lab version, GPU, and random seed.
Rewards are penalty-based (negative); higher (less negative) is better.

## Main Comparison (Mean Reward ± Std, 5 seeds, deterministic 100-episode eval)

Training: PPO, 512 envs, 10M timesteps, RTX 4060 Laptop GPU.

| Method       | Flat           | Push           | Randomized     | Terrain        |
|-------------|----------------|----------------|----------------|----------------|
| MLP         | -4.83 ± 0.12  | -5.01 ± 0.29  | -5.32 ± 0.45  | -4.82 ± 0.26  |
| LSTM        | -4.01 ± 0.04  | -4.30 ± 0.04  | -4.18 ± 0.04  | -4.06 ± 0.04  |
| Transformer | -5.02 ± 0.32  | -4.83 ± 0.62  | -4.77 ± 0.37  | -4.46 ± 0.11  |
| DynaMITE    | -4.88 ± 0.23  | -4.60 ± 0.12  | -4.48 ± 0.14  | -4.49 ± 0.13  |

**LSTM wins all four tasks** with the lowest seed variance (σ ≤ 0.04).
DynaMITE ranks second on push, randomized, and terrain.

## Ablation Expected Ordering (randomized task, 3 seeds)

| Variant                      | Eval Reward     | Δ vs Full |
|------------------------------|-----------------|-----------|
| DynaMITE (Full, 5-seed)     | -4.48 ± 0.14   | —         |
| No Latent                   | -4.88 ± 0.27   | -0.40     |
| No Aux Loss                 | -5.06 ± 0.58   | -0.58     |
| Single Latent (unfactored)  | -5.25 ± 0.36   | -0.77     |

Single Latent (unfactored) shows the largest degradation, confirming the factored structure is the most critical design choice.

## Latent Disentanglement (3 seeds)

| Seed | Disentanglement Score |
|------|----------------------|
| 42   | 0.496                |
| 43   | 0.482                |
| 44   | 0.521                |
| Mean | 0.500 ± 0.020        |

Chance level = 0.20 for 5 factors. Score ≥ 0.50 indicates strong disentanglement.

## OOD Robustness (DynaMITE vs LSTM, 3 seeds)

| Sweep           | DynaMITE Sensitivity | LSTM Sensitivity | DynaMITE Advantage |
|-----------------|---------------------|------------------|--------------------|
| Friction        | 0.03                | 0.20             | 6.7×               |
| Push magnitude  | 0.25                | 1.39             | 5.6×               |
| Action delay    | 0.02                | 0.05             | 2.5×               |

Sensitivity = max(mean reward) − min(mean reward). Lower is more robust.

## Acceptable Variance

- Main comparison: σ ≤ 0.04 (LSTM) to σ ≤ 0.62 (Transformer on push)
- Ablations (3 seeds): σ ≤ 0.58
- If a new seed deviates by more than 1.0 from the above means, check for issues
- OOD sweeps: σ ≤ 0.15 across seeds at each perturbation level
