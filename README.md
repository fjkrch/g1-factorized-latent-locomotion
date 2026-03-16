# DynaMITE: Dynamic Mismatch Inference via Transformer Encoder for Robust Humanoid Locomotion

We study whether a short-horizon transformer encoder can infer a factorized latent representation of hidden dynamics parameters — friction, mass, motor strength, contact properties, and actuation delay — from proprioceptive history alone.
DynaMITE conditions both the policy and value function on this latent vector and trains per-factor auxiliary identification losses during PPO optimization.
We evaluate on a Unitree G1 humanoid in Isaac Lab across four locomotion tasks with domain randomization.
DynaMITE achieves the best evaluation reward on the *randomized* and *push* tasks, outperforming an MLP, LSTM, and vanilla Transformer baseline.
It does not uniformly dominate: the MLP is strongest on flat terrain, and differences on the terrain task are within noise.
Results are from a single seed; multi-seed validation is needed before drawing strong conclusions.

---

## Contributions

1. **Factorized latent dynamics inference.** A transformer encoder maps an 8-step (160 ms) observation–action history to a 24-d latent vector decomposed into 5 semantically-aligned subspaces (friction, mass, motor, contact, delay).

2. **Per-factor auxiliary supervision.** Each subspace is trained against the corresponding ground-truth dynamics parameter via MSE loss during training. At deployment, no privileged information is required.

3. **Controlled comparison** of MLP, LSTM, Transformer, and DynaMITE policies under identical PPO training on the same four tasks, with shared observation/action embeddings, policy heads, and value heads.

4. **Single-GPU reproducible pipeline.** Full experiment (16 training runs + 7 ablations + evaluation + plotting) completes in ~3.5 hours on an RTX 4060 Laptop GPU.

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

All numbers: PPO, 512 envs, ~12M timesteps, seed 42, RTX 4060 Laptop GPU.
Rewards are penalty-based (negative); higher (less negative) is better.

### Evaluation Reward

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | **-4.65** | -4.41 | -4.35 | **-4.35** |
| LSTM | -4.70 | -4.87 | -4.79 | -4.79 |
| Transformer | -4.75 | -4.38 | -4.53 | -4.53 |
| DynaMITE | -4.83 | **-4.32** | **-4.27** | -4.45 |

DynaMITE is best on *randomized* (-4.27) and *push* (-4.32).
MLP is best on *flat* (-4.65) and *terrain* (-4.35).
The terrain result is within 0.10 of MLP; this difference is not necessarily significant given single-seed evaluation.

### Episode Length

| Method | Flat | Push | Randomized | Terrain |
|---|---|---|---|---|
| MLP | 17.0 | 23.7 | 25.2 | 25.2 |
| LSTM | 29.2 | 31.7 | 32.7 | 32.7 |
| Transformer | 21.7 | 18.7 | 19.9 | 19.9 |
| DynaMITE | 21.8 | 18.0 | 17.1 | 17.1 |

LSTM achieves the longest episodes. DynaMITE has the shortest on randomized/terrain, suggesting it accumulates less penalty per step rather than surviving longer.

### Ablation Study (Randomized Task)

**Caveat:** ablation variants were trained for 2M steps while the full model was trained for 12M steps. These deltas are therefore an *upper bound* on the true component contributions. Equal-budget ablations are needed to draw firm conclusions.

| Variant | Eval Reward | Δ vs Full |
|---|---|---|
| DynaMITE (Full, 12M) | -4.27 | — |
| Seq Len 4 (2M) | -4.35 | -0.08 |
| Seq Len 16 (2M) | -4.35 | -0.08 |
| No Latent (2M) | -4.49 | -0.22 |
| Single Latent (2M) | -5.03 | -0.76 |
| No Aux Loss (2M) | -5.08 | -0.82 |
| Depth 1 (2M) | -4.32 | -0.06 |
| Depth 4 (2M) | -4.76 | -0.50 |

Directionally: removing the auxiliary loss or collapsing to a single (unfactored) latent produces the largest degradation. Sequence length (4–16) has minimal effect. These trends are suggestive but not conclusive at unequal training budgets.

---

## Limitations

- **Single seed.** All reported numbers use seed 42. Method rankings could change with additional seeds. Multi-seed experiments are required before claiming statistical significance.
- **Unfair ablation comparison.** The full DynaMITE model was trained for 12M steps; ablation variants were trained for 2M steps. The ablation deltas overstate the true component contributions.
- **Narrow reward spread.** After 12M steps, all methods fall within [-4.27, -4.87] on randomized — a range of 0.60. Whether this difference is practically meaningful for a physical robot is unknown.
- **No sim-to-real transfer.** All experiments are in simulation (Isaac Lab). We have not validated on physical hardware.
- **No robustness sweeps.** Sweep infrastructure exists but sweep results were not successfully collected due to an environment teardown bug. Out-of-distribution robustness has not been quantified.
- **No disentanglement analysis.** We have not yet verified that the factored latent subspaces actually correlate with their intended dynamics parameters.
- **Reward is penalty-based.** The reward function sums several penalty terms. A method achieving -4.27 vs -4.53 is accumulating ~5% less penalty per step on average. The practical significance of this gap is unclear without real-world deployment.

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
| Single training run (12M steps) | ~10–13 min |
| All 16 main runs | ~2.5 hours |
| 7 ablations (2M steps each) | ~18 min |
| Full pipeline | ~3.5 hours |

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
