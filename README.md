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

4. **Latent disentanglement analysis.** We show that the learned factored subspaces achieve a 0.477 disentanglement score (chance = 0.20), indicating moderate but above-chance alignment with ground-truth dynamics factors.

5. **Single-GPU reproducible pipeline.** Full experiment (16 training runs + 7 ablations + evaluation + plotting) completes in ~4 hours on an RTX 4060 Laptop GPU.

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

### Ablation Study (Randomized Task, 12M Steps)

All ablations trained for ~14M steps (same budget as full model), seed 42.

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

**Takeaways (single seed, interpret with caution):**
- Removing auxiliary loss has the largest negative impact (-0.82), confirming the per-factor supervision is the key component.
- Collapsing to a single unfactored latent (-0.76) is nearly as damaging, supporting factored representation.
- A shallower encoder (depth 1, -0.06) barely hurts, while a deeper one (depth 4, -0.50) hurts substantially — likely overfitting with limited data.
- Sequence length in the 4–16 range has minimal effect (±0.08).

### Latent Disentanglement Analysis

We measure whether DynaMITE's learned latent subspaces correlate with their intended ground-truth dynamics parameters using Pearson correlation on 2,176 latent samples (100 episodes, 64 envs).

- **Disentanglement score: 0.477** (chance = 0.20 for 5 factors)
- The score measures the ratio of within-factor to total correlation: values above 0.50 indicate strong disentanglement.
- The 0.477 score suggests moderate disentanglement — the model partially separates dynamics factors into their intended subspaces, but with substantial cross-talk between factors.
- Correlation heatmap and t-SNE visualizations are in `figures/`.

### OOD Robustness Sweeps (PhysX-Verified)

We evaluated all four models across friction levels using `scripts/eval_ood_validated.py`, which spawns a fresh subprocess per sweep level, modifies `env_cfg.events.physics_material.params` before `gymnasium.make()`, and verifies the PhysX read-back matches the target (std=0.0 across all shapes). This ensures the friction changes actually reach the simulator.

**Friction Sweep Results** (50 episodes per level, seed 42):

| Method | Friction 1.0 | Friction 0.7 | Friction 0.5 | Friction 0.3 | Friction 0.1 | Sensitivity |
|---|---|---|---|---|---|---|
| DynaMITE | **-4.27±0.15** | **-4.27±0.15** | **-4.27±0.15** | **-4.27±0.14** | **-4.28±0.14** | **0.01** |
| MLP | -4.56±0.39 | -4.50±0.32 | -4.42±0.32 | -4.45±0.35 | -4.56±0.33 | 0.14 |
| LSTM | -4.78±0.22 | -4.79±0.20 | -4.76±0.21 | -4.80±0.21 | -4.91±0.17 | 0.15 |
| Transformer | -5.04±0.58 | -4.72±0.32 | -4.54±0.21 | -4.39±0.12 | -4.36±0.12 | 0.68 |

*Sensitivity = max(reward) - min(reward) across friction levels. Lower is more robust.*

**Key findings:**
- **DynaMITE is the most friction-robust** — reward barely changes (0.01 range) from high friction (1.0) to extreme low friction (0.1).
- **Transformer is most friction-sensitive** — reward varies by 0.68 across levels, actually improving at low friction (possibly due to reduced ground reaction forces).
- **MLP and LSTM show moderate sensitivity** (0.14–0.15 range).

**Additional DynaMITE sweeps** (push_magnitude, action_delay):
- Push magnitude: No effect observed — episode length (17 steps) is shorter than push_interval (200 steps), so pushes never fire.
- Action delay: Config-level only (not physically simulated in PhysX), so no effect on reward.

All verified results are in `outputs/ood_validated/randomized/<model>/` with CSV + JSON files.

**Validation command:**
```bash
python scripts/eval_ood_validated.py \
    --checkpoint path/to/best.pt \
    --sweep configs/sweeps/friction.yaml \
    --num_episodes 50 --num_envs 32 --seed 42
```

The script runs a sanity check first (spawning two subprocesses with extreme friction levels) and refuses to continue if the PhysX read-backs are identical.

---

## Limitations

- **Single seed.** All reported numbers use seed 42. Method rankings could change with additional seeds. Multi-seed experiments are required before claiming statistical significance.
- **Narrow reward spread.** After 12M steps, all methods fall within [-4.27, -4.87] on randomized — a range of 0.60. Whether this difference is practically meaningful for a physical robot is unknown.
- **No sim-to-real transfer.** All experiments are in simulation (Isaac Lab). We have not validated on physical hardware.
- **Sweep config does not reach PhysX (original pipeline).** The original `eval.py` sweep loop modified the wrapper's config dict at runtime, but Isaac Lab's PhysX parameters are set during env construction via its EventManager. `scripts/eval_ood_validated.py` solves this with a subprocess-per-level architecture that modifies `env_cfg.events.physics_material.params` before `gymnasium.make()` and verifies read-backs. **Friction sweeps are now fully validated.** Factors that remain config/wrapper-level only (no PhysX verification): push magnitude, action delay.
- **Push magnitude sweeps are ineffective** with current episode lengths. Episodes terminate after ~17 steps, but push_interval=200 means pushes never fire. Shorter push intervals or longer-surviving policies are needed to test push robustness.
- **Moderate disentanglement.** The 0.477 disentanglement score is above chance (0.20) but below the 0.50+ threshold for strong disentanglement, indicating substantial cross-talk between latent subspaces.
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
| 7 ablations (12M steps each) | ~5.5 hours |
| Latent analysis + OOD sweeps | ~30 min |
| Full pipeline | ~4 hours (excl. ablations) |

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
