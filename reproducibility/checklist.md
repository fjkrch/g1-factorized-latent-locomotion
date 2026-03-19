# DynaMITE Reproducibility Checklist

## Pre-Run Checks

- [ ] Git repo is clean (no uncommitted changes)
- [ ] `git rev-parse HEAD` recorded
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
- [ ] PyTorch version recorded: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA version recorded: `python -c "import torch; print(torch.version.cuda)"`
- [ ] Isaac Lab importable: `python -c "from omni.isaac.lab.envs import ManagerBasedRLEnv; print('OK')"`
- [ ] Config files unmodified since last check
- [ ] Seeds are set correctly in config
- [ ] Output directory is clean or resume is intentional
- [ ] Disk space > 50 GB free

## Per-Run Checks

- [ ] `config.yaml` saved in run directory
- [ ] `manifest.json` saved in run directory
- [ ] TensorBoard events appearing in `tb/`
- [ ] `metrics.csv` growing
- [ ] Checkpoints being saved at expected intervals
- [ ] No NaN/Inf in loss values
- [ ] Reward is decreasing in magnitude (less negative over time)

## Post-Run Checks

- [ ] Final checkpoint saved
- [ ] `latest.pt` and `best.pt` exist
- [ ] `eval_metrics.json` generated
- [ ] Manifest updated with `status: completed`
- [ ] Training curves plotted
- [ ] Results consistent with expected_results.md (within variance)

## Multi-Seed Checks

- [ ] Multiple seeds (3–5) completed successfully
- [ ] Aggregated results generated (`python scripts/aggregate_seeds.py`)
- [ ] Std across seeds reasonable (see expected_results.md for baselines)
- [ ] No outlier seeds (single seed much worse than others)

## Cross-Method Comparison Checks

- [ ] All methods trained on same tasks with same seeds
- [ ] Same total timesteps (10M) for all methods
- [ ] Same evaluation protocol (deterministic, 100 episodes)
- [ ] Same reward function and observation space
- [ ] Results tables generated (`python scripts/generate_tables.py`)
- [ ] Figures generated (`python scripts/plot_results.py`)

## Full Reproduction Checks

- [ ] All main experiments (4 methods × 4 tasks × 5 seeds = 80 runs)
- [ ] Multi-seed ablations (3 variants × 3 seeds = 9 runs)
- [ ] Single-seed ablations (4 variants × 1 seed = 4 runs)
- [ ] OOD robustness sweeps (2 models × 3 sweep types × 3 seeds = 18 runs)
- [ ] Latent disentanglement analysis (3 seeds)
- [ ] Tables match expected format
- [ ] Figures generated and saved to `figures/`
- [ ] No missing runs in aggregation
