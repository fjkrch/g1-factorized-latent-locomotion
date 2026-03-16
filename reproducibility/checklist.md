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
- [ ] Reward is increasing (at least initially)

## Post-Run Checks

- [ ] Final checkpoint saved
- [ ] `latest.pt` and `best.pt` exist
- [ ] `eval_metrics.json` generated
- [ ] Manifest updated with `status: completed`
- [ ] Training curves plotted
- [ ] Results consistent with expectations (within variance)

## Multi-Seed Checks

- [ ] All 3 seeds completed successfully
- [ ] Aggregated results generated
- [ ] Std across seeds < 20% of mean (reasonable variance)
- [ ] No outlier seeds (single seed much worse than others)

## Cross-Method Comparison Checks

- [ ] All methods trained on same tasks with same seeds
- [ ] Same total timesteps for all methods
- [ ] Same evaluation protocol (deterministic, same num episodes)
- [ ] Same reward function and observation space
- [ ] Results tables generated
- [ ] Figures generated

## Full Reproduction Checks

- [ ] All main experiments (4 methods x 4 tasks x 3 seeds = 48 runs)
- [ ] All ablations (7 ablations x 3 seeds = 21 runs)
- [ ] Robustness sweeps completed
- [ ] Tables match expected format
- [ ] Figures generated and saved
- [ ] No missing runs in aggregation
- [ ] Latent analysis plotted (for DynaMITE)
