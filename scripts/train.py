#!/usr/bin/env python3
"""
Training entrypoint for DynaMITE project.

Usage:
    # MLP baseline on flat terrain
    python scripts/train.py --task configs/task/flat.yaml --model configs/model/mlp.yaml

    # DynaMITE on randomized dynamics
    python scripts/train.py --task configs/task/randomized.yaml --model configs/model/dynamite.yaml

    # With ablation overlay
    python scripts/train.py --task configs/task/randomized.yaml --model configs/model/dynamite.yaml \
        --ablation configs/ablations/no_latent.yaml --variant no_latent

    # With CLI overrides
    python scripts/train.py --task configs/task/push.yaml --model configs/model/dynamite.yaml \
        --set seed=123 train.total_timesteps=10000000

    # Resume from checkpoint
    python scripts/train.py --task configs/task/flat.yaml --model configs/model/dynamite.yaml \
        --resume outputs/flat/dynamite_full/seed_42/20260316_120000

Inputs:
    - Config YAML files (base + task + model + optional train + optional ablation)
    - Optional: checkpoint directory for resuming

Outputs (in run_dir = outputs/{task}/{model}_{variant}/seed_{seed}/{timestamp}/):
    - config.yaml          : saved merged config
    - manifest.json        : experiment manifest (rich system/git metadata)
    - metrics.csv          : standardized 14-column step-level metrics
    - tb/                  : TensorBoard logs
    - checkpoints/         : model checkpoints
        - ckpt_{step}.pt
        - latest.pt
        - best.pt
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.config import load_config, make_run_dir, save_config
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint
from src.utils.metrics_io import write_step_header
from src.utils.run_naming import make_run_id
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.algos.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DynaMITE Training")
    parser.add_argument("--base", type=str, default="configs/base.yaml",
                        help="Base config file")
    parser.add_argument("--task", type=str, required=True,
                        help="Task config file (e.g., configs/task/flat.yaml)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model config file (e.g., configs/model/dynamite.yaml)")
    parser.add_argument("--train", type=str, default="configs/train/default.yaml",
                        help="Training config file")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Optional ablation config overlay")
    parser.add_argument("--variant", type=str, default="full",
                        help="Experiment variant tag (e.g., full, no_latent, seq_len_4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from run directory")
    parser.add_argument("--set", nargs="*", default=[],
                        help="CLI config overrides: key=value")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Isaac Sim headless (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Run Isaac Sim with GUI")
    return parser.parse_args()


def resolve_config_path(value: str, config_dir: str, label: str) -> str:
    """Resolve shorthand name to config file path. E.g. 'flat' -> 'configs/task/flat.yaml'."""
    if Path(value).exists():
        return value
    candidate = Path(f"configs/{config_dir}/{value}.yaml")
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Cannot resolve --{label} '{value}'. "
        f"Expected a valid path or a name matching configs/{config_dir}/<name>.yaml"
    )


def main():
    args = parse_args()

    # Resolve shorthand names to paths
    args.task = resolve_config_path(args.task, "task", "task")
    args.model = resolve_config_path(args.model, "model", "model")
    if args.ablation:
        args.ablation = resolve_config_path(args.ablation, "ablations", "ablation")

    # Load and merge configs
    cfg = load_config(
        base_path=args.base,
        task_path=args.task,
        model_path=args.model,
        train_path=args.train,
        ablation_path=args.ablation,
        overrides=args.set,
    )

    if args.seed is not None:
        cfg["seed"] = args.seed

    variant = args.variant

    # Set seed
    set_seed(cfg["seed"], deterministic=False)

    # Create run directory (with variant in path)
    if args.resume:
        run_dir = Path(args.resume)
        print(f"[Resume] Resuming from: {run_dir}")
    else:
        run_dir = make_run_dir(cfg, variant=variant)
        print(f"[Train] Run directory: {run_dir}")

    # Generate deterministic run ID
    run_id = make_run_id(cfg, variant=variant)
    print(f"[Train] Run ID: {run_id}")

    # Initialize standardized metrics CSV
    metrics_csv_path = run_dir / "metrics.csv"
    if not args.resume:
        write_step_header(metrics_csv_path)

    # Logger (TensorBoard + console)
    logger = Logger(run_dir)
    logger.log_config(cfg)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"[Train] GPU Memory: {gpu_mem} MB")

    # Build model
    model = build_model(cfg)
    model.to(device)

    # Initialise Isaac Sim (headless) — must happen before env creation
    init_sim(headless=args.headless)

    # Build environment
    env = make_env(cfg, device=device, headless=args.headless)

    # Build trainer — pass variant and metrics CSV path
    trainer = PPOTrainer(cfg, model, env, logger, str(run_dir),
                         variant=variant, metrics_csv_path=str(metrics_csv_path))

    # Resume if specified
    if args.resume:
        ckpt_path = find_latest_checkpoint(run_dir)
        if ckpt_path:
            print(f"[Resume] Loading checkpoint: {ckpt_path}")
            model, trainer.optimizer, trainer.global_step, stats = load_checkpoint(
                ckpt_path, model, trainer.optimizer, device=device
            )
            print(f"[Resume] Resumed at step {trainer.global_step}")
        else:
            print("[Resume] No checkpoint found. Starting from scratch.")

    # Train
    print(f"[Train] Starting training: {cfg['model']['name']} on {cfg['task']['name']}")
    print(f"[Train] Variant: {variant} | Seed: {cfg['seed']}")
    print(f"[Train] Total timesteps: {cfg['train']['total_timesteps']:,}")

    start_time = time.time()
    trainer.train()
    wall_time = time.time() - start_time

    print(f"[Train] Wall time: {wall_time:.0f}s ({wall_time/3600:.1f}h)")

    # Cleanup
    logger.close()
    env.close()
    print("[Train] Done.")


if __name__ == "__main__":
    main()
