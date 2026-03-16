#!/usr/bin/env python3
"""
Training entrypoint for DynaMITE project.

Usage:
    # MLP baseline on flat terrain
    python scripts/train.py --task configs/task/flat.yaml --model configs/model/mlp.yaml

    # DynaMITE on randomized dynamics
    python scripts/train.py --task configs/task/randomized.yaml --model configs/model/dynamite.yaml

    # With CLI overrides
    python scripts/train.py --task configs/task/push.yaml --model configs/model/dynamite.yaml \
        --set seed=123 train.total_timesteps=10000000

    # Resume from checkpoint
    python scripts/train.py --task configs/task/flat.yaml --model configs/model/dynamite.yaml \
        --resume outputs/flat/dynamite/seed_42/20260316_120000

Inputs:
    - Config YAML files (base + task + model + optional train + optional ablation)
    - Optional: checkpoint directory for resuming

Outputs (in run_dir = outputs/{task}/{model}/seed_{seed}/{timestamp}/):
    - config.yaml          : saved merged config
    - manifest.json        : experiment manifest
    - metrics.csv          : training metrics log
    - tb/                  : TensorBoard logs
    - checkpoints/         : model checkpoints
        - ckpt_{step}.pt
        - latest.pt
        - best.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.utils.config import load_config, make_run_dir
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.checkpoint import load_checkpoint, find_latest_checkpoint
from src.models import build_model
from src.envs.g1_env import make_env
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from run directory")
    parser.add_argument("--set", nargs="*", default=[],
                        help="CLI config overrides: key=value")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed")
    return parser.parse_args()


def main():
    args = parse_args()

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

    # Set seed
    set_seed(cfg["seed"], deterministic=False)

    # Create run directory
    if args.resume:
        run_dir = Path(args.resume)
        print(f"[Resume] Resuming from: {run_dir}")
    else:
        run_dir = make_run_dir(cfg)
        print(f"[Train] Run directory: {run_dir}")

    # Logger
    logger = Logger(run_dir)
    logger.log_config(cfg)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] GPU Memory: {torch.cuda.get_device_properties(0).total_mem // 1024**2} MB")

    # Build model
    model = build_model(cfg)
    model.to(device)

    # Build environment
    env = make_env(cfg, device=device)

    # Build trainer
    trainer = PPOTrainer(cfg, model, env, logger, str(run_dir))

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
    print(f"[Train] Total timesteps: {cfg['train']['total_timesteps']:,}")
    trainer.train()

    # Cleanup
    logger.close()
    env.close()
    print("[Train] Done.")


if __name__ == "__main__":
    main()
