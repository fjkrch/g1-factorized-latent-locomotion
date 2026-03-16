#!/usr/bin/env python3
"""
Evaluation entrypoint.

Usage:
    # Evaluate a single checkpoint
    python scripts/eval.py --checkpoint outputs/flat/dynamite_full/seed_42/.../checkpoints/best.pt

    # Evaluate with specific task config (e.g., eval on push when trained on flat)
    python scripts/eval.py --checkpoint path/to/best.pt --task configs/task/push.yaml

    # Batch evaluate all checkpoints in a directory
    python scripts/eval.py --run_dir outputs/flat/dynamite_full/seed_42/20260316_120000

    # Evaluate with sweep config
    python scripts/eval.py --checkpoint path/to/best.pt --sweep configs/sweeps/push_magnitude.yaml

Inputs:
    - Checkpoint file (.pt)
    - Optional: task config for cross-task evaluation

Outputs (in same run_dir or specified output_dir):
    - eval_metrics.json    : aggregated metrics + metadata
    - eval_episodes.csv    : per-episode data (when --save-episodes is set)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import MetricsTracker
from src.utils.metrics_io import write_eval_episodes, EPISODE_CSV_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(description="DynaMITE Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (uses best.pt)")
    parser.add_argument("--task", type=str, default=None,
                        help="Override task config for cross-task eval")
    parser.add_argument("--eval_config", type=str, default="configs/eval/default.yaml",
                        help="Evaluation config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint dir)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep config for robustness evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run Isaac Sim headless (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Run Isaac Sim with GUI")
    return parser.parse_args()


def evaluate_checkpoint(
    cfg: dict,
    model: torch.nn.Module,
    device: str = "cuda",
    num_episodes: int = 100,
) -> dict:
    """Run evaluation and return metrics."""
    from src.utils.history_buffer import HistoryBuffer

    model.eval()
    env = make_env(cfg, device=device)
    uses_history = getattr(model, 'uses_history', False)

    history_buf = None
    if uses_history:
        task_cfg = cfg["task"]["observation"]
        history_buf = HistoryBuffer(
            num_envs=cfg["task"]["num_envs"],
            history_len=task_cfg["history_len"],
            obs_dim=task_cfg["proprioception_dim"],
            act_dim=task_cfg["action_dim"],
            device=device,
        )

    tracker = MetricsTracker()
    episode_rewards = torch.zeros(env.num_envs, device=device)
    episode_lengths = torch.zeros(env.num_envs, device=device)
    completed = 0

    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data["cmd"]
    prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)

    # Seed history buffer with initial obs so mask isn't all-false
    if uses_history and history_buf is not None:
        history_buf.insert(obs, prev_action)

    with torch.no_grad():
        while completed < num_episodes:
            model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
            if uses_history and history_buf is not None:
                o, a, m = history_buf.get()
                model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})

            output = model(**model_input)
            action = output["action_mean"]  # deterministic

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            episode_rewards += step_data["reward"]
            episode_lengths += 1

            if uses_history and history_buf is not None:
                history_buf.insert(obs, action)
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if len(reset_ids) > 0:
                    history_buf.reset_envs(reset_ids)
                    # Re-seed reset envs so mask isn't all-false
                    history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                    history_buf.lengths[reset_ids] = 1

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids:
                tracker.update({
                    "episode_reward": episode_rewards[idx].item(),
                    "episode_length": episode_lengths[idx].item(),
                })
                episode_rewards[idx] = 0
                episode_lengths[idx] = 0
                completed += 1

            prev_action = action

    # Collect results BEFORE closing (SimulationApp.close() may terminate the process)
    results = tracker.summarize()
    # Don't close env here — caller saves results first, then closes
    return results, env


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.run_dir:
        ckpt_path = Path(args.run_dir) / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(args.run_dir) / "checkpoints" / "latest.pt"
    else:
        print("Error: specify --checkpoint or --run_dir")
        sys.exit(1)

    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load config from checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Override task if specified
    if args.task:
        task_override = load_yaml(args.task)
        from src.utils.config import deep_merge
        cfg = deep_merge(cfg, task_override)

    # Override eval settings
    cfg["task"]["num_envs"] = 32  # smaller for eval
    eval_cfg = load_yaml(args.eval_config)
    cfg = {**cfg, **eval_cfg}

    # Initialise Isaac Sim headless
    init_sim(headless=args.headless)

    # Build model and load weights (checkpoint cfg already has correct dims from training)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        # Robustness sweep evaluation
        sweep_cfg = load_yaml(args.sweep)
        sweep_name = sweep_cfg["sweep"]["name"]
        variable = sweep_cfg["sweep"]["variable"]
        values = sweep_cfg["sweep"]["values"]

        print(f"[Eval] Robustness sweep: {sweep_name} ({len(values)} levels)")
        sweep_results = {
            "sweep_name": sweep_name,
            "parameter": variable,
            "values": values,
            "checkpoint": str(ckpt_path),
            "task": cfg["task"]["name"],
            "model": cfg["model"]["name"],
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "results": [],
        }

        for val in values:
            # Apply sweep variable
            keys = variable.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val

            metrics, env = evaluate_checkpoint(cfg, model, device, args.num_episodes)
            entry = {"value": val, **metrics}
            sweep_results["results"].append(entry)
            print(f"  {variable}={val}: reward={metrics.get('episode_reward/mean', 0):.1f}")

        # Save results BEFORE closing env (SimulationApp.close may exit)
        with open(output_dir / f"sweep_{sweep_name}.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"[Eval] Sweep results saved to: {output_dir / f'sweep_{sweep_name}.json'}")
        if env is not None:
            env.close()

    else:
        # Standard evaluation
        print(f"[Eval] Evaluating: {ckpt_path}")
        print(f"[Eval] Task: {cfg['task']['name']}, Episodes: {args.num_episodes}")

        start_time = time.time()
        metrics, env = evaluate_checkpoint(cfg, model, device, args.num_episodes)
        wall_time = time.time() - start_time

        # Enrich output with metadata
        result = {
            "checkpoint": str(ckpt_path),
            "task": cfg["task"]["name"],
            "model": cfg["model"]["name"],
            "seed": args.seed,
            "num_episodes": args.num_episodes,
            "timestamp": datetime.now().isoformat(),
            "wall_time_s": round(wall_time, 2),
            **metrics,
        }

        # Save results BEFORE closing env (SimulationApp.close may exit)
        with open(output_dir / "eval_metrics.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n[Eval] Results:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}")
        print(f"\n[Eval] Wall time: {wall_time:.1f}s")
        print(f"[Eval] Saved to: {output_dir / 'eval_metrics.json'}")

        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
