#!/usr/bin/env python3
"""
Controlled Push-Recovery Protocol.

Applies fixed-magnitude pushes to a walking robot on FLAT terrain and measures
recovery metrics. This creates a discriminative behavioral benchmark where models
differ in non-fall rate and recovery quality (unlike rough terrain where all fall).

Protocol:
  1. Robot walks for 30 steps to reach steady state
  2. At step 30, apply a push of known magnitude in a random direction
  3. Measure: did the robot recover? how quickly? how much did COM deviate?
  4. Recovery = velocity tracking error drops back below threshold within 40 steps

Metrics per push magnitude:
  - non_fall_rate: fraction that do NOT fall within 2 seconds (40 steps at 50Hz)
  - recovery_rate: fraction that restored command tracking within threshold
  - mean_steps_to_recover: among recovered episodes, steps to restore tracking
  - peak_tracking_error: max velocity tracking deviation post-push
  - mean_post_push_reward: reward accumulated in 40 steps after push

Push magnitudes: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0 m/s

Usage:
    python scripts/push_recovery.py --task push --model dynamite --seed 42 \
        --num_episodes 50 --output_dir results/push_recovery/
"""

import argparse
import json
import math
import os
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

PUSH_MAGNITUDES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
WARMUP_STEPS = 30        # steps before push
POST_PUSH_WINDOW = 40    # steps to measure recovery (~0.8 seconds at 50Hz)
RECOVERY_THRESHOLD = 1.5  # tracking error below this = "recovered"
TOTAL_STEPS = WARMUP_STEPS + POST_PUSH_WINDOW + 10  # buffer


def parse_args():
    parser = argparse.ArgumentParser(description="Push Recovery Protocol")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/push_recovery")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def apply_controlled_push(env, push_magnitude):
    """Apply a push of exact magnitude in a random direction to ALL envs.
    
    Unlike the stochastic push system in g1_env, this applies pushes of a
    precise magnitude (not sampled from a range).
    """
    n_envs = env.num_envs
    angle = torch.empty(n_envs, device=env.device).uniform_(0, 2 * math.pi)
    push_vel = torch.zeros(n_envs, 3, device=env.device)
    push_vel[:, 0] = push_magnitude * torch.cos(angle)
    push_vel[:, 1] = push_magnitude * torch.sin(angle)

    if env._env is not None and hasattr(env._rl_env, "scene"):
        try:
            robot = env._rl_env.scene["robot"]
            all_ids = torch.arange(n_envs, device=env.device)
            vel = robot.data.root_vel_w[all_ids].clone()
            vel[:, :3] += push_vel
            robot.write_root_velocity_to_sim(vel, env_ids=all_ids)
        except Exception as e:
            print(f"[Push] ERROR: Could not apply push: {e}")


def evaluate_push_recovery(cfg, model, device, num_episodes, push_magnitude,
                           existing_env=None):
    """Run push-recovery protocol for one push magnitude.
    
    Returns dict with recovery metrics and the environment for reuse.
    """
    from src.utils.history_buffer import HistoryBuffer

    model.eval()
    if existing_env is not None:
        env = existing_env
    else:
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

    # Disable the environment's own push system for this controlled test
    env._push_steps = []

    # Per-episode results
    results = {
        "non_fall": [],  # 1 if survived post-push window, 0 if fell
        "recovered": [],  # 1 if tracking error returned below threshold
        "steps_to_recover": [],  # steps to recovery (NaN if didn't recover)
        "peak_tracking_error": [],
        "post_push_reward": [],
    }

    completed = 0
    while completed < num_episodes:
        # Reset env
        reset_data = env.reset()
        obs = reset_data["obs"]
        cmd = reset_data["cmd"]
        prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)

        if uses_history and history_buf is not None:
            history_buf.reset_all()
            history_buf.insert(obs, prev_action)

        # Track per-env state
        n = env.num_envs
        fell = torch.zeros(n, dtype=torch.bool, device=device)
        recovered = torch.zeros(n, dtype=torch.bool, device=device)
        steps_to_recover = torch.full((n,), float('nan'), device=device)
        peak_track_err = torch.zeros(n, device=device)
        post_push_reward = torch.zeros(n, device=device)
        push_applied = False

        with torch.no_grad():
            for step in range(TOTAL_STEPS):
                model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
                if uses_history and history_buf is not None:
                    o, a, m = history_buf.get()
                    model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})

                output = model(**model_input)
                action = output["action_mean"]

                step_data = env.step(action)
                obs = step_data["obs"]
                cmd = step_data.get("cmd", cmd)
                done = step_data["done"]
                terminated = step_data.get("terminated", None)

                if uses_history and history_buf is not None:
                    history_buf.insert(obs, action)
                    reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                    if len(reset_ids) > 0:
                        history_buf.reset_envs(reset_ids)
                        history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                        history_buf.lengths[reset_ids] = 1

                # Apply push at warmup step
                if step == WARMUP_STEPS and not push_applied:
                    apply_controlled_push(env, push_magnitude)
                    push_applied = True

                # Track recovery metrics after push
                if step > WARMUP_STEPS:
                    steps_since_push = step - WARMUP_STEPS

                    # Compute tracking error
                    if obs.shape[-1] >= 12:
                        base_vel = obs[:, 0:3]
                        cmd_vel = obs[:, 9:12]
                        track_err = torch.norm(base_vel - cmd_vel, dim=-1)
                    else:
                        track_err = torch.zeros(n, device=device)

                    # Update peak
                    peak_track_err = torch.maximum(peak_track_err, track_err)

                    # Check recovery (first time tracking error goes below threshold)
                    just_recovered = (~recovered & ~fell & 
                                     (track_err < RECOVERY_THRESHOLD))
                    steps_to_recover[just_recovered] = float(steps_since_push)
                    recovered |= just_recovered

                    # Accumulate post-push reward
                    if steps_since_push <= POST_PUSH_WINDOW:
                        post_push_reward += step_data["reward"]

                    # Detect falls
                    if terminated is not None and isinstance(terminated, torch.Tensor):
                        newly_fell = terminated & ~fell
                        fell |= newly_fell

                prev_action = action

        # Record results for each env (up to num_episodes remaining)
        for i in range(min(n, num_episodes - completed)):
            results["non_fall"].append(0.0 if fell[i].item() else 1.0)
            results["recovered"].append(1.0 if recovered[i].item() else 0.0)
            results["steps_to_recover"].append(steps_to_recover[i].item())
            results["peak_tracking_error"].append(peak_track_err[i].item())
            results["post_push_reward"].append(post_push_reward[i].item())
        completed += n

    return results, env


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Force flat terrain for push recovery (where models can actually walk)
    cfg["task"]["name"] = "push"
    cfg["task"]["domain_randomization"]["enabled"] = True
    cfg["task"]["domain_randomization"]["push_vel_range"] = [0.0, 0.0]  # disable random pushes
    cfg["task"]["num_envs"] = 32

    init_sim(headless=args.headless)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name from config
    model_name = cfg.get("model", {}).get("name", "unknown")

    print(f"[PushRecovery] Model: {model_name}, Checkpoint: {ckpt_path}")
    print(f"[PushRecovery] Push magnitudes: {PUSH_MAGNITUDES}")
    print(f"[PushRecovery] Episodes per magnitude: {args.num_episodes}")

    all_results = {
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "warmup_steps": WARMUP_STEPS,
        "post_push_window": POST_PUSH_WINDOW,
        "recovery_threshold": RECOVERY_THRESHOLD,
        "timestamp": datetime.now().isoformat(),
        "push_magnitudes": PUSH_MAGNITUDES,
        "results": {},
    }

    push_env = None
    for mag in PUSH_MAGNITUDES:
        print(f"\n  Push magnitude = {mag:.1f} m/s ...")
        per_mag_results, push_env = evaluate_push_recovery(
            cfg, model, device, args.num_episodes, mag, existing_env=push_env
        )

        # Compute summary statistics
        non_fall = np.array(per_mag_results["non_fall"])
        recovered = np.array(per_mag_results["recovered"])
        steps = np.array(per_mag_results["steps_to_recover"])
        peak_err = np.array(per_mag_results["peak_tracking_error"])
        post_rew = np.array(per_mag_results["post_push_reward"])

        valid_steps = steps[~np.isnan(steps)]

        summary = {
            "non_fall_rate": float(np.mean(non_fall)),
            "recovery_rate": float(np.mean(recovered)),
            "mean_steps_to_recover": float(np.mean(valid_steps)) if len(valid_steps) > 0 else float('nan'),
            "median_steps_to_recover": float(np.median(valid_steps)) if len(valid_steps) > 0 else float('nan'),
            "peak_tracking_error_mean": float(np.mean(peak_err)),
            "peak_tracking_error_std": float(np.std(peak_err)),
            "post_push_reward_mean": float(np.mean(post_rew)),
            "post_push_reward_std": float(np.std(post_rew)),
        }
        all_results["results"][str(mag)] = summary

        print(f"    non_fall_rate={summary['non_fall_rate']:.2f}  "
              f"recovery_rate={summary['recovery_rate']:.2f}  "
              f"steps_to_recover={summary['mean_steps_to_recover']:.1f}  "
              f"peak_err={summary['peak_tracking_error_mean']:.2f}")

    # Save
    out_file = output_dir / f"push_recovery_{model_name}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[PushRecovery] Saved to: {out_file}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
