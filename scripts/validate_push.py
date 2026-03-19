#!/usr/bin/env python3
"""Validate that push disturbances actually fire in the live simulator.

This script:
1. Creates a G1 environment with push_vel_range=[0.5, 2.0]
2. Runs ~100 steps (several episodes)
3. Checks push_statistics to confirm pushes fired
4. Reports pass/fail

Usage:
    python scripts/validate_push.py
"""
import sys
import os

# Must init sim before any isaaclab imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.envs.g1_env import init_sim, make_env
from src.utils.config import load_config

import torch


def main():
    print("=" * 60)
    print("PUSH VALIDATION TEST")
    print("=" * 60)

    # Load config with pushes enabled (randomized task)
    cfg = load_config(
        base_path="configs/base.yaml",
        task_path="configs/task/randomized.yaml",
        model_path="configs/model/mlp.yaml",
        train_path="configs/train/default.yaml",
    )

    # Override to ensure pushes are enabled
    cfg["task"]["domain_randomization"]["push_vel_range"] = [0.5, 2.0]
    cfg["task"]["domain_randomization"]["push_steps"] = [5, 10, 15]
    cfg["task"]["num_envs"] = 16  # very small for quick test

    print(f"\nConfig:")
    print(f"  push_vel_range: {cfg['task']['domain_randomization']['push_vel_range']}")
    print(f"  push_steps: {cfg['task']['domain_randomization'].get('push_steps', 'default')}")
    print(f"  num_envs: {cfg['task']['num_envs']}")

    # Create environment
    print("\n[INFO] Creating environment...")
    import traceback
    try:
        env = make_env(cfg, device="cuda", headless=True)
    except Exception as e:
        print(f"[FAIL] make_env crashed: {e}")
        traceback.print_exc()
        return 1
    print(f"[OK] Environment created: obs={env.obs_dim}, act={env.act_dim}", flush=True)
    print(f"[OK] Push steps: {env._push_steps}", flush=True)
    print(f"[OK] _env is None: {env._env is None}", flush=True)

    # Reset
    print("[INFO] Calling env.reset()...", flush=True)
    try:
        obs = env.reset()
        print(f"[OK] Reset complete, obs shape: {obs['obs'].shape}", flush=True)
    except Exception as e:
        print(f"[FAIL] reset() crashed: {e}", flush=True)
        traceback.print_exc()
        return 1

    # Run for 50 steps (should trigger pushes at steps 5, 10, 15)
    n_steps = 50
    total_push_events = 0
    steps_with_pushes = []

    print(f"[INFO] Running {n_steps} steps...", flush=True)
    for step_i in range(1, n_steps + 1):
        try:
            actions = torch.zeros(env.num_envs, env.act_dim, device=env.device)
            if step_i <= 3 or step_i in [5, 10, 15]:
                print(f"  [STEP] About to call env.step({step_i})...", flush=True)
            result = env.step(actions)
            if step_i <= 3 or step_i in [5, 10, 15]:
                print(f"  [STEP] step({step_i}) completed", flush=True)
        except Exception as e:
            print(f"[FAIL] step {step_i} crashed: {e}", flush=True)
            traceback.print_exc()
            return 1

        push_applied = result.get("push_applied", None)
        if push_applied is not None and push_applied.any():
            n_pushed = int(push_applied.sum().item())
            total_push_events += n_pushed
            steps_with_pushes.append(step_i)
            print(f"  Step {step_i}: {n_pushed}/{env.num_envs} envs pushed")

    # Check statistics
    stats = env.get_push_statistics()
    print(f"\n{'=' * 60}")
    print("PUSH STATISTICS:")
    print(f"  Total pushes: {stats['total_pushes']}")
    print(f"  Total episodes completed: {stats['total_episodes']}")
    print(f"  Avg pushes/episode: {stats['avg_pushes_per_episode']}")
    print(f"  % episodes with push: {stats['pct_episodes_with_push']}")
    if stats.get('avg_push_magnitude'):
        print(f"  Avg push magnitude: {stats['avg_push_magnitude']}")
    print(f"  Push steps config: {stats['push_steps_config']}")
    print(f"\n  Steps where pushes occurred: {steps_with_pushes}")

    # PASS/FAIL
    print(f"\n{'=' * 60}")
    if stats['total_pushes'] > 0 and stats.get('avg_pushes_per_episode', 0) >= 2.0:
        print("RESULT: ✓ PASS — Pushes are firing correctly!")
        print(f"  {stats['total_pushes']} pushes across {stats['total_episodes']} episodes")
        print(f"  Average {stats['avg_pushes_per_episode']} pushes/episode")
        env.close()
        return 0
    elif stats['total_pushes'] > 0:
        print(f"RESULT: ⚠ PARTIAL — {stats['total_pushes']} pushes fired but avg/episode low")
        print(f"  avg_pushes_per_episode = {stats['avg_pushes_per_episode']}")
        env.close()
        return 1
    else:
        print("RESULT: ✗ FAIL — No pushes fired!")
        env.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
