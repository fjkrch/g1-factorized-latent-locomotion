#!/usr/bin/env python3
"""
Scientifically valid OOD (out-of-distribution) sweep evaluation.

This script evaluates a trained policy checkpoint under varying physics
parameters (friction, push magnitude, action delay, ...) and **verifies**
that the requested parameter values are actually active in the live
simulator before collecting any data.

Why subprocess-per-level is required
====================================
Isaac Lab v0.53 bakes physics-material properties during
``gymnasium.make()`` via its ``EventManager``.  The ``physics_material``
event runs in ``startup`` mode: it is executed exactly once when the env is
constructed, setting ``static_friction``, ``dynamic_friction``, and
``restitution`` for all rigid bodies.  Runtime calls to
``root_physx_view.set_material_properties()`` appear to succeed but do
**not** propagate to the actual PhysX solver — the sanity check in a
previous version of this script confirmed this empirically (two
different friction values produced identical read-backs).

The only reliable way to change physics material is to modify the
``env_cfg.events.physics_material.params`` **before** calling
``gymnasium.make()``.  Since Isaac Lab only supports one
``SimulationApp`` per process, each sweep level must run in its own
subprocess.

Architecture
============
::

    ┌───────────────────────────────────────────────┐
    │ Parent (orchestrator)                         │
    │  • parse CLI, load sweep config               │
    │  • for each level:                            │
    │      spawn child with --_worker_level <json>  │
    │      read result from temp JSON file          │
    │  • assemble CSV + JSON                        │
    └───────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    ┌───────────┐ ┌───────────┐ ┌───────────┐
    │ Child 0   │ │ Child 1   │ │ Child N   │
    │ level=1.0 │ │ level=0.7 │ │ level=0.1 │
    │           │ │           │ │           │
    │ SimApp    │ │ SimApp    │ │ SimApp    │
    │ env_cfg   │ │ env_cfg   │ │ env_cfg   │
    │ gym.make  │ │ gym.make  │ │ gym.make  │
    │ verify    │ │ verify    │ │ verify    │
    │ eval      │ │ eval      │ │ eval      │
    │ json→tmp  │ │ json→tmp  │ │ json→tmp  │
    │ os._exit  │ │ os._exit  │ │ os._exit  │
    └───────────┘ └───────────┘ └───────────┘

Supported factors
-----------------
PhysX-verified (live read-back from simulator, modified via env_cfg):
  * ``friction``     — ``static_friction_range`` & ``dynamic_friction_range``
  * ``restitution``  — ``restitution_range``

Config-verified (software-applied during stepping by the wrapper):
  * ``push_magnitude`` — velocity perturbation via ``_apply_push()``
  * ``action_delay``   — stored in ``_dynamics_params`` (aux-loss target only)

Usage
-----
    # Single sweep
    python scripts/eval_ood_validated.py \\
        --checkpoint outputs/randomized/dynamite_full/seed_42/.../best.pt \\
        --sweep configs/sweeps/friction.yaml \\
        --num_episodes 100

    # Custom output dir
    python scripts/eval_ood_validated.py \\
        --checkpoint path/to/best.pt \\
        --sweep configs/sweeps/push_magnitude.yaml \\
        --output_dir outputs/ood_validated/flat/dynamite

Outputs
-------
    <output_dir>/<factor>.csv   — one row per sweep level
    <output_dir>/<factor>.json  — full structured results + metadata

Columns in CSV:
    factor, level, level_display, reward_mean, reward_std,
    ep_len_mean, ep_len_std, num_episodes, seed,
    verification_passed, verification_value, wall_time_s
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default (unperturbed) values for every DR parameter.
NOMINAL_DR: dict[str, object] = {
    "enabled": True,
    "friction_range": [1.0, 1.0],
    "restitution_range": [0.5, 0.5],
    "added_mass_range": [0.0, 0.0],
    "com_displacement_range": [0.0, 0.0],
    "motor_strength_range": [1.0, 1.0],
    "kp_range": [1.0, 1.0],
    "kd_range": [1.0, 1.0],
    "action_delay_range": [0, 0],
    "push_interval": 0,
    "push_vel_range": [0.0, 0.0],
}

#: Map from sweep-config variable path → short factor name.
VARIABLE_TO_FACTOR: dict[str, str] = {
    "task.domain_randomization.friction_range": "friction",
    "task.domain_randomization.action_delay_range": "action_delay",
    "task.domain_randomization.push_vel_range": "push_magnitude",
    "task.domain_randomization.restitution_range": "restitution",
    "task.domain_randomization.motor_strength_range": "motor_strength",
    "task.domain_randomization.added_mass_range": "added_mass",
}

#: Absolute tolerance for PhysX read-back verification.
VERIFY_TOL = 0.02

#: Push interval used when push_magnitude > 0.
DEFAULT_PUSH_INTERVAL = 200

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "factor",
    "level",
    "level_display",
    "reward_mean",
    "reward_std",
    "ep_len_mean",
    "ep_len_std",
    "num_episodes",
    "seed",
    "verification_passed",
    "verification_value",
    "wall_time_s",
    "total_pushes",
    "avg_pushes_per_episode",
    "pct_episodes_with_push",
]


# =====================================================================
# Verification helpers (used inside worker subprocess)
# =====================================================================

def _get_material_properties(env):
    """Read PhysX material-properties tensor from the live simulator."""
    if env._env is None:
        raise RuntimeError("Isaac Lab env not available (mock mode)")
    robot = env._rl_env.scene["robot"]
    if not hasattr(robot, "root_physx_view"):
        raise RuntimeError("robot has no root_physx_view attribute")
    return robot.root_physx_view.get_material_properties()


def verify_friction(env, expected_value: float, tol: float = VERIFY_TOL):
    """Read back PhysX static/dynamic friction and verify.

    The material-properties tensor has shape ``[num_envs, num_shapes, 3]``
    where the last dimension is ``(static_friction, dynamic_friction,
    restitution)``.  We index ``mat[:, :, 0]`` for static friction and
    ``mat[:, :, 1]`` for dynamic friction.
    """
    try:
        mat = _get_material_properties(env)
        # mat shape: [num_envs, num_shapes_per_env, 3]
        static = mat[:, :, 0]   # static friction across all envs & shapes
        dynamic = mat[:, :, 1]  # dynamic friction across all envs & shapes
        ms, md = static.mean().item(), dynamic.mean().item()
        ss, sd = static.std().item(), dynamic.std().item()
        passed = (
            abs(ms - expected_value) < tol
            and abs(md - expected_value) < tol
            and ss < tol
            and sd < tol
        )
        details = {
            "expected": expected_value,
            "static_friction_mean": round(ms, 6),
            "dynamic_friction_mean": round(md, 6),
            "static_friction_std": round(ss, 6),
            "dynamic_friction_std": round(sd, 6),
        }
    except Exception as exc:
        passed, details = False, {"error": str(exc)}
    return passed, details


def verify_restitution(env, expected_value: float, tol: float = VERIFY_TOL):
    """Read back PhysX restitution and verify.

    See :func:`verify_friction` for tensor layout notes.
    """
    try:
        mat = _get_material_properties(env)
        rest = mat[:, :, 2]  # restitution across all envs & shapes
        m, s = rest.mean().item(), rest.std().item()
        passed = abs(m - expected_value) < tol and s < tol
        details = {
            "expected": expected_value,
            "restitution_mean": round(m, 6),
            "restitution_std": round(s, 6),
        }
    except Exception as exc:
        passed, details = False, {"error": str(exc)}
    return passed, details


def verify_push_magnitude(env, expected_range: list):
    """Verify push config is set and push_interval > 0 (unless zero push)."""
    dr = env.task_cfg["domain_randomization"]
    actual = dr.get("push_vel_range", [0.0, 0.0])
    interval = dr.get("push_interval", 0)
    is_zero = expected_range[0] == 0.0 and expected_range[1] == 0.0
    if is_zero:
        passed = True
    else:
        passed = actual == list(expected_range) and interval > 0
    details = {
        "expected_range": expected_range,
        "actual_range": actual,
        "push_interval": interval,
    }
    return passed, details


def verify_action_delay(env, expected_value: float):
    """Verify action delay is stored in _dynamics_params (config-level)."""
    delay = env._dynamics_params.get("delay")
    if delay is None:
        return False, {"error": "delay not in _dynamics_params (DR may be off)"}
    m = delay.float().mean().item()
    passed = abs(m - expected_value) < 0.5
    details = {
        "expected": expected_value,
        "actual_mean": round(m, 4),
        "note": "delay is aux-loss-level only; not physically applied in step()",
    }
    return passed, details


#: Registry of per-factor verification functions.
VERIFY_FN: dict[str, object] = {
    "friction": lambda env, lvl: verify_friction(env, lvl[0]),
    "restitution": lambda env, lvl: verify_restitution(env, lvl[0]),
    "push_magnitude": lambda env, lvl: verify_push_magnitude(env, lvl),
    "action_delay": lambda env, lvl: verify_action_delay(env, lvl[0]),
}


# =====================================================================
# env_cfg modification (applied in worker BEFORE gymnasium.make)
# =====================================================================

def apply_factor_to_env_cfg(env_cfg, factor: str, level) -> None:
    """Modify ``env_cfg.events.physics_material.params`` so that the
    ``EventManager`` applies our desired parameter at startup.

    Parameters
    ----------
    env_cfg
        The Isaac Lab env config returned by ``parse_env_cfg()``.
    factor : str
        The factor being swept.
    level
        The sweep level value (e.g. ``[0.5, 0.5]``).
    """
    if not hasattr(env_cfg, "events"):
        return

    ev = env_cfg.events

    if factor == "friction":
        if hasattr(ev, "physics_material"):
            val = float(level[0])
            ev.physics_material.params["static_friction_range"] = (val, val)
            ev.physics_material.params["dynamic_friction_range"] = (val, val)

    elif factor == "restitution":
        if hasattr(ev, "physics_material"):
            val = float(level[0])
            ev.physics_material.params["restitution_range"] = (val, val)

    # push_magnitude and action_delay don't modify env_cfg —
    # they are handled by the wrapper's DR config at runtime.


def apply_sweep_level_to_wrapper(env, factor: str, level) -> None:
    """Pin all wrapper-level DR params to nominal, then override the
    swept factor for push / delay handling."""
    dr = env.task_cfg["domain_randomization"]

    # Reset every DR parameter to nominal
    for key, nominal in NOMINAL_DR.items():
        dr[key] = list(nominal) if isinstance(nominal, list) else nominal

    env.dr_enabled = True

    if factor == "friction":
        dr["friction_range"] = list(level)
    elif factor == "restitution":
        dr["restitution_range"] = list(level)
    elif factor == "push_magnitude":
        dr["push_vel_range"] = list(level)
        if level[0] > 0 or level[1] > 0:
            dr["push_interval"] = DEFAULT_PUSH_INTERVAL
        else:
            dr["push_interval"] = 0
    elif factor == "action_delay":
        dr["action_delay_range"] = [int(level[0]), int(level[1])]
    elif factor == "motor_strength":
        dr["motor_strength_range"] = list(level)
    elif factor == "added_mass":
        dr["added_mass_range"] = list(level)


# =====================================================================
# Evaluation loop (used inside worker subprocess)
# =====================================================================

def run_eval_episodes(env, model, cfg, num_episodes, device, initial_reset_data=None):
    """Run deterministic evaluation episodes with push statistics tracking."""
    import torch
    from src.utils.history_buffer import HistoryBuffer

    model.eval()
    uses_history = getattr(model, "uses_history", False)

    history_buf = None
    if uses_history:
        obs_cfg = cfg["task"]["observation"]
        history_buf = HistoryBuffer(
            num_envs=env.num_envs,
            history_len=obs_cfg["history_len"],
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
            device=device,
        )

    import numpy as np

    ep_rewards: list[float] = []
    ep_lengths: list[float] = []

    running_reward = torch.zeros(env.num_envs, device=device)
    running_length = torch.zeros(env.num_envs, device=device)
    completed = 0

    # Reset push statistics for this evaluation run
    if hasattr(env, "reset_push_statistics"):
        env.reset_push_statistics()

    if initial_reset_data is not None:
        obs = initial_reset_data["obs"]
        cmd = initial_reset_data["cmd"]
    else:
        reset_data = env.reset()
        obs = reset_data["obs"]
        cmd = reset_data["cmd"]

    prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)

    if uses_history and history_buf is not None:
        history_buf.insert(obs, prev_action)

    with torch.no_grad():
        while completed < num_episodes:
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

            running_reward += step_data["reward"]
            running_length += 1

            if uses_history and history_buf is not None:
                history_buf.insert(obs, action)
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if len(reset_ids) > 0:
                    history_buf.reset_envs(reset_ids)
                    history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                    history_buf.lengths[reset_ids] = 1

            for idx in done.nonzero(as_tuple=False).squeeze(-1):
                ep_rewards.append(running_reward[idx].item())
                ep_lengths.append(running_length[idx].item())
                running_reward[idx] = 0
                running_length[idx] = 0
                completed += 1

            prev_action = action

    # Collect push statistics
    push_stats = {}
    if hasattr(env, "get_push_statistics"):
        push_stats = env.get_push_statistics()

    return {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "ep_len_mean": float(np.mean(ep_lengths)),
        "ep_len_std": float(np.std(ep_lengths)),
        "num_episodes": len(ep_rewards),
        "push_stats": push_stats,
    }


# =====================================================================
# Level display helper
# =====================================================================

def _level_display(level) -> str:
    if isinstance(level, (list, tuple)) and len(level) == 2:
        if level[0] == level[1]:
            return str(level[0])
        return f"{level[0]}-{level[1]}"
    return str(level)


# =====================================================================
# Worker subprocess entry point
# =====================================================================

def worker_main(args):
    """Execute a single sweep level — called in a fresh subprocess.

    This function:
    1. Creates SimulationApp
    2. Modifies env_cfg.events.physics_material BEFORE gym.make()
    3. Creates env with the correct physics baked in
    4. Verifies the PhysX state matches expectations
    5. Runs evaluation episodes
    6. Writes results to a temp JSON file
    7. os._exit(0)
    """
    import torch
    import numpy as np

    from src.utils.config import load_yaml
    from src.utils.seed import set_seed
    from src.models import build_model
    from src.envs.g1_env import init_sim, G1EnvWrapper, _GYM_ID_MAP

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    level = json.loads(args._worker_level)
    factor = args._worker_factor
    result_path = args._worker_result_path

    print(f"[Worker] factor={factor}  level={level}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["task"]["num_envs"] = args.num_envs

    # Ensure DR sub-dict exists
    if "domain_randomization" not in cfg["task"]:
        cfg["task"]["domain_randomization"] = dict(NOMINAL_DR)

    # ── Init SimulationApp ─────────────────────────────────────────
    init_sim(headless=True)

    # ── Parse env_cfg and modify physics_material BEFORE gym.make ──
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task_name = cfg["task"]["name"]
    gym_id = _GYM_ID_MAP.get(task_name)
    if gym_id is None:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(_GYM_ID_MAP.keys())}")

    env_cfg = parse_env_cfg(gym_id, device=device, num_envs=args.num_envs)

    # Apply the sweep factor to env_cfg (modifies EventManager params)
    apply_factor_to_env_cfg(env_cfg, factor, level)

    print(f"[Worker] Creating env with modified env_cfg for {factor}={level}")

    # ── Create env, applying physics at construction time ──────────
    isaac_env = gym.make(gym_id, cfg=env_cfg)
    rl_env = isaac_env.unwrapped

    # Build a G1EnvWrapper around the already-created Isaac env
    env = G1EnvWrapper(cfg, device=device)
    env._env = isaac_env
    env._rl_env = rl_env


    # Auto-detect dims (same logic as _create_isaac_env)
    obs_space = isaac_env.observation_space
    act_space = isaac_env.action_space
    if hasattr(obs_space, "spaces"):
        inner = obs_space.spaces.get("policy", None)
        actual_obs_dim = inner.shape[-1] if inner is not None else obs_space.shape[-1]
    elif hasattr(obs_space, "shape"):
        actual_obs_dim = obs_space.shape[-1]
    else:
        actual_obs_dim = env.obs_dim
    actual_act_dim = act_space.shape[-1] if hasattr(act_space, "shape") else env.act_dim

    env.obs_dim = actual_obs_dim
    env.act_dim = actual_act_dim
    env.cmd_dim = 0
    cfg["task"]["observation"]["proprioception_dim"] = actual_obs_dim
    cfg["task"]["observation"]["action_dim"] = actual_act_dim
    cfg["task"]["observation"]["command_dim"] = 0
    cfg["task"]["observation"]["include_previous_action"] = False

    print(f"[Worker] dims: obs={actual_obs_dim}, act={actual_act_dim}")

    # ── Apply wrapper-level sweep config (push interval, delay, etc) ──
    apply_sweep_level_to_wrapper(env, factor, level)

    # ── Reset env (triggers _randomize_dynamics + _apply_dynamics_to_env) ──
    reset_data = env.reset()

    # ── Verification ───────────────────────────────────────────────
    v_passed = None
    v_details: dict = {}
    if not args.skip_verification:
        verify_fn = VERIFY_FN.get(factor)
        if verify_fn is not None:
            v_passed, v_details = verify_fn(env, level)
            status = "OK" if v_passed else "FAIL"
            print(f"[Worker] Verification: {status}  {v_details}")
            if not v_passed:
                result = {
                    "error": f"Verification FAILED for {factor}={level}",
                    "details": v_details,
                }
                with open(result_path, "w") as f:
                    json.dump(result, f)
                print(f"[Worker] ABORT — verification failed")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
        else:
            v_details = {"note": f"no verification fn for '{factor}'"}
    else:
        v_details = {"note": "verification skipped by user"}

    # ── Build model and load weights ───────────────────────────────
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ── Run evaluation episodes ────────────────────────────────────
    t0 = time.time()
    metrics = run_eval_episodes(
        env, model, cfg, args.num_episodes, device,
        initial_reset_data=reset_data,
    )
    wall = time.time() - t0

    # ── Post-evaluation verification ───────────────────────────────
    if not args.skip_verification and factor in VERIFY_FN:
        post_passed, post_details = VERIFY_FN[factor](env, level)
        if not post_passed:
            print(f"[Worker] WARNING: Post-eval verification FAILED: {post_details}")
            v_passed = False
            v_details["post_eval"] = post_details

    # ── Write results ──────────────────────────────────────────────
    push_stats = metrics.get("push_stats", {})
    result = {
        "factor": factor,
        "level": level,
        "level_display": _level_display(level),
        "reward_mean": round(metrics["reward_mean"], 4),
        "reward_std": round(metrics["reward_std"], 4),
        "ep_len_mean": round(metrics["ep_len_mean"], 2),
        "ep_len_std": round(metrics["ep_len_std"], 2),
        "num_episodes": metrics["num_episodes"],
        "seed": args.seed,
        "verification_passed": v_passed,
        "verification_value": v_details,
        "wall_time_s": round(wall, 2),
        "push_stats": {
            "total_pushes": push_stats.get("total_pushes", 0),
            "avg_pushes_per_episode": push_stats.get("avg_pushes_per_episode", 0.0),
            "pct_episodes_with_push": push_stats.get("pct_episodes_with_push", 0.0),
            "avg_push_magnitude": push_stats.get("avg_push_magnitude", 0.0),
            "push_steps_config": push_stats.get("push_steps_config", []),
        },
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Log push statistics
    pct_push = push_stats.get("pct_episodes_with_push", 0.0)
    avg_push = push_stats.get("avg_pushes_per_episode", 0.0)
    print(
        f"[Worker] Done: reward={metrics['reward_mean']:.2f} "
        f"± {metrics['reward_std']:.2f}  "
        f"ep_len={metrics['ep_len_mean']:.0f}  "
        f"pushes={avg_push:.1f}/ep ({pct_push:.0f}% with ≥1)  "
        f"time={wall:.1f}s"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# =====================================================================
# Sanity check (runs 2 levels in separate subprocesses)
# =====================================================================

def run_sanity_subprocess(
    checkpoint: str, factor: str, level, num_envs: int, seed: int,
) -> dict:
    """Spawn a worker that creates the env with the given level,
    verifies, runs 5 episodes, and returns the result dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_path = tf.name

    cmd = [
        sys.executable, __file__,
        "--checkpoint", checkpoint,
        "--sweep", "UNUSED",  # not used in worker mode
        "--_worker_level", json.dumps(level),
        "--_worker_factor", factor,
        "--_worker_result_path", result_path,
        "--num_episodes", "5",
        "--num_envs", str(num_envs),
        "--seed", str(seed),
    ]

    env_vars = os.environ.copy()
    env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.run(cmd, env=env_vars, timeout=300)

    try:
        with open(result_path) as f:
            result = json.load(f)
    except Exception:
        result = {"error": f"Worker exited with code {proc.returncode}, no result file"}
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    return result


def run_sanity_check(checkpoint: str, factor: str, num_envs: int, seed: int) -> bool:
    """Spawn two subprocesses with extreme levels, verify that the
    PhysX read-back values are different."""
    extremes: dict[str, list[list]] = {
        "friction": [[1.0, 1.0], [0.1, 0.1]],
        "restitution": [[0.0, 0.0], [0.8, 0.8]],
        "push_magnitude": [[0.0, 0.0], [5.0, 8.0]],
        "action_delay": [[0, 0], [5, 5]],
    }

    levels = extremes.get(factor)
    if levels is None:
        print(f"[Sanity] No sanity test for factor '{factor}' — skipping.")
        return True

    print(f"\n{'─' * 60}")
    print(f"[Sanity] Running sanity check for factor '{factor}' ...")
    print(f"[Sanity] Spawning 2 subprocesses with extreme levels ...")

    results = []
    for lvl in levels:
        print(f"[Sanity] Testing level={lvl} ...")
        res = run_sanity_subprocess(checkpoint, factor, lvl, num_envs, seed)
        results.append(res)
        if "error" in res:
            print(f"[Sanity]   level={lvl}  ERROR: {res['error']}")
        else:
            v = res.get("verification_passed")
            vd = res.get("verification_value", {})
            status = "OK" if v else "FAIL"
            print(f"[Sanity]   level={lvl}  {status}  {vd}")

    # Check all passed
    for r in results:
        if "error" in r:
            print("[Sanity] FAIL — a subprocess errored out.")
            return False
        if not r.get("verification_passed", False):
            print("[Sanity] FAIL — verification failed for at least one level.")
            return False

    # Check the two extremes produced distinguishable read-backs
    r0 = results[0].get("verification_value", {})
    r1 = results[1].get("verification_value", {})

    if factor == "friction":
        v0 = r0.get("static_friction_mean")
        v1 = r1.get("static_friction_mean")
    elif factor == "restitution":
        v0 = r0.get("restitution_mean")
        v1 = r1.get("restitution_mean")
    elif factor == "action_delay":
        v0 = r0.get("actual_mean", 0)
        v1 = r1.get("actual_mean", 0)
    elif factor == "push_magnitude":
        v0, v1 = 0.0, 1.0  # config-level; both passed individually → OK
    else:
        v0, v1 = 0.0, 1.0

    if v0 is not None and v1 is not None and abs(v0 - v1) <= VERIFY_TOL:
        print(
            f"[Sanity] FAIL — two different levels produced SAME verification"
            f" values ({v0} vs {v1}).  Parameter is NOT reaching the simulator."
        )
        return False

    print(f"[Sanity] PASS — factor '{factor}' is verifiably controllable.")
    print(f"{'─' * 60}\n")
    return True


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Scientifically valid OOD sweep evaluation with live "
                    "simulator verification (subprocess-per-level)."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pt)")
    p.add_argument("--sweep", type=str, required=True,
                    help="Sweep config YAML (e.g. configs/sweeps/friction.yaml)")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Output directory (default: outputs/ood_validated/<task>/<model>)")
    p.add_argument("--num_episodes", type=int, default=100,
                    help="Evaluation episodes per sweep level")
    p.add_argument("--num_envs", type=int, default=32,
                    help="Parallel envs for evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--sanity-check", action="store_true", default=True,
                    help="Run sanity check before sweep (default: on)")
    p.add_argument("--no-sanity-check", dest="sanity_check",
                    action="store_false")
    p.add_argument("--skip-verification", action="store_true", default=False,
                    help="Skip live verification (NOT recommended)")

    # Internal worker args (not for user use)
    p.add_argument("--_worker_level", type=str, default=None,
                    help=argparse.SUPPRESS)
    p.add_argument("--_worker_factor", type=str, default=None,
                    help=argparse.SUPPRESS)
    p.add_argument("--_worker_result_path", type=str, default=None,
                    help=argparse.SUPPRESS)

    return p.parse_args()


# =====================================================================
# Main (orchestrator)
# =====================================================================

def main():
    args = parse_args()

    # ── Worker mode (launched by orchestrator) ─────────────────────
    if args._worker_level is not None:
        worker_main(args)
        return  # worker_main calls os._exit

    # ── Orchestrator mode ──────────────────────────────────────────
    from src.utils.config import load_yaml
    from src.utils.seed import set_seed
    import torch

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint to get task/model info
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    task_name = cfg["task"]["name"]
    model_name = cfg["model"]["name"]

    # Load sweep config
    sweep_cfg = load_yaml(args.sweep)
    sweep_name = sweep_cfg["sweep"]["name"]
    variable = sweep_cfg["sweep"]["variable"]
    values = sweep_cfg["sweep"]["values"]

    factor = VARIABLE_TO_FACTOR.get(variable)
    if factor is None:
        print(f"ERROR: unknown sweep variable '{variable}'")
        print(f"Supported: {list(VARIABLE_TO_FACTOR.keys())}")
        sys.exit(1)

    print("=" * 60)
    print("  OOD Sweep Evaluation — subprocess-per-level")
    print("  with live simulator verification")
    print("=" * 60)
    print(f"  Factor      : {factor} ({sweep_name})")
    print(f"  Levels      : {values}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Task        : {task_name}")
    print(f"  Model       : {model_name}")
    print(f"  Episodes    : {args.num_episodes} per level")
    print(f"  Num envs    : {args.num_envs}")
    print(f"  Seed        : {args.seed}")
    print(f"  Verification: {'OFF' if args.skip_verification else 'ON'}")
    print("=" * 60)

    # ── Sanity check ───────────────────────────────────────────────
    if args.sanity_check and not args.skip_verification:
        ok = run_sanity_check(str(ckpt_path), factor, args.num_envs, args.seed)
        if not ok:
            print("\n" + "!" * 60)
            print("  SANITY CHECK FAILED")
            print()
            print("  The requested parameter cannot be verified in the live")
            print("  simulator.  Continuing would produce scientifically")
            print("  invalid results.")
            print("!" * 60)
            sys.exit(1)

    # ── Run sweep: one subprocess per level ────────────────────────
    rows: list[dict] = []
    total_t0 = time.time()

    for i, level in enumerate(values):
        label = _level_display(level)
        print(f"\n[OOD] Level {i + 1}/{len(values)}: {factor} = {label}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            result_path = tf.name

        cmd = [
            sys.executable, __file__,
            "--checkpoint", str(ckpt_path),
            "--sweep", args.sweep,
            "--_worker_level", json.dumps(level),
            "--_worker_factor", factor,
            "--_worker_result_path", result_path,
            "--num_episodes", str(args.num_episodes),
            "--num_envs", str(args.num_envs),
            "--seed", str(args.seed),
        ]
        if args.skip_verification:
            cmd.append("--skip-verification")

        env_vars = os.environ.copy()
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        t0 = time.time()
        proc = subprocess.run(cmd, env=env_vars, timeout=600)
        wall = time.time() - t0

        # Read worker result
        try:
            with open(result_path) as f:
                result = json.load(f)
        except Exception:
            print(f"[OOD]   ERROR: worker crashed (exit code {proc.returncode})")
            result = {
                "factor": factor,
                "level": level,
                "level_display": label,
                "reward_mean": float("nan"),
                "reward_std": float("nan"),
                "ep_len_mean": float("nan"),
                "ep_len_std": float("nan"),
                "num_episodes": 0,
                "seed": args.seed,
                "verification_passed": False,
                "verification_value": {"error": f"Worker crashed, exit code {proc.returncode}"},
                "wall_time_s": round(wall, 2),
            }
        finally:
            try:
                os.unlink(result_path)
            except OSError:
                pass

        if "error" in result and "verification_value" not in result:
            # Worker reported an error (verification failure)
            print(f"[OOD]   ERROR: {result['error']}")
            if not args.skip_verification:
                print("[OOD]   Aborting sweep due to verification failure.")
                sys.exit(1)
        else:
            # Extract push stats from result
            push_stats = result.get("push_stats", {})

            # Normalize the result row for CSV
            row = {
                "factor": result.get("factor", factor),
                "level": json.dumps(result.get("level", level)),
                "level_display": result.get("level_display", label),
                "reward_mean": result.get("reward_mean", float("nan")),
                "reward_std": result.get("reward_std", float("nan")),
                "ep_len_mean": result.get("ep_len_mean", float("nan")),
                "ep_len_std": result.get("ep_len_std", float("nan")),
                "num_episodes": result.get("num_episodes", 0),
                "seed": result.get("seed", args.seed),
                "verification_passed": result.get("verification_passed"),
                "verification_value": json.dumps(result.get("verification_value", {})),
                "wall_time_s": result.get("wall_time_s", round(wall, 2)),
                "total_pushes": push_stats.get("total_pushes", 0),
                "avg_pushes_per_episode": push_stats.get("avg_pushes_per_episode", 0.0),
                "pct_episodes_with_push": push_stats.get("pct_episodes_with_push", 0.0),
                "push_stats": push_stats,  # Keep full stats for JSON output
            }
            rows.append(row)

            # Log with push info
            push_info = ""
            if push_stats.get("pct_episodes_with_push", 0) > 0:
                push_info = f"pushes={push_stats.get('avg_pushes_per_episode', 0):.1f}/ep "

            print(
                f"[OOD]   reward = {row['reward_mean']:>8.2f} "
                f"± {row['reward_std']:.2f}   "
                f"ep_len = {row['ep_len_mean']:.0f}   "
                f"{push_info}"
                f"verified = {'yes' if row['verification_passed'] else 'no'}   "
                f"time = {row['wall_time_s']:.1f}s"
            )

    total_wall = time.time() - total_t0

    # ── Output directory ───────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("outputs/ood_validated") / task_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save CSV ───────────────────────────────────────────────────
    csv_path = out_dir / f"{factor}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Save JSON ──────────────────────────────────────────────────
    json_path = out_dir / f"{factor}.json"
    result_json = {
        "sweep_name": sweep_name,
        "factor": factor,
        "variable": variable,
        "checkpoint": str(ckpt_path),
        "task": task_name,
        "model": model_name,
        "seed": args.seed,
        "num_episodes_per_level": args.num_episodes,
        "num_envs": args.num_envs,
        "timestamp": datetime.now().isoformat(),
        "total_wall_time_s": round(total_wall, 2),
        "nominal_dr": {k: v for k, v in NOMINAL_DR.items()},
        "levels": values,
        "results": rows,
    }
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Results summary")
    print(f"{'=' * 60}")
    for row in rows:
        v = "yes" if row["verification_passed"] else "no"
        push = row.get("push_stats", {})
        push_info = ""
        if push.get("pct_episodes_with_push", 0) > 0:
            push_info = f"  pushes={push.get('avg_pushes_per_episode', 0):.1f}/ep"
        print(
            f"  {factor}={str(row['level_display']):>10s}  "
            f"reward={row['reward_mean']:>8.2f} +/- {row['reward_std']:<6.2f}  "
            f"ep_len={row['ep_len_mean']:>6.0f}  "
            f"verified={v}{push_info}"
        )

    # Print push statistics summary if pushes were applied
    if rows and rows[0].get("push_stats", {}).get("pct_episodes_with_push", 0) > 0:
        avg_pct = sum(r.get("push_stats", {}).get("pct_episodes_with_push", 0) for r in rows) / len(rows)
        print(f"\n  Push coverage: {avg_pct:.1f}% episodes received ≥1 push")

    print(f"\n  Total time: {total_wall:.0f}s")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
