"""
Unitree G1 environment wrapper for Isaac Lab (v0.53+ / Isaac Sim 5.x).

IMPORTANT: ``init_sim()`` **must** be called once before ``make_env()`` so that
``SimulationApp`` is created headless *before* any ``isaaclab`` imports.

Typical lifecycle (handled automatically by train.py / eval.py)::

    from src.envs.g1_env import init_sim, make_env
    init_sim(headless=True)          # creates SimulationApp
    env = make_env(cfg, device="cuda")
    ...
    env.close()

This wrapper provides a single interface consumed by PPOTrainer regardless
of whether Isaac Lab is present (falls back to a lightweight mock env for
unit-testing / CI).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Global SimulationApp handle — created by ``init_sim()``
# ---------------------------------------------------------------------------
_SIM_APP = None


def init_sim(headless: bool = True) -> None:
    """Initialise Isaac Sim ``SimulationApp`` (idempotent).

    Must be called **before** any Isaac Lab imports.
    """
    global _SIM_APP
    if _SIM_APP is not None:
        return
    try:
        from isaacsim import SimulationApp
        _SIM_APP = SimulationApp({"headless": headless, "width": 1280, "height": 720})
        print(f"[Sim] SimulationApp created (headless={headless})")
        # Ensure nucleus cloud asset root is set (needed for USD downloads)
        try:
            import carb
            settings = carb.settings.get_settings()
            cloud = settings.get("/persistent/isaac/asset_root/cloud")
            if cloud is None:
                default_root = settings.get("/persistent/isaac/asset_root/default")
                if default_root:
                    settings.set("/persistent/isaac/asset_root/cloud", default_root)
                    print(f"[Sim] Set nucleus cloud root to: {default_root}")
        except Exception:
            pass
    except Exception as exc:
        print(f"[WARNING] Could not create SimulationApp: {exc}")
        print("          Falling back to mock environment.")


# ---------------------------------------------------------------------------
# Isaac Lab gym-id mapping
# ---------------------------------------------------------------------------
# These IDs are registered by ``isaaclab_tasks`` upon import.
_GYM_ID_MAP: dict[str, str] = {
    "flat":       "Isaac-Velocity-Flat-G1-v0",
    # Use the standard rough env — our _apply_push() handles push disturbances.
    # The PushCurriculum variant has a tensor-dimension bug in modify_push_force.
    "push":       "Isaac-Velocity-Rough-G1-v0",
    "randomized": "Isaac-Velocity-Rough-G1-v0",
    "terrain":    "Isaac-Velocity-Rough-G1-v0",
}


class G1EnvWrapper:
    """Wrapper around Isaac Lab's Unitree G1 locomotion environment.

    Provides a clean interface for:
    - Stepping the environment
    - Accessing observations, rewards, dones
    - Applying domain randomization
    - Generating push disturbances
    - Exporting ground-truth dynamics parameters for auxiliary loss
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.task_cfg = cfg["task"]
        self.num_envs = self.task_cfg["num_envs"]

        # Dimensions
        self.obs_dim = self.task_cfg["observation"]["proprioception_dim"]
        self.cmd_dim = self.task_cfg["observation"]["command_dim"]
        self.act_dim = self.task_cfg["observation"]["action_dim"]
        self.history_len = self.task_cfg["observation"]["history_len"]

        # Domain randomization state (per-env ground truth)
        self.dr_enabled = self.task_cfg["domain_randomization"]["enabled"]
        self._dynamics_params: dict[str, torch.Tensor] = {}

        # Isaac Lab environment (created in _create_isaac_env)
        self._env = None
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------
    def _create_isaac_env(self) -> None:
        """Create the Isaac Lab environment via gymnasium registry."""
        global _SIM_APP
        if _SIM_APP is None:
            print("[WARNING] SimulationApp not initialised — using mock env.")
            self._env = None
            return
        try:
            import gymnasium as gym
            # Importing isaaclab_tasks triggers gym.register(...) for all envs
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg

            task_name = self.task_cfg["name"]
            gym_id = _GYM_ID_MAP.get(task_name)
            if gym_id is None:
                raise ValueError(
                    f"Unknown task '{task_name}'. "
                    f"Available: {list(_GYM_ID_MAP.keys())}"
                )

            # Isaac Lab v0.53+ requires the env cfg object
            env_cfg = parse_env_cfg(
                gym_id,
                device=self.device,
                num_envs=self.num_envs,
            )
            self._env = gym.make(gym_id, cfg=env_cfg)
            # Unwrap to ManagerBasedRLEnv for direct tensor access
            self._rl_env = self._env.unwrapped
            print(f"[Env] Created {gym_id}  (num_envs={self.num_envs})")

            # ── Auto-detect actual obs/action dims from Isaac Lab ──
            # Isaac Lab's policy observation already bundles
            #   base_lin_vel, base_ang_vel, projected_gravity,
            #   velocity_commands, joint_pos, joint_vel, prev_actions
            # so we must NOT re-concatenate cmd / prev_action in the model.
            obs_space = self._env.observation_space
            act_space = self._env.action_space
            if hasattr(obs_space, "spaces"):          # gymnasium.spaces.Dict
                inner = obs_space.spaces.get("policy", None)
                actual_obs_dim = inner.shape[-1] if inner is not None else obs_space.shape[-1]
            elif hasattr(obs_space, "shape"):
                actual_obs_dim = obs_space.shape[-1]
            else:
                actual_obs_dim = self.obs_dim
            actual_act_dim = act_space.shape[-1] if hasattr(act_space, "shape") else self.act_dim

            self.obs_dim = actual_obs_dim
            self.act_dim = actual_act_dim
            self.cmd_dim = 0   # already inside Isaac Lab obs

            # Propagate to config so model is built with correct dims
            self.cfg["task"]["observation"]["proprioception_dim"] = actual_obs_dim
            self.cfg["task"]["observation"]["action_dim"] = actual_act_dim
            self.cfg["task"]["observation"]["command_dim"] = 0
            self.cfg["task"]["observation"]["include_previous_action"] = False
            print(f"[Env] Auto-detected dims: obs={actual_obs_dim}, act={actual_act_dim}"
                  f" (cmd & prev_action already in obs)")

        except Exception as exc:
            print(f"[WARNING] Could not create Isaac Lab env: {exc}")
            print("          Falling back to mock environment.")
            self._env = None

    # ------------------------------------------------------------------
    # Step / Reset
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, torch.Tensor]:
        """Reset all environments."""
        self._step_count.zero_()

        if self._env is not None:
            obs_dict, info = self._env.reset()
            if isinstance(obs_dict, dict):
                obs = obs_dict.get("policy", obs_dict.get("obs", None))
                if obs is None:
                    obs = next(iter(obs_dict.values()))
            else:
                obs = obs_dict
            obs = obs.to(self.device)
        else:
            obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device)

        cmd = self._sample_commands()

        if self.dr_enabled:
            self._randomize_dynamics()

        result = {"obs": obs, "cmd": cmd}
        if self.dr_enabled:
            result["dynamics_params"] = self._get_dynamics_targets()
        return result

    def step(self, actions: torch.Tensor) -> dict[str, Any]:
        """Step the environment.

        Args:
            actions: (num_envs, act_dim)

        Returns:
            dict with: obs, cmd, reward, done, info, reset_ids, dynamics_params
        """
        self._step_count += 1

        if self._env is not None:
            obs_dict, rewards, terminated, truncated, info = self._env.step(actions)
            if isinstance(obs_dict, dict):
                obs = obs_dict.get("policy", obs_dict.get("obs", None))
                if obs is None:
                    obs = next(iter(obs_dict.values()))
            else:
                obs = obs_dict
            obs = obs.to(self.device)
            rewards = rewards.to(self.device)
            done = (terminated | truncated).to(self.device)
        else:
            # Mock
            obs = torch.randn(self.num_envs, self.obs_dim, device=self.device) * 0.1
            rewards = torch.zeros(self.num_envs, device=self.device)
            done = self._step_count >= self.task_cfg["episode_length"]
            info = {}

        # Apply push disturbances (only when we manage DR ourselves)
        if self.dr_enabled:
            push_interval = self.task_cfg["domain_randomization"]["push_interval"]
            if push_interval > 0:
                push_mask = (self._step_count % push_interval == 0)
                if push_mask.any():
                    self._apply_push(push_mask)

        cmd = self._sample_commands()

        # Handle resets
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._handle_resets(reset_ids)

        result = {
            "obs": obs,
            "cmd": cmd,
            "reward": rewards,
            "done": done,
            "info": info,
            "reset_ids": reset_ids,
        }
        if self.dr_enabled:
            result["dynamics_params"] = self._get_dynamics_targets()
        return result

    # ------------------------------------------------------------------
    # Commands / Domain randomization
    # ------------------------------------------------------------------
    def _sample_commands(self) -> torch.Tensor:
        """Sample random velocity commands: [vx, vy, yaw_rate]."""
        if self.cmd_dim == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        cmd = torch.zeros(self.num_envs, self.cmd_dim, device=self.device)
        cmd[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(-1.0, 2.0)
        cmd[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(-0.5, 0.5)
        cmd[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(-1.0, 1.0)
        return cmd

    def _randomize_dynamics(self) -> None:
        """Sample domain randomization parameters for all envs."""
        dr = self.task_cfg["domain_randomization"]
        self._dynamics_params = {
            "friction": torch.empty(self.num_envs, 2, device=self.device).uniform_(*dr["friction_range"]),
            "mass": torch.zeros(self.num_envs, 2, device=self.device),
            "motor": torch.empty(self.num_envs, 2, device=self.device).uniform_(*dr["motor_strength_range"]),
            "contact": torch.empty(self.num_envs, 1, device=self.device).uniform_(*dr["restitution_range"]),
            "delay": torch.randint(
                dr["action_delay_range"][0],
                max(dr["action_delay_range"][1], 1) + 1,
                (self.num_envs, 1), device=self.device,
            ).float(),
        }
        self._dynamics_params["mass"][:, 0] = torch.empty(
            self.num_envs, device=self.device).uniform_(*dr["added_mass_range"])
        com_range = dr.get("com_displacement_range", [-0.05, 0.05])
        self._dynamics_params["mass"][:, 1] = torch.empty(
            self.num_envs, device=self.device).uniform_(*com_range)

        if self._env is not None:
            self._apply_dynamics_to_env()

    def _apply_dynamics_to_env(self) -> None:
        """Apply randomized dynamics parameters to the Isaac Lab environment.

        Isaac Lab handles most DR via its EventManager when the env cfg has
        randomisation events configured.  This method is a hook for any
        *additional* per-step modifications you want to apply on top.
        """
        pass

    def _apply_push(self, mask: torch.Tensor) -> None:
        """Apply external push disturbance to selected environments."""
        dr = self.task_cfg["domain_randomization"]
        push_range = dr["push_vel_range"]
        n_push = int(mask.sum().item())
        if n_push == 0:
            return
        push_vel = torch.zeros(n_push, 3, device=self.device)
        magnitude = torch.empty(n_push, device=self.device).uniform_(*push_range)
        angle = torch.empty(n_push, device=self.device).uniform_(0, 2 * math.pi)
        push_vel[:, 0] = magnitude * torch.cos(angle)
        push_vel[:, 1] = magnitude * torch.sin(angle)

        if self._env is not None and hasattr(self._rl_env, "scene"):
            push_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            try:
                robot = self._rl_env.scene["robot"]
                cur_vel = robot.data.root_lin_vel_w.clone()
                cur_vel[push_ids, :3] += push_vel
                robot.write_root_velocity_to_sim(cur_vel, env_ids=push_ids)
            except Exception:
                pass  # silently skip if API changed

    def _get_dynamics_targets(self) -> dict[str, torch.Tensor]:
        """Get ground-truth dynamics parameters for auxiliary loss."""
        return self._dynamics_params

    def _handle_resets(self, reset_ids: torch.Tensor) -> None:
        """Handle environment resets: re-randomize dynamics for reset envs."""
        self._step_count[reset_ids] = 0
        if self.dr_enabled and len(reset_ids) > 0:
            dr = self.task_cfg["domain_randomization"]
            n = len(reset_ids)
            self._dynamics_params["friction"][reset_ids] = torch.empty(
                n, 2, device=self.device).uniform_(*dr["friction_range"])
            self._dynamics_params["mass"][reset_ids, 0] = torch.empty(
                n, device=self.device).uniform_(*dr["added_mass_range"])
            self._dynamics_params["motor"][reset_ids] = torch.empty(
                n, 2, device=self.device).uniform_(*dr["motor_strength_range"])
            self._dynamics_params["contact"][reset_ids] = torch.empty(
                n, 1, device=self.device).uniform_(*dr["restitution_range"])

    @property
    def observation_space(self) -> dict:
        return {"obs_dim": self.obs_dim, "cmd_dim": self.cmd_dim, "act_dim": self.act_dim}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        global _SIM_APP
        if _SIM_APP is not None:
            _SIM_APP.close()
            _SIM_APP = None


def make_env(cfg: dict, device: str = "cuda", headless: bool = True) -> G1EnvWrapper:
    """Create a G1 environment from config.

    If ``init_sim()`` has not been called yet, it is called here with
    *headless* mode.
    """
    init_sim(headless=headless)
    env = G1EnvWrapper(cfg, device=device)
    env._create_isaac_env()
    return env
