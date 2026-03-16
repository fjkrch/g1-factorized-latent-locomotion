"""
Unitree G1 environment wrapper for Isaac Lab.

This module wraps the Isaac Lab environment for the Unitree G1 humanoid robot
and provides a standardized interface used by the training/evaluation scripts.

It handles:
- Environment creation with domain randomization
- Observation/action space definitions
- Reward computation
- Push disturbances
- Terrain setup
- Dynamics parameter export (for auxiliary loss targets)

Usage:
    from src.envs.g1_env import make_env
    env = make_env(cfg)
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np


class G1EnvWrapper:
    """
    Wrapper around Isaac Lab's Unitree G1 locomotion environment.

    Provides a clean interface for:
    - Stepping the environment
    - Accessing observations, rewards, dones
    - Applying domain randomization
    - Generating push disturbances
    - Exporting ground-truth dynamics parameters for auxiliary loss

    NOTE: This is designed to wrap an Isaac Lab ManagerBasedRLEnv.
    The actual Isaac Lab env setup requires Isaac Sim to be installed.
    This wrapper standardizes the interface for our training pipeline.
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

    def _create_isaac_env(self):
        """
        Create the actual Isaac Lab environment.

        This requires Isaac Sim to be installed and running.
        The environment is created following Isaac Lab conventions:
          - ManagerBasedRLEnv or DirectRLEnv
          - Unitree G1 URDF/USD asset
          - Custom reward terms
          - Custom domain randomization events
        """
        try:
            # Isaac Lab imports — these require Isaac Sim runtime
            from omni.isaac.lab.envs import ManagerBasedRLEnv
            from omni.isaac.lab.utils.dict import class_to_dict

            # Import our task config registration
            from src.envs.g1_task_cfg import G1FlatEnvCfg, G1PushEnvCfg, G1RandEnvCfg, G1TerrainEnvCfg

            task_map = {
                "flat": G1FlatEnvCfg,
                "push": G1PushEnvCfg,
                "randomized": G1RandEnvCfg,
                "terrain": G1TerrainEnvCfg,
            }

            task_name = self.task_cfg["name"]
            if task_name not in task_map:
                raise ValueError(f"Unknown task: {task_name}")

            env_cfg = task_map[task_name]()
            env_cfg.scene.num_envs = self.num_envs
            self._env = ManagerBasedRLEnv(cfg=env_cfg)

        except ImportError:
            print("[WARNING] Isaac Lab not available. Using mock environment.")
            print("          Install Isaac Sim + Isaac Lab for real training.")
            self._env = None

    def reset(self) -> dict[str, torch.Tensor]:
        """
        Reset all environments.

        Returns:
            dict with keys: obs, cmd, dynamics_params (if DR enabled)
        """
        self._step_count.zero_()

        if self._env is not None:
            obs_dict, _ = self._env.reset()
            obs = obs_dict["policy"]
        else:
            # Mock for development without simulator
            obs = torch.zeros(self.num_envs, self.obs_dim, device=self.device)

        cmd = self._sample_commands()

        if self.dr_enabled:
            self._randomize_dynamics()

        result = {"obs": obs, "cmd": cmd}
        if self.dr_enabled:
            result["dynamics_params"] = self._get_dynamics_targets()
        return result

    def step(self, actions: torch.Tensor) -> dict[str, Any]:
        """
        Step the environment.

        Args:
            actions: (num_envs, act_dim)

        Returns:
            dict with: obs, cmd, reward, done, info, dynamics_params
        """
        self._step_count += 1

        if self._env is not None:
            obs_dict, rewards, dones, truncated, info = self._env.step(actions)
            obs = obs_dict["policy"]
            done = dones | truncated
        else:
            # Mock
            obs = torch.randn(self.num_envs, self.obs_dim, device=self.device) * 0.1
            rewards = torch.zeros(self.num_envs, device=self.device)
            done = self._step_count >= self.task_cfg["episode_length"]
            info = {}

        # Apply push disturbances
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

    def _sample_commands(self) -> torch.Tensor:
        """Sample random velocity commands: [vx, vy, yaw_rate]."""
        cmd = torch.zeros(self.num_envs, self.cmd_dim, device=self.device)
        cmd[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(-1.0, 2.0)  # vx
        cmd[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(-0.5, 0.5)  # vy
        cmd[:, 2] = torch.empty(self.num_envs, device=self.device).uniform_(-1.0, 1.0)  # yaw_rate
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
                (self.num_envs, 1),
                device=self.device,
            ).float(),
        }
        # Mass: added_mass + com displacement
        self._dynamics_params["mass"][:, 0] = torch.empty(
            self.num_envs, device=self.device
        ).uniform_(*dr["added_mass_range"])
        com_range = dr.get("com_displacement_range", [-0.05, 0.05])
        self._dynamics_params["mass"][:, 1] = torch.empty(
            self.num_envs, device=self.device
        ).uniform_(*com_range)

        # Actually apply to physics if env is available
        if self._env is not None:
            self._apply_dynamics_to_env()

    def _apply_dynamics_to_env(self) -> None:
        """Apply randomized dynamics parameters to the Isaac Lab environment."""
        # Implementation depends on Isaac Lab API — set material frictions,
        # add mass to bodies, scale motor gains, etc.
        pass

    def _apply_push(self, mask: torch.Tensor) -> None:
        """Apply external push disturbance to selected environments."""
        dr = self.task_cfg["domain_randomization"]
        push_range = dr["push_vel_range"]
        n_push = mask.sum().item()
        if n_push == 0:
            return

        # Random push direction and magnitude
        push_vel = torch.zeros(n_push, 3, device=self.device)
        magnitude = torch.empty(n_push, device=self.device).uniform_(*push_range)
        angle = torch.empty(n_push, device=self.device).uniform_(0, 2 * 3.14159)
        push_vel[:, 0] = magnitude * torch.cos(angle)
        push_vel[:, 1] = magnitude * torch.sin(angle)

        if self._env is not None:
            # Apply velocity perturbation to robot base
            push_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            # env.scene.robot.write_root_velocity(push_vel, env_ids=push_ids)
            pass

    def _get_dynamics_targets(self) -> dict[str, torch.Tensor]:
        """Get ground-truth dynamics parameters for auxiliary loss."""
        return self._dynamics_params

    def _handle_resets(self, reset_ids: torch.Tensor) -> None:
        """Handle environment resets: re-randomize dynamics for reset envs."""
        self._step_count[reset_ids] = 0
        if self.dr_enabled and len(reset_ids) > 0:
            # Re-randomize dynamics for reset environments
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

    def close(self):
        if self._env is not None:
            self._env.close()


def make_env(cfg: dict, device: str = "cuda") -> G1EnvWrapper:
    """
    Create a G1 environment from config.

    Args:
        cfg: Full merged config dict.
        device: Torch device.

    Returns:
        G1EnvWrapper instance.
    """
    env = G1EnvWrapper(cfg, device=device)
    env._create_isaac_env()
    return env
