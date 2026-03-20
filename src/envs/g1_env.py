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

    # Default push steps: apply pushes at these episode steps
    # Designed to work even with very short episodes (~17 steps)
    DEFAULT_PUSH_STEPS = [5, 10, 15]

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

        # Push disturbance system
        self._init_push_system()

    def _init_push_system(self) -> None:
        """Initialize the deterministic push disturbance system.

        This system guarantees pushes occur at specific episode steps,
        regardless of episode length. Works with vectorized environments.
        """
        dr = self.task_cfg["domain_randomization"]

        # Determine push steps from config or use defaults
        # push_steps can be a list like [5, 10, 15] or derived from push_interval
        if "push_steps" in dr and dr["push_steps"]:
            self._push_steps = list(dr["push_steps"])
        elif dr.get("push_interval", 0) > 0:
            # Legacy: convert interval to explicit steps (first 3 multiples)
            interval = dr["push_interval"]
            self._push_steps = [interval * i for i in range(1, 4)]
        else:
            # Default: push at steps 5, 10, 15 to guarantee pushes in short episodes
            self._push_steps = self.DEFAULT_PUSH_STEPS.copy()

        # Per-env tracking
        self._pushes_this_episode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Logging: track all pushes for statistics
        self._push_log: list[dict] = []  # list of {env_id, step, magnitude}
        self._episode_push_counts: list[int] = []  # pushes per completed episode
        self._total_episodes_completed = 0

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
            # NEVER silently fall back to mock — a mock run produces
            # zero reward and wastes hours.  Raise so the caller can
            # retry or abort.
            raise RuntimeError(
                f"[FATAL] Could not create Isaac Lab env: {exc}\n"
                "         Check network/USD cache. Refusing to fall back to mock."
            ) from exc

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
            terminated = terminated.to(self.device)
            truncated = truncated.to(self.device)
            done = terminated | truncated
        else:
            # Mock
            obs = torch.randn(self.num_envs, self.obs_dim, device=self.device) * 0.1
            rewards = torch.zeros(self.num_envs, device=self.device)
            done = self._step_count >= self.task_cfg["episode_length"]
            terminated = done.clone() if isinstance(done, torch.Tensor) else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            info = {}

        # Apply deterministic push disturbances at configured steps.
        # Always uses fixed episode steps (e.g. [5, 10, 15]) to guarantee
        # pushes fire even in short episodes. The old interval-based system
        # (push_interval % step == 0) is removed because it never triggered
        # when episodes were shorter than push_interval.
        push_applied = self._apply_deterministic_pushes()

        cmd = self._sample_commands()

        # Handle resets (must be after push to log episode push counts)
        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._handle_resets(reset_ids)

        result = {
            "obs": obs,
            "cmd": cmd,
            "reward": rewards,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "reset_ids": reset_ids,
            "push_applied": push_applied,  # bool tensor indicating which envs got pushed
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

    def _apply_dynamics_to_env(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply randomized dynamics parameters to the Isaac Lab physics engine.

        Sets physics material properties (friction, restitution) and body masses
        via Isaac Lab's PhysX rigid body view API.

        Args:
            env_ids: Optional tensor of env indices to update. If None, updates all.
        """
        if self._env is None:
            return

        try:
            robot = self._rl_env.scene["robot"]
            if not hasattr(robot, "root_physx_view"):
                print("[DR] Warning: robot has no root_physx_view — skipping DR")
                return

            view = robot.root_physx_view

            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)

            # -- Friction & restitution via material properties --
            # PhysX tensor API: set_material_properties(data, indices)
            #   data:    [count, max_shapes, 3]  where count == len(indices)
            #   indices: which articulations to update
            # "Sparse subset of shape data is not supported" — so we get the
            # full tensor, modify target rows, and write back with ALL indices.
            mat = view.get_material_properties()  # [num_envs, num_shapes, 3]
            mat_device = mat.device  # PhysX tensors may be on CPU
            idx = env_ids.to(mat_device)
            fric = self._dynamics_params["friction"][env_ids, 0].to(mat_device)  # [n]
            rest = self._dynamics_params["contact"][env_ids, 0].to(mat_device)   # [n]
            # Modify only target env rows in the full tensor
            mat[idx, :, 0] = fric.unsqueeze(1)  # static friction
            mat[idx, :, 1] = fric.unsqueeze(1)  # dynamic friction
            mat[idx, :, 2] = rest.unsqueeze(1)   # restitution
            all_indices = torch.arange(view.count, device=mat_device)
            view.set_material_properties(mat, all_indices)

            # -- Mass randomization --
            # PhysX tensor API: set_masses(data, indices) — same pattern.
            # "Sparse subset of link masses is not supported" — full write.
            if not hasattr(self, '_base_masses'):
                self._base_masses = view.get_masses().clone()
            masses = self._base_masses.clone()  # [num_envs, num_bodies]
            mass_device = masses.device
            m_idx = env_ids.to(mass_device)
            added_mass = self._dynamics_params["mass"][env_ids, 0].to(mass_device)
            # Modify only target env rows in the full tensor
            masses[m_idx, 0] += added_mass  # modify root body mass
            all_mass_indices = torch.arange(view.count, device=mass_device)
            view.set_masses(masses, all_mass_indices)

        except Exception as e:
            print(f"[DR] Warning: Could not apply dynamics to physics: {e}")

    def _apply_deterministic_pushes(self) -> torch.Tensor:
        """Apply pushes at deterministic episode steps.

        Guarantees pushes occur even in short episodes by using fixed
        step indices (e.g. [5, 10, 15]) instead of modular intervals.
        Used in both training and evaluation.

        Returns:
            Boolean tensor [num_envs] indicating which envs received a push.
        """
        dr = self.task_cfg["domain_randomization"]
        push_range = dr.get("push_vel_range", [0.0, 0.0])

        # Skip if pushes are disabled (zero magnitude range)
        if push_range[0] == 0.0 and push_range[1] == 0.0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check which envs are at a push step
        push_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for step in self._push_steps:
            push_mask |= (self._step_count == step)

        if push_mask.any():
            self._apply_push(push_mask, log=True)

        return push_mask

    def _apply_push(self, mask: torch.Tensor, log: bool = True) -> torch.Tensor:
        """Apply external push disturbance to selected environments.

        Args:
            mask: Boolean tensor [num_envs] indicating which envs to push.
            log: Whether to log the push for statistics.

        Returns:
            Tensor of push magnitudes applied [n_push].
        """
        dr = self.task_cfg["domain_randomization"]
        push_range = dr.get("push_vel_range", [0.0, 0.0])
        n_push = int(mask.sum().item())

        if n_push == 0:
            return torch.tensor([], device=self.device)

        # Generate random push velocities
        magnitude = torch.empty(n_push, device=self.device).uniform_(*push_range)
        angle = torch.empty(n_push, device=self.device).uniform_(0, 2 * math.pi)
        push_vel = torch.zeros(n_push, 3, device=self.device)
        push_vel[:, 0] = magnitude * torch.cos(angle)
        push_vel[:, 1] = magnitude * torch.sin(angle)

        # Apply to simulator
        push_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if self._env is not None and hasattr(self._rl_env, "scene"):
            try:
                robot = self._rl_env.scene["robot"]
                # root_vel_w is [num_envs, 6]: [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
                # write_root_velocity_to_sim expects [len(env_ids), 6] when env_ids given
                subset_vel = robot.data.root_vel_w[push_ids].clone()  # [n_push, 6]
                subset_vel[:, :3] += push_vel  # add push to linear velocity
                robot.write_root_velocity_to_sim(subset_vel, env_ids=push_ids)
            except Exception as e:
                print(f"[Push] ERROR: Could not apply push velocity: {e}")

        # Update tracking and logging
        if log:
            self._pushes_this_episode[push_ids] += 1
            for i, env_id in enumerate(push_ids.tolist()):
                self._push_log.append({
                    "env_id": env_id,
                    "step": self._step_count[env_id].item(),
                    "magnitude": magnitude[i].item(),
                })

        return magnitude

    def _get_dynamics_targets(self) -> dict[str, torch.Tensor]:
        """Get ground-truth dynamics parameters for auxiliary loss."""
        return self._dynamics_params

    def _handle_resets(self, reset_ids: torch.Tensor) -> None:
        """Handle environment resets: re-randomize dynamics for reset envs."""
        # Log push counts for completed episodes before resetting
        for env_id in reset_ids.tolist():
            push_count = self._pushes_this_episode[env_id].item()
            self._episode_push_counts.append(push_count)
            self._total_episodes_completed += 1

        # Reset step count and push tracking for these envs
        self._step_count[reset_ids] = 0
        self._pushes_this_episode[reset_ids] = 0

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

            # Apply re-randomized dynamics to Isaac Lab physics
            if self._env is not None:
                self._apply_dynamics_to_env(env_ids=reset_ids)

    def get_push_statistics(self) -> dict:
        """Get statistics about push disturbances applied during evaluation.

        Returns:
            Dict with push statistics:
            - total_pushes: Total number of pushes applied
            - total_episodes: Total completed episodes
            - avg_pushes_per_episode: Average pushes per episode
            - pct_episodes_with_push: Percentage of episodes with ≥1 push
            - push_steps_config: Configured push steps
            - push_log: Detailed log of all pushes
        """
        total_pushes = len(self._push_log)
        total_episodes = len(self._episode_push_counts)

        if total_episodes == 0:
            return {
                "total_pushes": total_pushes,
                "total_episodes": 0,
                "avg_pushes_per_episode": 0.0,
                "pct_episodes_with_push": 0.0,
                "push_steps_config": self._push_steps,
                "push_log": self._push_log,
            }

        avg_pushes = total_pushes / total_episodes
        episodes_with_push = sum(1 for c in self._episode_push_counts if c > 0)
        pct_with_push = 100.0 * episodes_with_push / total_episodes

        # Compute magnitude statistics
        magnitudes = [p["magnitude"] for p in self._push_log]
        avg_magnitude = sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

        return {
            "total_pushes": total_pushes,
            "total_episodes": total_episodes,
            "avg_pushes_per_episode": round(avg_pushes, 2),
            "pct_episodes_with_push": round(pct_with_push, 1),
            "avg_push_magnitude": round(avg_magnitude, 3),
            "push_steps_config": self._push_steps,
            "episode_push_counts": self._episode_push_counts.copy(),
        }

    def reset_push_statistics(self) -> None:
        """Reset push statistics for a new evaluation run."""
        self._push_log.clear()
        self._episode_push_counts.clear()
        self._total_episodes_completed = 0
        self._pushes_this_episode.zero_()

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
