"""
Reward function definitions for Unitree G1 locomotion.

Each reward term is a standalone function taking environment state
and returning a per-environment scalar reward.

Reward composition is controlled by weights in the config YAML.

Usage:
    from src.envs.reward import compute_rewards
    reward_dict = compute_rewards(state, cfg)
    total_reward = sum(w * r for w, r in reward_dict.values())
"""

from __future__ import annotations

import torch


def tracking_lin_vel(
    cmd_vel: torch.Tensor,
    base_lin_vel: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """
    Reward for tracking commanded linear velocity (x, y).
    Gaussian kernel: exp(-||v_cmd - v_actual||^2 / sigma).
    """
    error = torch.sum((cmd_vel[:, :2] - base_lin_vel[:, :2]) ** 2, dim=-1)
    return torch.exp(-error / sigma)


def tracking_ang_vel(
    cmd_yaw_rate: torch.Tensor,
    base_ang_vel_z: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Reward for tracking commanded yaw rate."""
    error = (cmd_yaw_rate - base_ang_vel_z) ** 2
    return torch.exp(-error / sigma)


def lin_vel_z_penalty(base_lin_vel_z: torch.Tensor) -> torch.Tensor:
    """Penalize vertical linear velocity (bouncing)."""
    return base_lin_vel_z ** 2


def ang_vel_xy_penalty(base_ang_vel: torch.Tensor) -> torch.Tensor:
    """Penalize roll/pitch angular velocity."""
    return torch.sum(base_ang_vel[:, :2] ** 2, dim=-1)


def orientation_penalty(projected_gravity: torch.Tensor) -> torch.Tensor:
    """Penalize non-upright orientation. projected_gravity should be ~[0,0,-1]."""
    return torch.sum(projected_gravity[:, :2] ** 2, dim=-1)


def base_height_penalty(
    base_height: torch.Tensor,
    target_height: float = 0.7,
) -> torch.Tensor:
    """Penalize deviation from target standing height."""
    return (base_height - target_height) ** 2


def torque_penalty(torques: torch.Tensor) -> torch.Tensor:
    """Penalize large joint torques."""
    return torch.sum(torques ** 2, dim=-1)


def joint_acceleration_penalty(
    joint_vel: torch.Tensor,
    prev_joint_vel: torch.Tensor,
    dt: float = 0.02,
) -> torch.Tensor:
    """Penalize large joint accelerations (smoothness)."""
    acc = (joint_vel - prev_joint_vel) / dt
    return torch.sum(acc ** 2, dim=-1)


def action_rate_penalty(
    action: torch.Tensor,
    prev_action: torch.Tensor,
) -> torch.Tensor:
    """Penalize large changes in action between steps."""
    return torch.sum((action - prev_action) ** 2, dim=-1)


def feet_air_time_reward(
    feet_air_time: torch.Tensor,
    contact_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Reward feet for spending appropriate time in the air (gait encouragement).
    feet_air_time: time since last contact for each foot.
    contact_mask: True if foot is in contact.
    """
    # Reward when foot lands after being in air for reasonable time
    reward = torch.sum(
        (feet_air_time - threshold) * contact_mask.float(),
        dim=-1,
    )
    return torch.clamp(reward, min=0.0)


def collision_penalty(contact_forces: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Penalize contacts on non-foot body parts (knees, torso, etc)."""
    return (contact_forces.norm(dim=-1) > threshold).float().sum(dim=-1)


def stumble_penalty(
    feet_contact_forces: torch.Tensor,
    feet_air_time: torch.Tensor,
) -> torch.Tensor:
    """Penalize foot contacts that happen too quickly (stumbling)."""
    quick_contact = (feet_air_time < 0.1).float()
    has_force = (feet_contact_forces.norm(dim=-1) > 5.0).float()
    return torch.sum(quick_contact * has_force, dim=-1)


def stand_still_penalty(
    cmd_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint motion when standing still (cmd ~= 0)."""
    cmd_mag = cmd_vel.norm(dim=-1)
    should_stand = (cmd_mag < threshold).float()
    motion = torch.sum(joint_vel ** 2, dim=-1)
    return should_stand * motion


def compute_rewards(
    state: dict[str, torch.Tensor],
    reward_cfg: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute total weighted reward from state.

    Args:
        state: dict of tensors (base_lin_vel, base_ang_vel, etc)
        reward_cfg: dict of reward_name -> weight

    Returns:
        total_reward: (num_envs,) summed weighted reward
        reward_components: dict of reward_name -> (num_envs,) individual terms
    """
    components = {}
    total = torch.zeros(state["base_lin_vel"].shape[0], device=state["base_lin_vel"].device)

    reward_fns = {
        "tracking_lin_vel": lambda: tracking_lin_vel(state["cmd_vel"], state["base_lin_vel"]),
        "tracking_ang_vel": lambda: tracking_ang_vel(state["cmd_vel"][:, 2], state["base_ang_vel"][:, 2]),
        "lin_vel_z": lambda: lin_vel_z_penalty(state["base_lin_vel"][:, 2]),
        "ang_vel_xy": lambda: ang_vel_xy_penalty(state["base_ang_vel"]),
        "orientation": lambda: orientation_penalty(state["projected_gravity"]),
        "base_height": lambda: base_height_penalty(state["base_height"]),
        "torques": lambda: torque_penalty(state["torques"]),
        "action_rate": lambda: action_rate_penalty(state["action"], state["prev_action"]),
    }

    for name, weight in reward_cfg.items():
        if name in reward_fns and weight != 0.0:
            r = reward_fns[name]()
            components[name] = r
            total += weight * r

    return total, components
