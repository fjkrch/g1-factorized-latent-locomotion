"""
PPO + MLP baseline policy.

Standard feedforward MLP policy with no history conditioning.
Processes only the current observation.

Architecture:
    obs (obs_dim) -> MLP [256, 256, 128] -> action_mean (act_dim)
    obs (obs_dim) -> MLP [256, 256, 128] -> value (1)

Approximate parameter count: ~200k
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.components import MLP, PolicyHead, ValueHead, count_parameters


class MLPPolicy(nn.Module):
    """
    MLP policy for PPO. No history, no recurrence.

    This is the simplest baseline: the policy sees only the current
    proprioceptive observation and velocity command.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        obs_dim = cfg["task"]["observation"]["proprioception_dim"]
        cmd_dim = cfg["task"]["observation"]["command_dim"]
        act_dim = cfg["task"]["observation"]["action_dim"]
        include_prev_action = cfg["task"]["observation"].get("include_previous_action", False)

        model_cfg = cfg["model"]["mlp"]
        hidden_dims = model_cfg["hidden_dims"]
        activation = model_cfg["activation"]

        input_dim = obs_dim + cmd_dim
        if include_prev_action:
            input_dim += act_dim
        self.include_prev_action = include_prev_action

        # Policy network
        self.policy = PolicyHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=act_dim,
            activation=activation,
        )

        # Value network (separate)
        value_cfg = cfg["model"]["value"]
        self.value = ValueHead(
            input_dim=input_dim,
            hidden_dims=value_cfg["hidden_dims"],
            activation=value_cfg["activation"],
        )

        # Metadata
        self.model_type = "mlp"
        self.uses_history = False

    def forward(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            cmd: (batch, cmd_dim)
            prev_action: (batch, act_dim) optional

        Returns:
            dict with keys: action_mean, action_log_std, value
        """
        parts = [obs, cmd]
        if self.include_prev_action and prev_action is not None:
            parts.append(prev_action)
        x = torch.cat(parts, dim=-1)

        action_mean, action_log_std = self.policy(x)
        value = self.value(x)

        return {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
        }

    def get_param_count(self) -> int:
        return count_parameters(self)
