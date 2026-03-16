"""
PPO + LSTM baseline policy.

Recurrent policy using LSTM to process observation history implicitly.
Hidden state is maintained across steps and reset on episode boundaries.

Architecture:
    [obs, cmd, prev_action] -> Linear -> LSTM(128, 1 layer) -> hidden
    hidden -> PolicyHead -> action_mean
    hidden -> ValueHead -> value

Approximate parameter count: ~300k
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.components import MLP, PolicyHead, ValueHead, count_parameters


class LSTMPolicy(nn.Module):
    """
    LSTM-based recurrent policy for PPO.

    Maintains hidden state across timesteps within an episode.
    Hidden state must be reset on episode boundaries.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        obs_dim = cfg["task"]["observation"]["proprioception_dim"]
        cmd_dim = cfg["task"]["observation"]["command_dim"]
        act_dim = cfg["task"]["observation"]["action_dim"]

        lstm_cfg = cfg["model"]["lstm"]
        hidden_dim = lstm_cfg["hidden_dim"]
        num_layers = lstm_cfg["num_layers"]
        dropout = lstm_cfg.get("dropout", 0.0)

        input_dim = obs_dim + cmd_dim + act_dim  # always include prev action for LSTM

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Policy head
        policy_cfg = cfg["model"]["policy_head"]
        self.policy = PolicyHead(
            input_dim=hidden_dim,
            hidden_dims=policy_cfg["hidden_dims"],
            action_dim=act_dim,
            activation=policy_cfg["activation"],
        )

        # Value head
        value_cfg = cfg["model"]["value_head"]
        self.value = ValueHead(
            input_dim=hidden_dim,
            hidden_dims=value_cfg["hidden_dims"],
            activation=value_cfg["activation"],
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = "lstm"
        self.uses_history = True

    def init_hidden(self, batch_size: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def forward(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass (single step).

        Args:
            obs: (batch, obs_dim) current observation
            cmd: (batch, cmd_dim) velocity command
            prev_action: (batch, act_dim) previous action
            hidden: tuple of (h, c), each (num_layers, batch, hidden_dim)

        Returns:
            dict with: action_mean, action_log_std, value, hidden
        """
        batch_size = obs.shape[0]

        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)

        x = torch.cat([obs, cmd, prev_action], dim=-1)
        x = self.input_proj(x)

        # LSTM expects (batch, seq_len, input_dim)
        x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        features = lstm_out.squeeze(1)  # (batch, hidden_dim)

        action_mean, action_log_std = self.policy(features)
        value = self.value(features)

        return {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
            "hidden": hidden,
        }

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        cmd_seq: torch.Tensor,
        act_seq: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Forward pass over a sequence (for training with full rollouts).

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            cmd_seq: (batch, seq_len, cmd_dim)
            act_seq: (batch, seq_len, act_dim) previous actions
            hidden: initial hidden state
            mask: (batch, seq_len) True=valid

        Returns:
            dict with sequence outputs.
        """
        batch_size, seq_len = obs_seq.shape[:2]

        if hidden is None:
            hidden = self.init_hidden(batch_size, obs_seq.device)

        x = torch.cat([obs_seq, cmd_seq, act_seq], dim=-1)
        x = self.input_proj(x)

        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_dim)

        # Flatten for heads
        features_flat = lstm_out.reshape(-1, self.hidden_dim)
        action_mean, action_log_std = self.policy(features_flat)
        value = self.value(features_flat)

        return {
            "action_mean": action_mean.reshape(batch_size, seq_len, -1),
            "action_log_std": action_log_std.reshape(batch_size, seq_len, -1),
            "value": value.reshape(batch_size, seq_len),
            "hidden": hidden,
        }

    def get_param_count(self) -> int:
        return count_parameters(self)
