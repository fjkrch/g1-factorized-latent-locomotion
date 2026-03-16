"""
History buffer for storing observation-action windows.

Maintains a fixed-length FIFO buffer of recent (obs, action) pairs
for each environment. Used by Transformer and DynaMITE models.

Usage:
    from src.utils.history_buffer import HistoryBuffer
    buf = HistoryBuffer(num_envs=2048, history_len=8, obs_dim=45, act_dim=19, device="cuda")
    buf.insert(obs, action)
    obs_hist, act_hist, mask = buf.get()
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HistoryBuffer:
    """
    Fixed-length sliding window buffer for observation-action history.
    Stored on GPU for fast access during rollouts.

    Attributes:
        obs_buf: (num_envs, history_len, obs_dim) observation history
        act_buf: (num_envs, history_len, act_dim) action history
        lengths: (num_envs,) current valid length per env (for masking)
    """

    def __init__(
        self,
        num_envs: int,
        history_len: int,
        obs_dim: int,
        act_dim: int,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.obs_buf = torch.zeros(num_envs, history_len, obs_dim, device=device)
        self.act_buf = torch.zeros(num_envs, history_len, act_dim, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    def insert(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        """
        Insert a new (obs, action) pair into the buffer.
        Shifts existing entries left (oldest is dropped).

        Args:
            obs: (num_envs, obs_dim)
            action: (num_envs, act_dim)
        """
        # Shift left
        self.obs_buf[:, :-1] = self.obs_buf[:, 1:].clone()
        self.act_buf[:, :-1] = self.act_buf[:, 1:].clone()

        # Insert at end
        self.obs_buf[:, -1] = obs
        self.act_buf[:, -1] = action

        # Update valid lengths
        self.lengths = torch.clamp(self.lengths + 1, max=self.history_len)

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get current history window and padding mask.

        Returns:
            obs_hist: (num_envs, history_len, obs_dim)
            act_hist: (num_envs, history_len, act_dim)
            mask: (num_envs, history_len) boolean, True = valid, False = padding
        """
        # Create mask based on valid lengths
        indices = torch.arange(self.history_len, device=self.device).unsqueeze(0)
        # Valid positions are the most recent `length` entries (right-aligned)
        mask = indices >= (self.history_len - self.lengths.unsqueeze(1))
        return self.obs_buf, self.act_buf, mask

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """
        Reset history for specified environments (on episode reset).

        Args:
            env_ids: (N,) indices of environments to reset
        """
        self.obs_buf[env_ids] = 0.0
        self.act_buf[env_ids] = 0.0
        self.lengths[env_ids] = 0

    def reset_all(self) -> None:
        """Reset all environments."""
        self.obs_buf.zero_()
        self.act_buf.zero_()
        self.lengths.zero_()
