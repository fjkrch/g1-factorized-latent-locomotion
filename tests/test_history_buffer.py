"""
Tests for history buffer.

Usage:
    pytest tests/test_history_buffer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from src.utils.history_buffer import HistoryBuffer


class TestHistoryBuffer:
    def setup_method(self):
        self.num_envs = 4
        self.history_len = 8
        self.obs_dim = 45
        self.act_dim = 19
        self.buf = HistoryBuffer(
            self.num_envs, self.history_len, self.obs_dim, self.act_dim, device="cpu"
        )

    def test_initial_state(self):
        obs, act, mask = self.buf.get()
        assert obs.shape == (self.num_envs, self.history_len, self.obs_dim)
        assert act.shape == (self.num_envs, self.history_len, self.act_dim)
        assert mask.shape == (self.num_envs, self.history_len)
        assert mask.sum() == 0  # all padding initially

    def test_insert_updates_buffer(self):
        obs = torch.ones(self.num_envs, self.obs_dim)
        act = torch.ones(self.num_envs, self.act_dim) * 2
        self.buf.insert(obs, act)
        o, a, mask = self.buf.get()
        assert torch.allclose(o[:, -1], obs)
        assert torch.allclose(a[:, -1], act)
        assert mask[:, -1].all()  # last position valid

    def test_fifo_behavior(self):
        for i in range(self.history_len + 2):
            obs = torch.full((self.num_envs, self.obs_dim), float(i))
            act = torch.full((self.num_envs, self.act_dim), float(i))
            self.buf.insert(obs, act)

        o, a, mask = self.buf.get()
        # Buffer should contain [2, 3, ..., 9] (oldest=2, newest=9)
        assert torch.allclose(o[0, -1, 0], torch.tensor(float(self.history_len + 1)))
        assert torch.allclose(o[0, 0, 0], torch.tensor(float(2)))
        assert mask.all()  # all valid after enough inserts

    def test_reset_envs(self):
        obs = torch.ones(self.num_envs, self.obs_dim)
        act = torch.ones(self.num_envs, self.act_dim)
        self.buf.insert(obs, act)

        # Reset env 0 and 2
        self.buf.reset_envs(torch.tensor([0, 2]))
        o, a, mask = self.buf.get()
        assert o[0].sum() == 0  # reset
        assert o[2].sum() == 0  # reset
        assert o[1, -1].sum() > 0  # not reset

    def test_mask_correctness(self):
        """After N inserts, exactly N positions should be valid."""
        for i in range(3):
            obs = torch.ones(self.num_envs, self.obs_dim)
            act = torch.ones(self.num_envs, self.act_dim)
            self.buf.insert(obs, act)

        _, _, mask = self.buf.get()
        assert mask[0].sum() == 3
        assert mask[0, -1] == True
        assert mask[0, -2] == True
        assert mask[0, -3] == True
        assert mask[0, -4] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
