"""
Tests for model architectures.

Verifies:
- Forward pass shapes
- Parameter counts in expected range
- Gradient flow
- History conditioning works
- Latent head produces valid outputs
- Auxiliary loss computes correctly

Usage:
    pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from src.models import build_model
from src.models.components import count_parameters
from src.utils.config import load_config


def _make_cfg(model_type: str) -> dict:
    """Load config for a given model type."""
    cfg = load_config(
        base_path="configs/base.yaml",
        task_path="configs/task/flat.yaml",
        model_path=f"configs/model/{model_type}.yaml",
    )
    # Reduce num_envs for testing
    cfg["task"]["num_envs"] = 4
    return cfg


# ─── MLP Tests ───

class TestMLPPolicy:
    def setup_method(self):
        self.cfg = _make_cfg("mlp")
        self.model = build_model(self.cfg)
        self.batch = 4
        self.obs_dim = self.cfg["task"]["observation"]["proprioception_dim"]
        self.cmd_dim = self.cfg["task"]["observation"]["command_dim"]
        self.act_dim = self.cfg["task"]["observation"]["action_dim"]

    def test_forward_shape(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        prev_act = torch.randn(self.batch, self.act_dim)
        out = self.model(obs=obs, cmd=cmd, prev_action=prev_act)
        assert out["action_mean"].shape == (self.batch, self.act_dim)
        assert out["action_log_std"].shape == (self.batch, self.act_dim)
        assert out["value"].shape == (self.batch,)

    def test_param_count(self):
        params = count_parameters(self.model)
        assert 50_000 < params < 500_000, f"MLP params: {params}"

    def test_gradient_flow(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        prev_act = torch.randn(self.batch, self.act_dim)
        out = self.model(obs=obs, cmd=cmd, prev_action=prev_act)
        loss = out["action_mean"].sum() + out["value"].sum()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ─── LSTM Tests ───

class TestLSTMPolicy:
    def setup_method(self):
        self.cfg = _make_cfg("lstm")
        self.model = build_model(self.cfg)
        self.batch = 4
        self.obs_dim = self.cfg["task"]["observation"]["proprioception_dim"]
        self.cmd_dim = self.cfg["task"]["observation"]["command_dim"]
        self.act_dim = self.cfg["task"]["observation"]["action_dim"]

    def test_forward_shape(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        prev_act = torch.randn(self.batch, self.act_dim)
        out = self.model(obs=obs, cmd=cmd, prev_action=prev_act)
        assert out["action_mean"].shape == (self.batch, self.act_dim)
        assert out["value"].shape == (self.batch,)
        assert out["hidden"] is not None

    def test_hidden_state_persistence(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        prev_act = torch.randn(self.batch, self.act_dim)
        out1 = self.model(obs=obs, cmd=cmd, prev_action=prev_act)
        out2 = self.model(obs=obs, cmd=cmd, prev_action=prev_act, hidden=out1["hidden"])
        # Outputs should differ due to different hidden state
        assert not torch.allclose(out1["action_mean"], out2["action_mean"])


# ─── Transformer Tests ───

class TestTransformerPolicy:
    def setup_method(self):
        self.cfg = _make_cfg("transformer")
        self.model = build_model(self.cfg)
        self.batch = 4
        self.obs_dim = self.cfg["task"]["observation"]["proprioception_dim"]
        self.cmd_dim = self.cfg["task"]["observation"]["command_dim"]
        self.act_dim = self.cfg["task"]["observation"]["action_dim"]
        self.hist_len = self.cfg["model"]["history_len"]

    def test_forward_shape(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim)
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim)
        mask = torch.ones(self.batch, self.hist_len, dtype=torch.bool)
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask)
        assert out["action_mean"].shape == (self.batch, self.act_dim)
        assert out["value"].shape == (self.batch,)

    def test_partial_mask(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim)
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim)
        mask = torch.zeros(self.batch, self.hist_len, dtype=torch.bool)
        mask[:, -3:] = True  # only last 3 valid
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask)
        assert out["action_mean"].shape == (self.batch, self.act_dim)


# ─── DynaMITE Tests ───

class TestDynaMITEPolicy:
    def setup_method(self):
        self.cfg = _make_cfg("dynamite")
        self.model = build_model(self.cfg)
        self.batch = 4
        self.obs_dim = self.cfg["task"]["observation"]["proprioception_dim"]
        self.cmd_dim = self.cfg["task"]["observation"]["command_dim"]
        self.act_dim = self.cfg["task"]["observation"]["action_dim"]
        self.hist_len = self.cfg["model"]["history_len"]

    def test_forward_shape(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim)
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim)
        mask = torch.ones(self.batch, self.hist_len, dtype=torch.bool)
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask)
        assert out["action_mean"].shape == (self.batch, self.act_dim)
        assert out["value"].shape == (self.batch,)
        assert out["latent_z"] is not None
        assert out["latent_z"].shape == (self.batch, 24)  # total_dim

    def test_latent_factors(self):
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim)
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim)
        mask = torch.ones(self.batch, self.hist_len, dtype=torch.bool)
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask)
        factors = out["latent_factors"]
        assert "friction" in factors
        assert "mass" in factors
        assert factors["friction"].shape == (self.batch, 4)
        assert factors["mass"].shape == (self.batch, 6)

    def test_auxiliary_loss(self):
        self.model.train()
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim)
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim)
        mask = torch.ones(self.batch, self.hist_len, dtype=torch.bool)

        # Provide dynamics targets
        targets = {
            "friction": torch.randn(self.batch, 2),
            "mass": torch.randn(self.batch, 2),
            "motor": torch.randn(self.batch, 2),
            "contact": torch.randn(self.batch, 1),
            "delay": torch.randn(self.batch, 1),
        }
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask,
                        dynamics_targets=targets)
        assert out["aux_loss"].item() > 0

    def test_latent_bounded(self):
        """Latent values should be bounded by tanh activation."""
        obs = torch.randn(self.batch, self.obs_dim)
        cmd = torch.randn(self.batch, self.cmd_dim)
        obs_hist = torch.randn(self.batch, self.hist_len, self.obs_dim) * 10
        act_hist = torch.randn(self.batch, self.hist_len, self.act_dim) * 10
        mask = torch.ones(self.batch, self.hist_len, dtype=torch.bool)
        out = self.model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                        act_hist=act_hist, hist_mask=mask)
        z = out["latent_z"]
        assert z.abs().max() <= 1.0 + 1e-6, f"Latent not bounded: max={z.abs().max()}"

    def test_param_count(self):
        params = count_parameters(self.model)
        assert 200_000 < params < 1_000_000, f"DynaMITE params: {params}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
