"""
Tests for config loading and merging.

Usage:
    pytest tests/test_config.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.utils.config import load_config, deep_merge, apply_cli_overrides


class TestConfig:
    def test_load_base(self):
        cfg = load_config(base_path="configs/base.yaml")
        assert "task" in cfg
        assert "model" in cfg
        assert "train" in cfg
        assert cfg["seed"] == 42

    def test_task_override(self):
        cfg = load_config(
            base_path="configs/base.yaml",
            task_path="configs/task/push.yaml",
        )
        assert cfg["task"]["name"] == "push"
        assert cfg["task"]["domain_randomization"]["enabled"] == True
        assert cfg["task"]["domain_randomization"]["push_interval"] == 100

    def test_model_override(self):
        cfg = load_config(
            base_path="configs/base.yaml",
            model_path="configs/model/dynamite.yaml",
        )
        assert cfg["model"]["name"] == "dynamite"
        assert cfg["model"]["type"] == "dynamite"
        assert cfg["model"]["latent"]["factorized"] == True

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        result = deep_merge(base, override)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["d"] == 3
        assert result["e"] == 5

    def test_cli_overrides(self):
        cfg = {"train": {"learning_rate": 0.001}, "seed": 42}
        result = apply_cli_overrides(cfg, ["train.learning_rate=0.0001", "seed=123"])
        assert result["train"]["learning_rate"] == 0.0001
        assert result["seed"] == 123

    def test_full_stack(self):
        cfg = load_config(
            base_path="configs/base.yaml",
            task_path="configs/task/randomized.yaml",
            model_path="configs/model/dynamite.yaml",
            train_path="configs/train/default.yaml",
            overrides=["seed=99"],
        )
        assert cfg["task"]["name"] == "randomized"
        assert cfg["model"]["name"] == "dynamite"
        assert cfg["seed"] == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
