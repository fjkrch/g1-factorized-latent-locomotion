"""
Configuration loading and merging utility.

Usage:
    from src.utils.config import load_config, merge_configs, save_config

Flow:
    base.yaml -> task/*.yaml -> model/*.yaml -> train/*.yaml -> CLI overrides
    Final merged config dict is used everywhere downstream.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return as dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_cli_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-separated CLI overrides.
    Example: --set train.learning_rate=1e-4
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Use key=value")
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        # Try to parse value as YAML for type inference
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[keys[-1]] = parsed
    return cfg


def load_config(
    base_path: str = "configs/base.yaml",
    task_path: str | None = None,
    model_path: str | None = None,
    train_path: str | None = None,
    eval_path: str | None = None,
    ablation_path: str | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load and merge configs in priority order:
    base -> task -> model -> train -> eval -> ablation -> CLI overrides
    """
    cfg = load_yaml(base_path)

    for path in [task_path, model_path, train_path, eval_path, ablation_path]:
        if path is not None:
            layer = load_yaml(path)
            cfg = deep_merge(cfg, layer)

    if overrides:
        cfg = apply_cli_overrides(cfg, overrides)

    return cfg


def save_config(cfg: dict, output_dir: str | Path) -> Path:
    """Save merged config to output directory for reproducibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def config_to_flat(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config to dot-separated keys for logging."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(config_to_flat(v, key))
        else:
            flat[key] = v
    return flat


def make_run_dir(cfg: dict) -> Path:
    """
    Create and return run output directory.
    Pattern: outputs/{task}/{model}/seed_{seed}/{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = cfg["task"]["name"]
    model_name = cfg["model"]["name"]
    seed = cfg["seed"]
    base = cfg.get("output", {}).get("base_dir", "outputs")

    run_dir = Path(base) / task_name / model_name / f"seed_{seed}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    save_config(cfg, run_dir)

    return run_dir
