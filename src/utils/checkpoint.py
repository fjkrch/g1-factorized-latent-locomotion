"""
Checkpoint save/load utility.

Saves: model state dict, optimizer state, running stats normalizer,
       step count, config, and training stats.

Usage:
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    save_checkpoint(run_dir, model, optimizer, step, cfg, stats)
    model, optimizer, step, stats = load_checkpoint(path, model, optimizer)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    run_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict,
    stats: dict[str, Any] | None = None,
    obs_normalizer: Any | None = None,
    value_normalizer: Any | None = None,
    is_best: bool = False,
) -> Path:
    """
    Save a training checkpoint.

    Saves to: {run_dir}/checkpoints/ckpt_{step}.pt
    Also saves: {run_dir}/checkpoints/latest.pt (symlink or copy)
    If is_best: {run_dir}/checkpoints/best.pt

    Args:
        run_dir: Run output directory.
        model: Policy model.
        optimizer: Optimizer.
        step: Current training step.
        cfg: Full config dict.
        stats: Optional training statistics dict.
        obs_normalizer: Optional observation normalizer state.
        value_normalizer: Optional value normalizer state.
        is_best: Whether this is the best checkpoint so far.

    Returns:
        Path to saved checkpoint file.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }

    if stats is not None:
        payload["stats"] = stats
    if obs_normalizer is not None:
        payload["obs_normalizer"] = (
            obs_normalizer.state_dict() if hasattr(obs_normalizer, "state_dict") else obs_normalizer
        )
    if value_normalizer is not None:
        payload["value_normalizer"] = (
            value_normalizer.state_dict() if hasattr(value_normalizer, "state_dict") else value_normalizer
        )

    # Save numbered checkpoint
    path = ckpt_dir / f"ckpt_{step}.pt"
    torch.save(payload, path)

    # Save latest
    latest_path = ckpt_dir / "latest.pt"
    torch.save(payload, latest_path)

    # Save best
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(payload, best_path)

    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
) -> tuple[nn.Module, torch.optim.Optimizer | None, int, dict]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Tuple of (model, optimizer, step, stats).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    step = ckpt.get("step", 0)
    stats = ckpt.get("stats", {})

    return model, optimizer, step, stats


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in a run directory."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest

    # Fallback: find highest numbered checkpoint
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def find_best_checkpoint(run_dir: str | Path) -> Path | None:
    """Find the best checkpoint in a run directory."""
    best = Path(run_dir) / "checkpoints" / "best.pt"
    return best if best.exists() else None
