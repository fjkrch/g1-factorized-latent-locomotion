"""
Checkpoint utilities — extended helpers beyond basic save/load.

Provides:
  - Checkpoint integrity verification
  - Batch checkpoint discovery
  - Checkpoint metadata extraction
  - Corrupted checkpoint detection

Usage:
    from src.utils.checkpoint_utils import (
        verify_checkpoint, list_checkpoints, get_checkpoint_meta,
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def verify_checkpoint(path: str | Path) -> tuple[bool, str]:
    """
    Verify a checkpoint file is loadable and contains required keys.

    Returns (is_valid, message).
    """
    path = Path(path)
    if not path.exists():
        return False, f"file not found: {path}"
    if path.stat().st_size < 1024:
        return False, f"file too small ({path.stat().st_size} bytes), likely corrupted"
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"failed to load: {e}"

    required = ["model_state_dict"]
    for key in required:
        if key not in ckpt:
            return False, f"missing key: {key}"

    return True, "ok"


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    """List all numbered checkpoint files in a run directory, sorted by step."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    files = sorted(ckpt_dir.glob("ckpt_*.pt"))
    return files


def get_checkpoint_meta(path: str | Path) -> dict[str, Any]:
    """Extract metadata from a checkpoint without loading tensors fully."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = {
            "path": str(path),
            "step": ckpt.get("step", "unknown"),
            "has_optimizer": "optimizer_state_dict" in ckpt,
            "has_config": "config" in ckpt,
            "has_stats": "stats" in ckpt,
            "model_keys": len(ckpt.get("model_state_dict", {})),
            "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
        }
        if "stats" in ckpt:
            meta["stats"] = ckpt["stats"]
        return meta
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def find_best_and_latest(run_dir: str | Path) -> dict[str, Path | None]:
    """Return paths to best.pt and latest.pt if they exist."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    result = {
        "best": ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else None,
        "latest": ckpt_dir / "latest.pt" if (ckpt_dir / "latest.pt").exists() else None,
    }
    return result


def cleanup_old_checkpoints(
    run_dir: str | Path,
    keep_last: int = 3,
    keep_best: bool = True,
    keep_latest: bool = True,
    dry_run: bool = True,
) -> list[Path]:
    """
    Remove old numbered checkpoints, keeping the N most recent plus best/latest.

    Returns list of files that would be / were removed.
    """
    ckpts = list_checkpoints(run_dir)
    keep = set()
    ckpt_dir = Path(run_dir) / "checkpoints"

    if keep_best and (ckpt_dir / "best.pt").exists():
        keep.add(ckpt_dir / "best.pt")
    if keep_latest and (ckpt_dir / "latest.pt").exists():
        keep.add(ckpt_dir / "latest.pt")

    # Keep the last N numbered checkpoints
    for p in ckpts[-keep_last:]:
        keep.add(p)

    to_remove = [p for p in ckpts if p not in keep]

    if not dry_run:
        for p in to_remove:
            p.unlink()

    return to_remove
