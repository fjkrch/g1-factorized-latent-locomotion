"""
Experiment manifest utility.

Records all metadata about a run for reproducibility:
- config, git hash, hardware, python/torch versions, timestamps,
  final metrics, etc.

Usage:
    from src.utils.manifest import create_manifest, save_manifest
    manifest = create_manifest(cfg, run_dir)
    save_manifest(manifest, run_dir)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_diff_stat() -> str:
    """Get git diff stat to detect uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        info["num_gpus"] = torch.cuda.device_count()
    return info


def create_manifest(
    cfg: dict,
    run_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a full experiment manifest with rich system/git metadata.

    Args:
        cfg: Merged config dict.
        run_dir: Output directory for this run.
        extra: Optional extra metadata.

    Returns:
        Manifest dict.
    """
    from src.utils.system_info import collect_system_info
    from src.utils.git_info import collect_git_info

    system_info = collect_system_info()
    git_info = collect_git_info()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "config": cfg,
        "git": git_info,
        "system": system_info,
        "hardware": get_hardware_info(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not_set"),
        },
        "status": "started",
        "training": {},
        "final_metrics": {},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def save_manifest(manifest: dict, run_dir: str | Path) -> Path:
    """Save manifest to run directory."""
    path = Path(run_dir) / "manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def update_manifest(run_dir: str | Path, updates: dict[str, Any]) -> None:
    """Update an existing manifest with new fields (e.g., final metrics)."""
    path = Path(run_dir) / "manifest.json"
    if path.exists():
        with open(path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {}
    manifest.update(updates)
    manifest["last_updated"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Load manifest from a run directory."""
    path = Path(run_dir) / "manifest.json"
    with open(path, "r") as f:
        return json.load(f)
