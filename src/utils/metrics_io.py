"""
Metrics I/O — standardized reading and writing of metrics files.

Handles three tiers:
  1. Step-level  : per-PPO-update row in metrics.csv
  2. Episode-level: per-eval-episode in eval_episodes.csv
  3. Summary-level: final aggregated JSON in eval_metrics.json / run_summary.json

Usage:
    from src.utils.metrics_io import (
        write_step_metrics, read_step_metrics,
        write_eval_episodes, read_eval_episodes,
        write_run_summary, read_run_summary,
    )
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# ── Step-level (training curve) ──────────────────────────────────────────────

STEP_CSV_COLUMNS = [
    "iteration",
    "global_step",
    "wall_time_s",
    "reward_mean",
    "reward_std",
    "episode_length_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "aux_loss",
    "learning_rate",
    "fps",
    "gpu_mem_mb",
]


def write_step_header(path: str | Path) -> None:
    """Write CSV header for step-level metrics."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(STEP_CSV_COLUMNS)


def append_step_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row of step-level metrics."""
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([row.get(c, "") for c in STEP_CSV_COLUMNS])


def read_step_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read entire step-level CSV as list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ── Episode-level (eval episodes) ────────────────────────────────────────────

EPISODE_CSV_COLUMNS = [
    "episode_idx",
    "reward",
    "length",
    "success",
    "fall",
    "lin_vel_error",
    "ang_vel_error",
    "max_torque",
]


def write_eval_episodes(
    path: str | Path,
    episodes: list[dict[str, Any]],
) -> None:
    """Write per-episode evaluation results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for ep in episodes:
            w.writerow(ep)


def read_eval_episodes(path: str | Path) -> list[dict[str, Any]]:
    """Read per-episode evaluation CSV."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            parsed = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v) if v != "" else None
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ── Summary-level ────────────────────────────────────────────────────────────

def write_run_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write final run summary (eval_metrics.json or run_summary.json)."""
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)


def read_run_summary(path: str | Path) -> dict[str, Any]:
    """Read run summary JSON."""
    with open(path, "r") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ── Discovery helpers ────────────────────────────────────────────────────────

def discover_run_dirs(
    base_dir: str | Path,
    task: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
) -> list[Path]:
    """
    Discover run directories under base_dir matching optional filters.

    Expected structure: base_dir/{task}/{model}_{variant}/seed_{seed}/{timestamp}/
    """
    base = Path(base_dir)
    if not base.exists():
        return []

    results = []
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        if task and task_dir.name != task:
            continue
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            if model and not method_dir.name.startswith(model):
                continue
            if variant and f"_{variant}" not in method_dir.name:
                continue
            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                if seed is not None and seed_dir.name != f"seed_{seed}":
                    continue
                for ts_dir in sorted(seed_dir.iterdir()):
                    if ts_dir.is_dir():
                        results.append(ts_dir)
    return results


def is_run_valid(run_dir: str | Path) -> tuple[bool, list[str]]:
    """
    Check if a run directory contains all mandatory files.

    Returns (is_valid, list_of_issues).
    Mandatory:
      - config.yaml
      - manifest.json
      - metrics.csv (or tb/ directory)
      - checkpoints/best.pt OR checkpoints/latest.pt
    """
    d = Path(run_dir)
    issues = []

    if not (d / "config.yaml").exists():
        issues.append("missing config.yaml")
    if not (d / "manifest.json").exists():
        issues.append("missing manifest.json")
    if not (d / "metrics.csv").exists() and not (d / "tb").is_dir():
        issues.append("missing metrics.csv and tb/")
    ckpt = d / "checkpoints"
    if not ckpt.is_dir():
        issues.append("missing checkpoints/")
    elif not (ckpt / "best.pt").exists() and not (ckpt / "latest.pt").exists():
        issues.append("no best.pt or latest.pt in checkpoints/")

    return (len(issues) == 0, issues)


def is_run_complete(run_dir: str | Path) -> bool:
    """Check if manifest status == 'completed'."""
    mpath = Path(run_dir) / "manifest.json"
    if not mpath.exists():
        return False
    try:
        with open(mpath) as f:
            m = json.load(f)
        return m.get("status") == "completed"
    except Exception:
        return False
