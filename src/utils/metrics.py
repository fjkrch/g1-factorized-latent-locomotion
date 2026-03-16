"""
Metrics collection and aggregation.

Collects per-step, per-episode, and per-eval metrics.
Supports multi-seed aggregation.

Usage:
    from src.utils.metrics import MetricsTracker, aggregate_seeds
    tracker = MetricsTracker()
    tracker.update({"reward": 100.0, "episode_length": 500})
    summary = tracker.summarize()
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class MetricsTracker:
    """
    Track scalar metrics over training/evaluation.
    Accumulates values and provides summary statistics.
    """

    def __init__(self):
        self._data: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict[str, float]) -> None:
        """Add a set of metrics."""
        for k, v in metrics.items():
            self._data[k].append(float(v))

    def update_single(self, key: str, value: float) -> None:
        """Add a single metric."""
        self._data[key].append(float(value))

    def summarize(self, prefix: str = "") -> dict[str, float]:
        """
        Compute summary statistics for all tracked metrics.
        Returns dict with mean, std, min, max for each key.
        """
        summary = {}
        for k, vals in self._data.items():
            arr = np.array(vals)
            key = f"{prefix}{k}" if prefix else k
            summary[f"{key}/mean"] = float(arr.mean())
            summary[f"{key}/std"] = float(arr.std())
            summary[f"{key}/min"] = float(arr.min())
            summary[f"{key}/max"] = float(arr.max())
        return summary

    def get_mean(self, key: str) -> float:
        """Get mean of a single metric."""
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._data.clear()

    def save(self, path: str | Path) -> None:
        """Save raw metric data to JSON."""
        with open(path, "w") as f:
            json.dump({k: v for k, v in self._data.items()}, f, indent=2)


def aggregate_seeds(
    run_dirs: list[str | Path],
    metric_file: str = "eval_metrics.json",
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics across multiple seed runs.

    Falls back to extracting final metrics from metrics.csv when
    eval_metrics.json is not available.

    Args:
        run_dirs: List of run directories (one per seed).
        metric_file: Name of the JSON file containing eval metrics in each dir.

    Returns:
        Dict mapping metric name -> {mean, std, min, max, n_seeds}.
    """
    all_metrics: dict[str, list[float]] = defaultdict(list)

    for run_dir in run_dirs:
        path = Path(run_dir) / metric_file
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    all_metrics[k].append(float(v))
        else:
            # Fallback: extract final training metrics from metrics.csv
            csv_path = Path(run_dir) / "metrics.csv"
            if csv_path.exists():
                data = _extract_final_metrics_from_csv(csv_path)
                for k, v in data.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        all_metrics[k].append(float(v))
            else:
                print(f"Warning: neither {path} nor {csv_path} found, skipping.")

    result = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        result[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(vals),
        }
    return result


def _extract_final_metrics_from_csv(csv_path: Path) -> dict[str, float]:
    """Extract final training metrics from a metrics.csv file.

    Handles both standardized format (reward_mean) and Logger format (reward/mean).
    """
    import csv as csv_mod
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    # Column name mapping: Logger format -> standardized names
    col_map = {
        "reward/mean": "reward_mean", "reward_mean": "reward_mean",
        "reward/std": "reward_std", "reward_std": "reward_std",
        "loss/policy": "policy_loss", "policy_loss": "policy_loss",
        "loss/value": "value_loss", "value_loss": "value_loss",
        "loss/entropy": "entropy", "entropy": "entropy",
        "perf/fps": "fps", "fps": "fps",
        "global_step": "final_step",
    }

    # Find the last row with valid reward data
    reward_keys = ["reward_mean", "reward/mean"]
    for row in reversed(rows):
        for rk in reward_keys:
            val_str = row.get(rk, "")
            if val_str and val_str.strip():
                try:
                    val = float(val_str)
                    if not np.isnan(val):
                        result = {}
                        for src_col, dst_col in col_map.items():
                            if src_col in row and row[src_col]:
                                try:
                                    v = float(row[src_col])
                                    if not np.isnan(v):
                                        if dst_col == "final_step":
                                            result[dst_col] = int(v)
                                        else:
                                            result[dst_col] = v
                                except (ValueError, TypeError):
                                    pass
                        return result
                except (ValueError, TypeError):
                    continue
    return {}


def save_aggregated_results(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Save aggregated results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def load_eval_metrics(run_dir: str | Path) -> dict[str, float]:
    """Load evaluation metrics from a run directory."""
    path = Path(run_dir) / "eval_metrics.json"
    with open(path, "r") as f:
        return json.load(f)
