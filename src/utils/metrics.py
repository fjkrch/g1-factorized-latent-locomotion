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

    Args:
        run_dirs: List of run directories (one per seed).
        metric_file: Name of the JSON file containing eval metrics in each dir.

    Returns:
        Dict mapping metric name -> {mean, std, min, max, n_seeds}.
    """
    all_metrics: dict[str, list[float]] = defaultdict(list)

    for run_dir in run_dirs:
        path = Path(run_dir) / metric_file
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, (int, float)):
                all_metrics[k].append(float(v))

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
