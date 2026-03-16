#!/usr/bin/env python3
"""
Plot robustness sweep results.

Produces line plots showing reward degradation as a perturbation parameter
is varied beyond the training distribution.

Usage:
    python src/analysis/plot_robustness.py --sweep-dir results/sweeps/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

COLORS = {
    "mlp_full": "#1f77b4",
    "lstm_full": "#ff7f0e",
    "transformer_full": "#2ca02c",
    "dynamite_full": "#d62728",
}
LABELS = {
    "mlp_full": "MLP",
    "lstm_full": "LSTM",
    "transformer_full": "Transformer",
    "dynamite_full": "DynaMITE (ours)",
}

SWEEP_LABELS = {
    "push_magnitude": "Push Force (N/kg)",
    "friction": "Friction Coefficient",
    "motor_strength": "Motor Strength Multiplier",
    "action_delay": "Action Delay (steps)",
}


def load_sweep_results(
    sweep_dir: str | Path,
    sweep_name: str,
) -> dict[str, dict[str, float]]:
    """
    Load sweep results for all methods.

    Returns: {method: {param_value_str: reward_mean}}
    """
    base = Path(sweep_dir)
    results = {}

    for method_dir in sorted(base.iterdir()):
        if not method_dir.is_dir():
            continue
        sweep_file = method_dir / sweep_name / f"sweep_{sweep_name}.json"
        if not sweep_file.exists():
            continue

        with open(sweep_file) as f:
            data = json.load(f)

        method_name = method_dir.name
        results[method_name] = {}
        for param_val, metrics in data.items():
            reward = metrics.get("episode_reward/mean", 0.0)
            results[method_name][param_val] = reward

    return results


def plot_robustness(
    results: dict[str, dict[str, float]],
    sweep_name: str,
    output_path: str | Path,
) -> None:
    """Plot robustness sweep for all methods."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for method, data in sorted(results.items()):
        if not data:
            continue

        # Sort by parameter value
        items = sorted(data.items(), key=lambda x: float(x[0]))
        x_vals = [float(k) for k, _ in items]
        y_vals = [v for _, v in items]

        color = COLORS.get(method, "#333333")
        label = LABELS.get(method, method)

        ax.plot(x_vals, y_vals, "o-", color=color, label=label, linewidth=2, markersize=6)

    xlabel = SWEEP_LABELS.get(sweep_name, sweep_name)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Mean Episodic Reward", fontsize=13)
    ax.set_title(f"Robustness: {xlabel}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot robustness sweeps")
    parser.add_argument("--sweep-dir", type=str, default="results/sweeps")
    parser.add_argument("--output", type=str, default="figures")
    args = parser.parse_args()

    sweeps = ["push_magnitude", "friction", "motor_strength", "action_delay"]

    for sweep_name in sweeps:
        results = load_sweep_results(args.sweep_dir, sweep_name)
        if not results:
            print(f"No data for sweep: {sweep_name}")
            continue
        output_path = Path(args.output) / f"fig_sweep_{sweep_name}.png"
        plot_robustness(results, sweep_name, output_path)


if __name__ == "__main__":
    main()
