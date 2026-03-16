#!/usr/bin/env python3
"""
Plot learning curves from training metrics.csv files.

Produces publication-quality training curve plots showing reward vs. timesteps
for all methods, with per-seed shading.

Usage:
    python src/analysis/plot_learning_curves.py --base-dir outputs/ --output figures/
    python src/analysis/plot_learning_curves.py --base-dir outputs/ --task randomized --output figures/
"""

from __future__ import annotations

import argparse
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

from src.utils.metrics_io import discover_run_dirs, read_step_metrics

# ── Style ──
COLORS = {
    "mlp": "#1f77b4",
    "lstm": "#ff7f0e",
    "transformer": "#2ca02c",
    "dynamite": "#d62728",
}
LABELS = {
    "mlp": "MLP",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "dynamite": "DynaMITE (ours)",
}


def extract_curves(
    base_dir: str | Path,
    task: str,
    models: list[str],
    variant: str = "full",
    metric_key: str = "reward_mean",
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract training curves from metrics.csv files.

    Returns: {model: {"steps": [...], "seeds": array (n_seeds, n_steps)}}
    """
    base = Path(base_dir)
    curves = {}

    for model in models:
        method_dir = base / task / f"{model}_{variant}"
        if not method_dir.exists():
            continue

        seed_data = []
        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            ts_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
            if not ts_dirs:
                continue
            run_dir = ts_dirs[-1]
            csv_path = run_dir / "metrics.csv"
            if not csv_path.exists():
                continue

            rows = read_step_metrics(csv_path)
            if not rows:
                continue

            steps = [r.get("global_step", i) for i, r in enumerate(rows)]
            vals = [r.get(metric_key, 0) or 0 for r in rows]
            seed_data.append((steps, vals))

        if seed_data:
            # Align to shortest curve
            min_len = min(len(s) for s, _ in seed_data)
            steps_arr = np.array(seed_data[0][0][:min_len])
            seeds_arr = np.array([v[:min_len] for _, v in seed_data])
            curves[model] = {"steps": steps_arr, "seeds": seeds_arr}

    return curves


def plot_learning_curves(
    curves: dict[str, dict[str, np.ndarray]],
    title: str = "Training Curves",
    xlabel: str = "Environment Steps",
    ylabel: str = "Mean Episodic Reward",
    output_path: str | Path = "figures/fig_learning_curves.png",
    smoothing: int = 5,
) -> None:
    """Plot training curves with mean ± std shading."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model, data in curves.items():
        steps = data["steps"]
        seeds = data["seeds"]  # shape: (n_seeds, n_steps)

        # Smooth
        if smoothing > 1:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.array([np.convolve(s, kernel, mode="valid") for s in seeds])
            steps = steps[:smoothed.shape[1]]
        else:
            smoothed = seeds

        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        color = COLORS.get(model, "#333333")
        label = LABELS.get(model, model)

        ax.plot(steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves")
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--task", type=str, default="randomized")
    parser.add_argument("--output", type=str, default="figures")
    parser.add_argument("--smoothing", type=int, default=5)
    args = parser.parse_args()

    models = ["mlp", "lstm", "transformer", "dynamite"]
    curves = extract_curves(args.base_dir, args.task, models)

    if not curves:
        print(f"No training data found for task={args.task}")
        return

    output_path = Path(args.output) / f"fig_learning_curves_{args.task}.png"
    plot_learning_curves(
        curves,
        title=f"Training Curves — {args.task.capitalize()} Task",
        output_path=output_path,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()
