"""
Plotting utilities for generating publication-quality figures.

Produces:
- Training curves (reward vs timestep, all methods)
- Evaluation bar plots (per task)
- Ablation bar plots
- Robustness sweep line plots
- Latent analysis scatter plots

All plots use matplotlib with consistent styling.

Usage:
    python scripts/plot_results.py --results_dir results/ --output_dir figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Plotting disabled.")


# Consistent style
COLORS = {
    "mlp": "#1f77b4",
    "lstm": "#ff7f0e",
    "transformer": "#2ca02c",
    "dynamite": "#d62728",
}
METHOD_LABELS = {
    "mlp": "PPO + MLP",
    "lstm": "PPO + LSTM",
    "transformer": "PPO + Transformer",
    "dynamite": "DynaMITE (Ours)",
}
TASK_LABELS = {
    "flat": "Flat",
    "push": "Push",
    "randomized": "Randomized",
    "terrain": "Terrain",
}


def set_style():
    """Set consistent matplotlib style."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (7, 4.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_training_curves(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    metric: str = "reward/mean",
    title: str = "Training Curves",
):
    """
    Plot training curves for multiple methods.

    Args:
        results: dict of method_name -> {"steps": [...], "values": [...], "std": [...]}
        output_path: Path to save figure.
        metric: Metric name for y-axis label.
        title: Plot title.
    """
    if not HAS_MPL:
        return
    set_style()
    fig, ax = plt.subplots()

    for method, data in results.items():
        steps = np.array(data["steps"])
        values = np.array(data["values"])
        color = COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)

        ax.plot(steps, values, color=color, label=label, linewidth=1.5)
        if "std" in data:
            std = np.array(data["std"])
            ax.fill_between(steps, values - std, values + std, alpha=0.15, color=color)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_eval_bars(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str | Path,
    metric: str = "reward_mean",
    title: str = "Evaluation Results",
):
    """
    Plot grouped bar chart: methods x tasks.

    Args:
        results: dict of task_name -> {method_name -> {"mean": ..., "std": ...}}
    """
    if not HAS_MPL:
        return
    set_style()

    tasks = list(results.keys())
    methods = list(next(iter(results.values())).keys())
    n_tasks = len(tasks)
    n_methods = len(methods)

    x = np.arange(n_tasks)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        means = [results[t][method].get("mean", 0) for t in tasks]
        stds = [results[t][method].get("std", 0) for t in tasks]
        color = COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        ax.bar(x + i * width, means, width, yerr=stds, label=label,
               color=color, capsize=3, alpha=0.85)

    ax.set_xlabel("Task")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_robustness_sweep(
    results: dict[str, dict[str, list[float]]],
    sweep_values: list[float],
    output_path: str | Path,
    xlabel: str = "Perturbation Magnitude",
    ylabel: str = "Reward",
    title: str = "Robustness Sweep",
):
    """
    Plot line chart for robustness sweep: performance vs perturbation level.

    Args:
        results: dict of method_name -> {"values": [...], "stds": [...]}
        sweep_values: x-axis values (perturbation magnitudes).
    """
    if not HAS_MPL:
        return
    set_style()
    fig, ax = plt.subplots()

    for method, data in results.items():
        vals = np.array(data["values"])
        color = COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        ax.plot(sweep_values, vals, "-o", color=color, label=label, linewidth=1.5, markersize=5)
        if "stds" in data:
            stds = np.array(data["stds"])
            ax.fill_between(sweep_values, vals - stds, vals + stds, alpha=0.15, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ablation_bars(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str = "Ablation Study",
):
    """
    Plot horizontal bar chart for ablation results.

    Args:
        results: dict of ablation_name -> {"mean": ..., "std": ...}
    """
    if not HAS_MPL:
        return
    set_style()
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.5)))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, capsize=3, color="#d62728", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Reward")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_latent_tsne(
    latent_z: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "Latent Space t-SNE",
):
    """
    Plot t-SNE of latent dynamics vectors colored by a dynamics parameter.

    Args:
        latent_z: (N, latent_dim) latent vectors.
        labels: (N,) continuous label for coloring (e.g., friction).
    """
    if not HAS_MPL:
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Warning: sklearn not available for t-SNE. Skipping.")
        return

    set_style()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(latent_z)

    fig, ax = plt.subplots()
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="viridis", s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Dynamics parameter")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")
