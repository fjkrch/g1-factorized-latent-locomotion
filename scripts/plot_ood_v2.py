#!/usr/bin/env python3
"""
Plot OOD sweep results with full behavioral metrics.

Generates:
  - figures/ood_reward_*.png     : reward vs perturbation (4 models, std bands)
  - figures/ood_failure_*.png    : failure rate vs perturbation
  - figures/ood_tracking_*.png   : tracking error vs perturbation
  - figures/ood_heatmap.png      : summary heatmap across all sweeps

Usage:
    python scripts/plot_ood_v2.py [--results-dir results/ood_v2]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS = ["dynamite", "lstm", "transformer", "mlp"]
MODEL_LABELS = {
    "dynamite": "DynaMITE (Ours)",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "mlp": "MLP",
}
MODEL_COLORS = {
    "dynamite": "#1f77b4",
    "lstm": "#ff7f0e",
    "transformer": "#2ca02c",
    "mlp": "#d62728",
}
SEEDS = [42, 43, 44, 45, 46]


def load_sweep_results(results_dir, sweep_name, task, model, seeds):
    """Load sweep results for a model across seeds."""
    all_data = []
    for seed in seeds:
        path = results_dir / sweep_name / task / f"{model}_seed{seed}" / f"sweep_{sweep_name}.json"
        if path.exists():
            with open(path) as f:
                all_data.append(json.load(f))
    return all_data


def extract_metric(data_list, metric_key):
    """(n_seeds, n_levels) array for a given metric."""
    rows = []
    for d in data_list:
        vals = [float(r.get(metric_key, float("nan"))) for r in d["results"]]
        rows.append(vals)
    return np.array(rows)


def get_x_vals(data):
    """Get numeric x values from sweep."""
    values = data["results"][0]["value"] if isinstance(data["results"][0]["value"], list) else data["values"]
    x = []
    for v in data["values"]:
        if isinstance(v, list):
            x.append(v[0] if v[0] == v[1] else (v[0] + v[1]) / 2)
        else:
            x.append(float(v))
    return x


def plot_metric_sweep(results_dir, sweep_name, task, metric_key, ylabel, title_suffix,
                      fig_path, x_label=None):
    """Plot one metric for one sweep across all models."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_vals = None
    for model in MODELS:
        data_list = load_sweep_results(results_dir, sweep_name, task, model, SEEDS)
        if not data_list:
            continue
        if x_vals is None:
            x_vals = get_x_vals(data_list[0])
        arr = extract_metric(data_list, metric_key)
        if arr.size == 0:
            continue
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

        color = MODEL_COLORS[model]
        ax.plot(x_vals, mean, "o-", color=color, label=MODEL_LABELS[model], linewidth=2, markersize=5)
        ax.fill_between(x_vals, mean - std, mean + std, color=color, alpha=0.15)

    if x_vals is None:
        plt.close()
        return

    ax.set_xlabel(x_label or sweep_name.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{sweep_name.replace('_', ' ').title()} — {title_suffix}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def plot_summary_heatmap(results_dir, tasks_sweeps, fig_path):
    """Plot a summary heatmap: models × sweeps, colored by severe-level reward."""
    sweep_names = []
    model_severe = {m: [] for m in MODELS}

    for task, sweeps in tasks_sweeps:
        for sweep in sweeps:
            label = f"{sweep}\n({task})"
            has_data = False
            for model in MODELS:
                data_list = load_sweep_results(results_dir, sweep, task, model, SEEDS)
                if not data_list:
                    model_severe[model].append(float("nan"))
                    continue
                has_data = True
                arr = extract_metric(data_list, "episode_reward/mean")
                if arr.size == 0:
                    model_severe[model].append(float("nan"))
                    continue
                mean_per_level = np.nanmean(arr, axis=0)
                n_severe = min(2, len(mean_per_level))
                severe = np.sort(mean_per_level)[:n_severe].mean()
                model_severe[model].append(severe)
            if has_data:
                sweep_names.append(label)

    if not sweep_names:
        return

    matrix = np.array([model_severe[m][:len(sweep_names)] for m in MODELS])

    fig, ax = plt.subplots(figsize=(max(8, len(sweep_names) * 1.8), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(sweep_names)))
    ax.set_xticklabels(sweep_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)

    for i in range(len(MODELS)):
        for j in range(len(sweep_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if val < np.nanmedian(matrix) else "black")

    plt.colorbar(im, ax=ax, label="Severe-Level Reward", shrink=0.8)
    ax.set_title("OOD Robustness Summary — Severe-Level Mean Reward", fontsize=12)
    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/ood_v2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path("figures")

    # Define what to plot
    sweeps_config = [
        # (task, sweep, x_label)
        ("randomized", "friction", "Friction Coefficient"),
        ("randomized", "push_magnitude", "Push Velocity (m/s)"),
        ("randomized", "action_delay", "Action Delay (steps)"),
        ("randomized", "action_delay_unseen", "Action Delay (steps) — Unseen Range"),
        ("randomized", "combined_shift", "Combined Shift Severity Level"),
        ("push", "push_magnitude", "Push Velocity (m/s) — Push Task"),
        ("terrain", "push_magnitude", "Push Velocity (m/s) — Terrain Task"),
    ]

    metrics = [
        ("episode_reward/mean", "Reward", "Reward"),
        ("failure_rate", "Failure Rate", "Failure Rate"),
        ("tracking_error/mean", "Tracking Error (m/s)", "Tracking Error"),
        ("completion_rate", "Completion Rate", "Completion Rate"),
    ]

    for task, sweep, x_label in sweeps_config:
        for metric_key, ylabel, title_suffix in metrics:
            safe_name = metric_key.replace("/", "_")
            fig_path = fig_dir / f"ood_{safe_name}_{sweep}_{task}.png"
            plot_metric_sweep(results_dir, sweep, task, metric_key, ylabel,
                            title_suffix, fig_path, x_label)

    # Summary heatmap
    tasks_sweeps = [
        ("randomized", ["friction", "push_magnitude", "action_delay",
                        "action_delay_unseen", "combined_shift"]),
        ("push", ["push_magnitude"]),
        ("terrain", ["push_magnitude"]),
    ]
    plot_summary_heatmap(results_dir, tasks_sweeps, fig_dir / "ood_heatmap.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
