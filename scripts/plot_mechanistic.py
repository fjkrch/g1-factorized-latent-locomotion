#!/usr/bin/env python3
"""
Generate all mechanistic analysis figures.

Reads JSON results from results/mechanistic/ and produces:
  1. figures/gradient_norms.png
  2. figures/cosine_similarity.png
  3. figures/geometry_comparison.png
  4. figures/mi_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Match project style ──────────────────────────────────────────────────
COLORS = {
    "mlp":         "#1f77b4",
    "lstm":        "#ff7f0e",
    "transformer": "#2ca02c",
    "dynamite":    "#d62728",
}
MODEL_LABELS = {
    "dynamite": "DynaMITE (Ours)",
    "lstm":     "PPO + LSTM",
}
FACTOR_COLORS = {
    "ppo":         "#1f77b4",
    "aux_friction": "#ff7f0e",
    "aux_mass":     "#2ca02c",
    "aux_motor":    "#d62728",
    "aux_contact":  "#9467bd",
    "aux_delay":    "#8c564b",
    "total":        "#333333",
}
FACTOR_LABELS = {
    "ppo":         "PPO (policy+value−entropy)",
    "aux_friction": "Aux: friction",
    "aux_mass":     "Aux: mass",
    "aux_motor":    "Aux: motor",
    "aux_contact":  "Aux: contact",
    "aux_delay":    "Aux: delay",
    "total":        "Total gradient",
}

def set_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Gradient Norms Over Training
# ═══════════════════════════════════════════════════════════════════════════

def plot_gradient_norms(results_dir, output_path):
    """Plot gradient norms per loss component over training steps."""
    grad_dir = Path(results_dir) / "gradient_analysis"
    files = sorted(grad_dir.glob("seed_*.json"))
    if not files:
        print("  [SKIP] No gradient analysis files found")
        return

    # Parse all seeds
    all_data = []
    for f in files:
        with open(f) as fh:
            all_data.append(json.load(fh))

    # Collect all component names
    all_components = set()
    for seed_data in all_data:
        for entry in seed_data:
            all_components.update(entry["norms"].keys())

    # Build time series per component
    # Use global_step as x-axis (common across seeds)
    component_series = {}
    for comp in sorted(all_components):
        seed_series = []
        for seed_data in all_data:
            steps = [e["global_step"] for e in seed_data if comp in e["norms"]]
            vals = [e["norms"][comp] for e in seed_data if comp in e["norms"]]
            if steps:
                seed_series.append((steps, vals))
        component_series[comp] = seed_series

    fig, ax = plt.subplots(figsize=(9, 5))

    for comp, series in component_series.items():
        if not series:
            continue
        # Align to common steps (use first seed's steps)
        ref_steps = series[0][0]
        n_steps = min(len(s[0]) for s in series)
        common_steps = ref_steps[:n_steps]

        vals_matrix = np.array([s[1][:n_steps] for s in series])
        mean = vals_matrix.mean(axis=0)
        std = vals_matrix.std(axis=0)

        color = FACTOR_COLORS.get(comp, "#777777")
        label = FACTOR_LABELS.get(comp, comp)
        lw = 2.0 if comp in ("ppo", "total") else 1.2
        ls = "--" if comp == "total" else "-"

        ax.plot(np.array(common_steps) / 1e6, mean, color=color, label=label,
                linewidth=lw, linestyle=ls)
        ax.fill_between(np.array(common_steps) / 1e6, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Gradient L2 norm")
    ax.set_title("Per-Component Gradient Norms During DynaMITE Training")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Cosine Similarity (aux vs PPO gradients)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_similarity(results_dir, output_path):
    """Plot cosine similarity between PPO and each aux factor gradient."""
    grad_dir = Path(results_dir) / "gradient_analysis"
    files = sorted(grad_dir.glob("seed_*.json"))
    if not files:
        print("  [SKIP] No gradient analysis files found")
        return

    all_data = []
    for f in files:
        with open(f) as fh:
            all_data.append(json.load(fh))

    # All aux components
    aux_components = set()
    for seed_data in all_data:
        for entry in seed_data:
            aux_components.update(entry.get("cosines", {}).keys())

    fig, ax = plt.subplots(figsize=(9, 5))

    for comp in sorted(aux_components):
        seed_series = []
        for seed_data in all_data:
            steps = [e["global_step"] for e in seed_data if comp in e.get("cosines", {})]
            vals = [e["cosines"][comp] for e in seed_data if comp in e.get("cosines", {})]
            if steps:
                seed_series.append((steps, vals))

        if not seed_series:
            continue

        n_steps = min(len(s[0]) for s in seed_series)
        common_steps = seed_series[0][0][:n_steps]
        vals_matrix = np.array([s[1][:n_steps] for s in seed_series])
        mean = vals_matrix.mean(axis=0)
        std = vals_matrix.std(axis=0)

        color = FACTOR_COLORS.get(comp, "#777777")
        label = FACTOR_LABELS.get(comp, comp)

        ax.plot(np.array(common_steps) / 1e6, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(np.array(common_steps) / 1e6, mean - std, mean + std,
                        alpha=0.15, color=color)

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Cosine similarity with PPO gradient")
    ax.set_title("Gradient Alignment: Auxiliary Losses vs PPO Objective")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Geometry Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_geometry_comparison(results_dir, output_path):
    """Plot side-by-side representation geometry metrics."""
    geo_file = Path(results_dir) / "geometry_analysis" / "results.json"
    if not geo_file.exists():
        print("  [SKIP] No geometry results file found")
        return

    with open(geo_file) as f:
        data = json.load(f)

    models = sorted(data.keys())
    if not models:
        print("  [SKIP] No model data in geometry results")
        return

    metrics = [
        ("effective_rank", "Effective Rank"),
        ("condition_number", "Condition Number"),
        ("participation_ratio", "Participation Ratio"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Bar plots for 3 scalar metrics
    for i, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(models))
        means = [data[m][metric_key]["mean"] for m in models]
        stds = [data[m][metric_key]["std"] for m in models]
        colors = [COLORS.get(m, "#777") for m in models]
        labels = [MODEL_LABELS.get(m, m) for m in models]

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(axis="y", alpha=0.3)

        if metric_key == "condition_number":
            ax.set_yscale("log")

    # PCA cumulative variance curve
    ax = axes[3]
    for m in models:
        curves = data[m].get("cumulative_variance_curves", [])
        if not curves:
            continue
        max_d = min(len(c) for c in curves)
        curves_arr = np.array([c[:max_d] for c in curves])
        mean_curve = curves_arr.mean(axis=0)
        std_curve = curves_arr.std(axis=0)
        dims = np.arange(1, max_d + 1)
        color = COLORS.get(m, "#777")
        label = MODEL_LABELS.get(m, m)
        ax.plot(dims, mean_curve, color=color, label=label, linewidth=1.5)
        ax.fill_between(dims, mean_curve - std_curve, np.minimum(mean_curve + std_curve, 1.0),
                        alpha=0.15, color=color)
    ax.axhline(y=0.95, color="gray", linestyle=":", linewidth=0.8, label="95% var")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Variance Curve")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Representation Geometry: DynaMITE Latent vs LSTM Hidden State", fontsize=14, y=1.02)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Mutual Information Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_mi_comparison(results_dir, output_path):
    """Plot MI estimates per factor, grouped by model."""
    mi_file = Path(results_dir) / "mi_analysis" / "mine_results.json"
    if not mi_file.exists():
        print("  [SKIP] No MINE results file found")
        return

    with open(mi_file) as f:
        data = json.load(f)

    models = sorted(data.keys())
    if not models:
        print("  [SKIP] No model data in MINE results")
        return

    # Collect all factors
    all_factors = set()
    for m in models:
        all_factors.update(data[m].keys())
    factor_order = ["friction", "mass", "motor", "contact", "delay", "overall"]
    factors = [f for f in factor_order if f in all_factors]

    fig, ax = plt.subplots(figsize=(10, 5))

    n_factors = len(factors)
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(n_factors)

    for i, m in enumerate(models):
        means = []
        stds = []
        for f in factors:
            if f in data[m]:
                means.append(data[m][f]["mean"])
                stds.append(data[m][f]["std"])
            else:
                means.append(0)
                stds.append(0)

        color = COLORS.get(m, "#777")
        label = MODEL_LABELS.get(m, m)
        offset = (i - n_models / 2 + 0.5) * bar_width

        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
               color=color, alpha=0.8, label=label,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in factors])
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("Mutual Information Between Representations and Dynamics Parameters")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot Mechanistic Analysis Results")
    parser.add_argument("--results_dir", type=str, default="results/mechanistic")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    set_style()
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[PlotMechanistic] Generating figures...")

    print("\n1. Gradient norms...")
    plot_gradient_norms(args.results_dir, fig_dir / "gradient_norms.png")

    print("\n2. Cosine similarity...")
    plot_cosine_similarity(args.results_dir, fig_dir / "cosine_similarity.png")

    print("\n3. Geometry comparison...")
    plot_geometry_comparison(args.results_dir, fig_dir / "geometry_comparison.png")

    print("\n4. MI comparison...")
    plot_mi_comparison(args.results_dir, fig_dir / "mi_comparison.png")

    print("\n[PlotMechanistic] Done!")


if __name__ == "__main__":
    main()
