#!/usr/bin/env python3
"""
Plot robustness sweep results.

Reads sweep JSON files from results/sweeps/{model}/{sweep}/sweep_{name}.json
Produces one figure per sweep type with all methods overlaid.

Usage:
    python scripts/plot_sweeps.py --results_dir results/sweeps --output_dir figures
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required")
    sys.exit(1)


MODEL_LABELS = {
    "mlp": "MLP",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "dynamite": "DynaMITE",
}

MODEL_COLORS = {
    "mlp": "#1f77b4",
    "lstm": "#ff7f0e",
    "transformer": "#2ca02c",
    "dynamite": "#d62728",
}

MODEL_MARKERS = {
    "mlp": "o",
    "lstm": "s",
    "transformer": "^",
    "dynamite": "D",
}

SWEEP_TITLES = {
    "friction": "Robustness: Friction Coefficient",
    "action_delay": "Robustness: Actuation Delay",
    "push_magnitude": "Robustness: Push Magnitude",
}

SWEEP_XLABELS = {
    "friction": "Friction Coefficient",
    "action_delay": "Delay (steps)",
    "push_magnitude": "Push Velocity Range (m/s)",
}


def load_sweep_results(results_dir: Path) -> dict:
    """Load all sweep results into nested dict: sweep_name -> model -> results."""
    all_results = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for sweep_dir in sorted(model_dir.iterdir()):
            if not sweep_dir.is_dir():
                continue
            sweep_name = sweep_dir.name

            # Find sweep JSON
            json_files = list(sweep_dir.glob("sweep_*.json"))
            if not json_files:
                continue

            with open(json_files[0]) as f:
                data = json.load(f)

            if sweep_name not in all_results:
                all_results[sweep_name] = {}
            all_results[sweep_name][model_name] = data

    return all_results


def extract_x_values(sweep_data: dict) -> list:
    """Extract x-axis values from sweep results."""
    values = sweep_data.get("values", [])
    # For ranges like [0.3, 0.3], take the first element
    x = []
    for v in values:
        if isinstance(v, list):
            x.append(v[0] if len(v) == 1 else (v[0] + v[1]) / 2 if v[0] != v[1] else v[0])
        else:
            x.append(v)
    return x


def plot_sweep(sweep_name: str, model_results: dict, output_dir: Path):
    """Plot a single sweep with all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models_order = ["mlp", "lstm", "transformer", "dynamite"]

    for model_name in models_order:
        if model_name not in model_results:
            continue
        data = model_results[model_name]
        x = extract_x_values(data)
        results = data.get("results", [])

        y = []
        for r in results:
            reward = r.get("episode_reward/mean", r.get("reward_mean", 0))
            y.append(reward)

        if len(x) != len(y):
            print(f"  WARN: {model_name}/{sweep_name}: x={len(x)}, y={len(y)} mismatch")
            continue

        label = MODEL_LABELS.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, "gray")
        marker = MODEL_MARKERS.get(model_name, "o")

        ax.plot(x, y, marker=marker, label=label, color=color,
                linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel(SWEEP_XLABELS.get(sweep_name, sweep_name), fontsize=12)
    ax.set_ylabel("Eval Reward (higher is better)", fontsize=12)
    ax.set_title(SWEEP_TITLES.get(sweep_name, f"Sweep: {sweep_name}"), fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / f"sweep_{sweep_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_combined_sweeps(all_results: dict, output_dir: Path):
    """Plot all sweeps in a single row of panels."""
    sweep_order = ["friction", "action_delay", "push_magnitude"]
    available = [s for s in sweep_order if s in all_results]

    if not available:
        print("  No sweep results to plot")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    models_order = ["mlp", "lstm", "transformer", "dynamite"]

    for idx, sweep_name in enumerate(available):
        ax = axes[idx]
        model_results = all_results[sweep_name]

        for model_name in models_order:
            if model_name not in model_results:
                continue
            data = model_results[model_name]
            x = extract_x_values(data)
            results = data.get("results", [])
            y = [r.get("episode_reward/mean", r.get("reward_mean", 0)) for r in results]

            if len(x) != len(y):
                continue

            label = MODEL_LABELS.get(model_name, model_name)
            color = MODEL_COLORS.get(model_name, "gray")
            marker = MODEL_MARKERS.get(model_name, "o")
            ax.plot(x, y, marker=marker, label=label, color=color,
                    linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=0.5)

        ax.set_xlabel(SWEEP_XLABELS.get(sweep_name, sweep_name), fontsize=11)
        ax.set_ylabel("Eval Reward" if idx == 0 else "", fontsize=11)
        ax.set_title(SWEEP_TITLES.get(sweep_name, sweep_name), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == len(available) - 1:
            ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out_path = output_dir / "sweep_robustness_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def print_robustness_table(all_results: dict):
    """Print a markdown table of robustness gaps at extreme OOD points."""
    print("\n  === Robustness Gap at Most Extreme OOD Point ===\n")
    sweep_order = ["friction", "action_delay", "push_magnitude"]
    models_order = ["mlp", "lstm", "transformer", "dynamite"]

    for sweep_name in sweep_order:
        if sweep_name not in all_results:
            continue
        model_results = all_results[sweep_name]
        print(f"  {sweep_name}:")

        # Get reward at last (most extreme) sweep point
        rewards = {}
        for model_name in models_order:
            if model_name in model_results:
                results = model_results[model_name].get("results", [])
                if results:
                    last = results[-1]
                    rewards[model_name] = last.get("episode_reward/mean", last.get("reward_mean", 0))

        if "dynamite" in rewards:
            for model_name in models_order:
                if model_name in rewards:
                    delta = rewards[model_name] - rewards.get("dynamite", 0)
                    label = MODEL_LABELS.get(model_name, model_name)
                    print(f"    {label:>12s}: {rewards[model_name]:>8.2f}  (Δ vs DynaMITE: {delta:+.2f})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("--results_dir", default="results/sweeps")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found")
        sys.exit(1)

    print("[Sweep Plots] Loading results...")
    all_results = load_sweep_results(results_dir)
    print(f"  Found sweeps: {list(all_results.keys())}")

    for sweep_name, model_results in all_results.items():
        print(f"\n  Plotting {sweep_name} ({len(model_results)} models)...")
        plot_sweep(sweep_name, model_results, output_dir)

    # Combined plot
    print("\n  Plotting combined figure...")
    plot_combined_sweeps(all_results, output_dir)

    # Print robustness table
    print_robustness_table(all_results)

    print("\n[Sweep Plots] Done.")


if __name__ == "__main__":
    main()
