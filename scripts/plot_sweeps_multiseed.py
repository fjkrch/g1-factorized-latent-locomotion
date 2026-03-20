#!/usr/bin/env python3
"""
Regenerate sweep_robustness_combined.png from multi-seed data.

Uses results/sweeps_multiseed/ (3 seeds × 2 models × 3 sweep types)
to match README's OOD sweep tables exactly.
"""

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


MODEL_LABELS = {"dynamite": "DynaMITE (Ours)", "lstm": "LSTM"}
MODEL_COLORS = {"dynamite": "#d62728", "lstm": "#ff7f0e"}
MODEL_MARKERS = {"dynamite": "D", "lstm": "s"}

SWEEP_TITLES = {
    "friction": "Friction Coefficient",
    "action_delay": "Actuation Delay (steps)",
    "push_magnitude": "Push Magnitude (m/s)",
}


def extract_x(values):
    """Convert sweep values (may be ranges) to x-axis values."""
    x = []
    for v in values:
        if isinstance(v, list):
            x.append(v[0] if v[0] == v[1] else (v[0] + v[1]) / 2)
        else:
            x.append(v)
    return x


def main():
    base_dir = Path("results/sweeps_multiseed")
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    sweep_order = ["friction", "action_delay", "push_magnitude"]
    models = ["dynamite", "lstm"]
    seeds = [42, 43, 44]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, sweep_name in enumerate(sweep_order):
        ax = axes[idx]

        for model in models:
            all_means = []
            x_vals = None
            for seed in seeds:
                path = base_dir / sweep_name / f"{model}_seed{seed}" / f"sweep_{sweep_name}.json"
                if not path.exists():
                    print(f"  MISSING: {path}")
                    continue
                with open(path) as f:
                    data = json.load(f)
                means = [r["episode_reward/mean"] for r in data["results"]]
                all_means.append(means)
                if x_vals is None:
                    x_vals = extract_x(data["values"])

            if not all_means:
                continue

            arr = np.array(all_means)
            avg = np.mean(arr, axis=0)
            std = np.std(arr, axis=0, ddof=1)

            label = MODEL_LABELS[model]
            color = MODEL_COLORS[model]
            marker = MODEL_MARKERS[model]

            ax.plot(x_vals, avg, marker=marker, label=label, color=color,
                    linewidth=2, markersize=7, markeredgecolor="white",
                    markeredgewidth=0.5)
            ax.fill_between(x_vals, avg - std, avg + std, alpha=0.15, color=color)

        ax.set_xlabel(SWEEP_TITLES[sweep_name], fontsize=12)
        ax.set_ylabel("Eval Reward" if idx == 0 else "", fontsize=12)
        ax.set_title(f"OOD: {SWEEP_TITLES[sweep_name]}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 2:
            ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    out_path = output_dir / "sweep_robustness_combined.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
