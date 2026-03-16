#!/usr/bin/env python3
"""
Plot generation script.

Usage:
    # Generate all plots from results directory
    python scripts/plot_results.py --results_dir results/ --output_dir figures/

    # Generate specific plot type
    python scripts/plot_results.py --results_dir results/ --output_dir figures/ --plot training_curves
    python scripts/plot_results.py --results_dir results/ --output_dir figures/ --plot eval_bars
    python scripts/plot_results.py --results_dir results/ --output_dir figures/ --plot robustness

Inputs:
    - results/ directory containing aggregated JSON result files
    - Expected files:
        results/aggregated/main_comparison.json
        results/aggregated/ablation_results.json
        results/aggregated/sweep_*.json
        results/aggregated/training_curves.json

Outputs:
    - figures/training_curves.png
    - figures/eval_bars.png
    - figures/ablation.png
    - figures/sweep_push_magnitude.png
    - figures/sweep_friction.png
    - etc.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.plotting import (
    plot_training_curves,
    plot_eval_bars,
    plot_robustness_sweep,
    plot_ablation_bars,
    plot_latent_tsne,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--results_dir", type=str, default="results/aggregated",
                        help="Directory containing aggregated results")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Output directory for figures")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["all", "training_curves", "eval_bars", "ablation", "robustness", "latent"],
                        help="Which plot to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    if args.plot in ["all", "training_curves"]:
        tc_path = results_dir / "training_curves.json"
        if tc_path.exists():
            with open(tc_path) as f:
                data = json.load(f)
            plot_training_curves(data, output_dir / "training_curves.png")
        else:
            print(f"Skipping training curves: {tc_path} not found")

    # Evaluation bars
    if args.plot in ["all", "eval_bars"]:
        mc_path = results_dir / "main_comparison.json"
        if mc_path.exists():
            with open(mc_path) as f:
                data = json.load(f)
            plot_eval_bars(data, output_dir / "eval_bars.png")
        else:
            print(f"Skipping eval bars: {mc_path} not found")

    # Ablation
    if args.plot in ["all", "ablation"]:
        ab_path = results_dir / "ablation_results.json"
        if ab_path.exists():
            with open(ab_path) as f:
                data = json.load(f)
            plot_ablation_bars(data, output_dir / "ablation.png")
        else:
            print(f"Skipping ablation: {ab_path} not found")

    # Robustness sweeps
    if args.plot in ["all", "robustness"]:
        for sweep_file in sorted(results_dir.glob("sweep_*.json")):
            sweep_name = sweep_file.stem
            with open(sweep_file) as f:
                data = json.load(f)
            if "sweep_values" in data and "methods" in data:
                plot_robustness_sweep(
                    data["methods"],
                    data["sweep_values"],
                    output_dir / f"{sweep_name}.png",
                    xlabel=data.get("xlabel", "Level"),
                    title=data.get("title", sweep_name),
                )

    # Latent analysis
    if args.plot in ["all", "latent"]:
        import numpy as np
        latent_path = results_dir / "latent_analysis.json"
        if latent_path.exists():
            with open(latent_path) as f:
                data = json.load(f)
            if "latent_z" in data:
                z = np.array(data["latent_z"])
                for param_key in [k for k in data.keys() if k.startswith("params_")]:
                    labels = np.array(data[param_key])
                    if labels.ndim > 1:
                        labels = labels[:, 0]
                    plot_latent_tsne(
                        z, labels,
                        output_dir / f"latent_tsne_{param_key}.png",
                        title=f"Latent t-SNE colored by {param_key}",
                    )

    print(f"[Plot] All requested plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
