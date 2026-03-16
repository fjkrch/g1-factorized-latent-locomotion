#!/usr/bin/env python3
"""
Generate result tables (Markdown and LaTeX).

Usage:
    python scripts/generate_tables.py --results_dir results/aggregated --output_dir figures/

Inputs:
    - results/aggregated/main_comparison.json
    - results/aggregated/ablation_results.json

Outputs:
    - figures/main_table.md
    - figures/main_table.tex
    - figures/ablation_table.md
    - figures/ablation_table.tex
    - figures/efficiency_table.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.tables import generate_main_table, generate_ablation_table, generate_efficiency_table


def parse_args():
    parser = argparse.ArgumentParser(description="Generate result tables")
    parser.add_argument("--results_dir", type=str, default="results/aggregated")
    parser.add_argument("--output_dir", type=str, default="figures/")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main comparison table
    mc_path = results_dir / "main_comparison.json"
    if mc_path.exists():
        with open(mc_path) as f:
            data = json.load(f)
        generate_main_table(data, output_dir / "main_table.md", fmt="markdown")
        generate_main_table(data, output_dir / "main_table.tex", fmt="latex")
    else:
        print(f"Skipping main table: {mc_path} not found")

    # Ablation table
    ab_path = results_dir / "ablation_results.json"
    if ab_path.exists():
        with open(ab_path) as f:
            data = json.load(f)
        generate_ablation_table(data, output_dir / "ablation_table.md", fmt="markdown")
        generate_ablation_table(data, output_dir / "ablation_table.tex", fmt="latex")
    else:
        print(f"Skipping ablation table: {ab_path} not found")

    # Efficiency table
    eff_path = results_dir / "efficiency.json"
    if eff_path.exists():
        with open(eff_path) as f:
            data = json.load(f)
        generate_efficiency_table(data, output_dir / "efficiency_table.md")
    else:
        print(f"Skipping efficiency table: {eff_path} not found")

    print(f"[Tables] Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
