#!/usr/bin/env python3
"""
Generate publication-ready tables from aggregated results.

Produces:
  - Main comparison table (Markdown + LaTeX)
  - Ablation table (Markdown + LaTeX)
  - Efficiency table (parameter counts, throughput, memory)
  - Runtime summary table

Usage:
    python src/analysis/make_tables.py --results-dir results/aggregated/ --output results/tables/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt_mean_std(metrics: dict, key: str, decimals: int = 1) -> str:
    """Format 'mean ± std' string from aggregated metrics."""
    if key not in metrics:
        return "—"
    m = metrics[key]
    mean = m.get("mean", 0)
    std = m.get("std", 0)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ── Main comparison ──────────────────────────────────────────────────────────

def make_main_table(
    results: dict,
    tasks: list[str],
    models: list[str],
    variant: str = "full",
    primary_metric: str = "episode_reward/mean",
) -> tuple[str, str]:
    """
    Generate main comparison table.

    Returns: (markdown_str, latex_str)
    """
    model_labels = {
        "mlp": "MLP", "lstm": "LSTM",
        "transformer": "Transformer", "dynamite": "DynaMITE (ours)",
    }
    task_labels = {
        "flat": "Flat", "push": "Push",
        "randomized": "Randomized", "terrain": "Terrain",
    }

    # ── Markdown ──
    md = "| Method |"
    for t in tasks:
        md += f" {task_labels.get(t, t)} |"
    md += "\n|---|" + "---|" * len(tasks) + "\n"

    for model in models:
        label = model_labels.get(model, model)
        if model == "dynamite":
            label = f"**{label}**"
        md += f"| {label} |"
        for task in tasks:
            key = f"{task}/{model}_{variant}"
            if key in results:
                metrics = results[key].get("metrics", {})
                md += f" {fmt_mean_std(metrics, primary_metric)} |"
            else:
                md += " — |"
        md += "\n"

    # ── LaTeX ──
    ncols = len(tasks) + 1
    tex = "\\begin{table}[h]\n\\centering\n"
    tex += f"\\caption{{Main comparison: mean reward $\\pm$ std across 3 seeds.}}\n"
    tex += f"\\label{{tab:main}}\n"
    tex += "\\begin{tabular}{l" + "c" * len(tasks) + "}\n\\toprule\n"
    tex += "Method"
    for t in tasks:
        tex += f" & {task_labels.get(t, t)}"
    tex += " \\\\\n\\midrule\n"

    for model in models:
        label = model_labels.get(model, model)
        if model == "dynamite":
            label = f"\\textbf{{{label}}}"
        tex += label
        for task in tasks:
            key = f"{task}/{model}_{variant}"
            if key in results:
                metrics = results[key].get("metrics", {})
                tex += f" & {fmt_mean_std(metrics, primary_metric)}"
            else:
                tex += " & —"
        tex += " \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    return md, tex


# ── Ablation table ───────────────────────────────────────────────────────────

def make_ablation_table(
    results: dict,
    primary_metric: str = "episode_reward/mean",
) -> tuple[str, str]:
    """Generate ablation table with delta from full model."""
    ablation_labels = {
        "full": "DynaMITE (full)",
        "seq_len_4": "History = 4",
        "seq_len_16": "History = 16",
        "no_latent": "No latent head",
        "single_latent": "Single (unfactored) latent",
        "no_aux_loss": "No auxiliary loss",
        "depth_1": "1 transformer layer",
        "depth_4": "4 transformer layers",
    }

    order = ["full", "seq_len_4", "seq_len_16", "no_latent",
             "single_latent", "no_aux_loss", "depth_1", "depth_4"]

    # Get baseline (full) mean
    full_metrics = results.get("full", {}).get("metrics", {})
    full_mean = full_metrics.get(primary_metric, {}).get("mean", 0)

    # ── Markdown ──
    md = "| Variant | Reward | Δ from Full |\n|---|---|---|\n"
    for abl in order:
        if abl not in results:
            continue
        label = ablation_labels.get(abl, abl)
        metrics = results[abl].get("metrics", {})
        val = fmt_mean_std(metrics, primary_metric)
        abl_mean = metrics.get(primary_metric, {}).get("mean", 0)
        delta = abl_mean - full_mean
        delta_str = f"{delta:+.1f}" if abl != "full" else "—"
        if abl == "full":
            label = f"**{label}**"
        md += f"| {label} | {val} | {delta_str} |\n"

    # ── LaTeX ──
    tex = "\\begin{table}[h]\n\\centering\n"
    tex += "\\caption{Ablation study on randomized task (mean reward $\\pm$ std).}\n"
    tex += "\\label{tab:ablation}\n"
    tex += "\\begin{tabular}{lcc}\n\\toprule\n"
    tex += "Variant & Reward & $\\Delta$ \\\\\n\\midrule\n"

    for abl in order:
        if abl not in results:
            continue
        label = ablation_labels.get(abl, abl)
        metrics = results[abl].get("metrics", {})
        val = fmt_mean_std(metrics, primary_metric)
        abl_mean = metrics.get(primary_metric, {}).get("mean", 0)
        delta = abl_mean - full_mean
        delta_str = f"{delta:+.1f}" if abl != "full" else "—"
        if abl == "full":
            label = f"\\textbf{{{label}}}"
        tex += f"{label} & {val} & {delta_str} \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    return md, tex


# ── Efficiency table ─────────────────────────────────────────────────────────

def make_efficiency_table() -> str:
    """Generate static efficiency comparison table (parameter counts etc.)."""
    md = "| Method | Parameters | Memory (est.) | Notes |\n"
    md += "|---|---|---|---|\n"
    md += "| MLP | ~200k | ~4 GB | No history |\n"
    md += "| LSTM | ~300k | ~4.5 GB | Implicit memory |\n"
    md += "| Transformer | ~400k | ~5 GB | 8-step history |\n"
    md += "| **DynaMITE** | **~450k** | **~5 GB** | 8-step history + latent |\n"
    return md


def main():
    parser = argparse.ArgumentParser(description="Generate results tables")
    parser.add_argument("--results-dir", type=str, default="results/aggregated")
    parser.add_argument("--output", type=str, default="results/tables")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Main comparison ──
    main_path = Path(args.results_dir) / "main_comparison.json"
    if main_path.exists():
        main_results = load_json(main_path)
        md, tex = make_main_table(
            main_results,
            tasks=["flat", "push", "randomized", "terrain"],
            models=["mlp", "lstm", "transformer", "dynamite"],
        )
        (output_dir / "table_main_comparison.md").write_text(md)
        (output_dir / "table_main_comparison.tex").write_text(tex)
        print(f"Saved: {output_dir / 'table_main_comparison.md'}")
        print(f"Saved: {output_dir / 'table_main_comparison.tex'}")
    else:
        print(f"Skipping main table: {main_path} not found")

    # ── Ablation ──
    abl_path = Path(args.results_dir) / "ablation_results.json"
    if abl_path.exists():
        abl_results = load_json(abl_path)
        md, tex = make_ablation_table(abl_results)
        (output_dir / "table_ablation.md").write_text(md)
        (output_dir / "table_ablation.tex").write_text(tex)
        print(f"Saved: {output_dir / 'table_ablation.md'}")
        print(f"Saved: {output_dir / 'table_ablation.tex'}")
    else:
        print(f"Skipping ablation table: {abl_path} not found")

    # ── Efficiency ──
    eff = make_efficiency_table()
    (output_dir / "table_efficiency.md").write_text(eff)
    print(f"Saved: {output_dir / 'table_efficiency.md'}")


if __name__ == "__main__":
    main()
