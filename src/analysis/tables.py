"""
Table generation utility for results.

Produces LaTeX and Markdown tables from aggregated experiment results.

Usage:
    python scripts/generate_tables.py --results_dir results/ --output_dir figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_main_table(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str | Path,
    fmt: str = "markdown",
):
    """
    Generate main comparison table: methods (rows) x tasks (columns).

    Args:
        results: {task -> {method -> {"mean": ..., "std": ...}}}
        output_path: Output file path.
        fmt: "markdown" or "latex"
    """
    tasks = list(results.keys())
    methods = list(next(iter(results.values())).keys())

    if fmt == "markdown":
        lines = _gen_markdown_table(results, tasks, methods)
    else:
        lines = _gen_latex_table(results, tasks, methods)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved table: {output_path}")


def _gen_markdown_table(
    results: dict, tasks: list[str], methods: list[str]
) -> list[str]:
    """Generate Markdown table."""
    header = "| Method | " + " | ".join(tasks) + " |"
    sep = "|" + "|".join(["---"] * (len(tasks) + 1)) + "|"
    lines = [header, sep]

    for method in methods:
        row = f"| {method} |"
        for task in tasks:
            m = results[task].get(method, {})
            mean = m.get("mean", 0)
            std = m.get("std", 0)
            row += f" {mean:.1f} ± {std:.1f} |"
        lines.append(row)

    return lines


def _gen_latex_table(
    results: dict, tasks: list[str], methods: list[str]
) -> list[str]:
    """Generate LaTeX table."""
    ncols = len(tasks)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Main Results: Mean Reward (±std) across tasks.}",
        "\\label{tab:main}",
        "\\begin{tabular}{l" + "c" * ncols + "}",
        "\\toprule",
        "Method & " + " & ".join(tasks) + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        row = f"{method}"
        for task in tasks:
            m = results[task].get(method, {})
            mean = m.get("mean", 0)
            std = m.get("std", 0)
            row += f" & {mean:.1f} $\\pm$ {std:.1f}"
        row += " \\\\"
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return lines


def generate_ablation_table(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
    fmt: str = "markdown",
):
    """
    Generate ablation table.

    Args:
        results: {ablation_name -> {"mean": ..., "std": ...}}
    """
    if fmt == "markdown":
        header = "| Variant | Reward |"
        sep = "|---|---|"
        lines = [header, sep]
        for name, data in results.items():
            lines.append(f"| {name} | {data['mean']:.1f} ± {data['std']:.1f} |")
    else:
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Ablation Study}",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Variant & Reward \\\\",
            "\\midrule",
        ]
        for name, data in results.items():
            lines.append(f"{name} & {data['mean']:.1f} $\\pm$ {data['std']:.1f} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved table: {output_path}")


def generate_efficiency_table(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
):
    """
    Generate efficiency comparison table.

    Args:
        results: {method -> {"params": ..., "throughput": ..., "gpu_mem_mb": ..., "wall_time_h": ...}}
    """
    header = "| Method | Params | Throughput (fps) | GPU Mem (MB) | Wall Time (h) |"
    sep = "|---|---|---|---|---|"
    lines = [header, sep]

    for method, data in results.items():
        lines.append(
            f"| {method} | {data['params']:,} | {data.get('throughput', 'N/A')} | "
            f"{data.get('gpu_mem_mb', 'N/A')} | {data.get('wall_time_h', 'N/A')} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved table: {output_path}")
