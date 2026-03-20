#!/usr/bin/env python3
"""
Generate updated README tables from OOD v2 results.

Reads results/ood_v2/ and produces markdown tables that can be dropped into README.md.
Run after the OOD campaign and analysis are complete.

Usage:
    python scripts/generate_ood_readme.py > /tmp/ood_readme_tables.md
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

MODELS = ["dynamite", "lstm", "transformer", "mlp"]
MODEL_LABELS = {
    "dynamite": "DynaMITE (Ours)",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "mlp": "MLP",
}
SEEDS = [42, 43, 44, 45, 46]


def load_sweep(base_dir, sweep_name, task, model, seeds):
    all_data = []
    for seed in seeds:
        path = base_dir / sweep_name / task / f"{model}_seed{seed}" / f"sweep_{sweep_name}.json"
        if path.exists():
            with open(path) as f:
                all_data.append(json.load(f))
    return all_data


def extract_metric(data_list, metric_key):
    rows = []
    for d in data_list:
        vals = [float(r.get(metric_key, float("nan"))) for r in d["results"]]
        rows.append(vals)
    return np.array(rows) if rows else np.array([])


def fmt_val(v):
    if isinstance(v, list):
        if v[0] == v[1]:
            return str(v[0])
        return f"{v[0]}–{v[1]}"
    return str(v)


def fmt_cell(mean, std, best=False, fmt=".2f"):
    s = f"{mean:{fmt}} ± {std:{fmt}}"
    return f"**{s}**" if best else s


def generate_reward_table(base_dir, sweep_name, task, header_prefix=""):
    """Generate one markdown table for a sweep (reward + fail + track_err)."""
    lines = []

    # Gather data for all models
    model_data = {}
    values = None
    for model in MODELS:
        data_list = load_sweep(base_dir, sweep_name, task, model, SEEDS)
        if data_list:
            if values is None:
                values = data_list[0]["values"]
            model_data[model] = data_list

    if not model_data or values is None:
        return ""

    col_labels = [fmt_val(v) for v in values]

    # Reward table
    lines.append(f"#### {header_prefix}{sweep_name.replace('_', ' ').title()} Sweep — Reward")
    header = "| Method | " + " | ".join(col_labels) + " | Sensitivity |"
    sep = "|" + "---|" * (len(col_labels) + 2)
    lines.append(header)
    lines.append(sep)

    # Find best reward per level
    level_means = {}
    for model in MODELS:
        if model not in model_data:
            continue
        arr = extract_metric(model_data[model], "episode_reward/mean")
        if arr.size == 0:
            continue
        level_means[model] = np.nanmean(arr, axis=0)

    for model in MODELS:
        if model not in model_data:
            continue
        arr = extract_metric(model_data[model], "episode_reward/mean")
        if arr.size == 0:
            continue
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(means)
        sens = float(np.max(means) - np.min(means))

        # Determine best per level
        cells = []
        for i in range(len(means)):
            is_best = all(means[i] >= level_means[m][i] - 1e-6
                          for m in level_means if m != model)
            cells.append(fmt_cell(means[i], stds[i], best=is_best))

        # Best sensitivity
        all_sens = []
        for m in MODELS:
            if m in level_means:
                all_sens.append(np.max(level_means[m]) - np.min(level_means[m]))
        is_best_sens = sens <= min(all_sens) + 1e-6

        sens_str = f"**{sens:.2f}**" if is_best_sens else f"{sens:.2f}"
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + f" | {sens_str} |")

    lines.append("")

    # Failure rate table
    lines.append(f"#### {header_prefix}{sweep_name.replace('_', ' ').title()} Sweep — Failure Rate")
    header = "| Method | " + " | ".join(col_labels) + " |"
    sep = "|" + "---|" * (len(col_labels) + 1)
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        if model not in model_data:
            continue
        arr = extract_metric(model_data[model], "failure_rate")
        if arr.size == 0:
            continue
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(means)
        cells = [fmt_cell(means[i], stds[i]) for i in range(len(means))]
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")

    lines.append("")

    # Tracking error table
    lines.append(f"#### {header_prefix}{sweep_name.replace('_', ' ').title()} Sweep — Tracking Error")
    header = "| Method | " + " | ".join(col_labels) + " |"
    sep = "|" + "---|" * (len(col_labels) + 1)
    lines.append(header)
    lines.append(sep)

    for model in MODELS:
        if model not in model_data:
            continue
        arr = extract_metric(model_data[model], "tracking_error/mean")
        if arr.size == 0:
            continue
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(means)
        cells = [fmt_cell(means[i], stds[i]) for i in range(len(means))]
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_overall_summary(base_dir):
    """Generate overall robustness summary with all metrics."""
    lines = []
    lines.append("#### Overall OOD Robustness Summary (v2)")
    lines.append("")
    lines.append("| Model | Avg Reward | Worst Reward | Max Sensitivity | "
                 "Max Fail Rate | Mean Track Err |")
    lines.append("|---|---|---|---|---|---|")

    all_sweeps = [
        ("randomized", "friction"),
        ("randomized", "push_magnitude"),
        ("randomized", "action_delay"),
        ("randomized", "action_delay_unseen"),
        ("randomized", "combined_shift"),
        ("push", "push_magnitude"),
        ("terrain", "push_magnitude"),
    ]

    for model in MODELS:
        rewards = []
        worst_rew = float("inf")
        max_sens = 0
        max_fail = 0
        track_errs = []

        for task, sweep in all_sweeps:
            data_list = load_sweep(base_dir, sweep, task, model, SEEDS)
            if not data_list:
                continue

            rew_arr = extract_metric(data_list, "episode_reward/mean")
            fail_arr = extract_metric(data_list, "failure_rate")
            track_arr = extract_metric(data_list, "tracking_error/mean")

            if rew_arr.size > 0:
                mean_per_level = np.nanmean(rew_arr, axis=0)
                rewards.append(np.mean(mean_per_level))
                worst_rew = min(worst_rew, np.min(mean_per_level))
                sens = np.max(mean_per_level) - np.min(mean_per_level)
                max_sens = max(max_sens, sens)

            if fail_arr.size > 0:
                max_fail = max(max_fail, np.nanmax(np.nanmean(fail_arr, axis=0)))

            if track_arr.size > 0:
                track_errs.append(np.nanmean(track_arr))

        if not rewards:
            continue

        avg_rew = np.mean(rewards)
        avg_track = np.mean(track_errs) if track_errs else float("nan")
        track_str = f"{avg_track:.2f}" if not np.isnan(avg_track) else "—"

        lines.append(f"| {MODEL_LABELS[model]} | {avg_rew:.2f} | {worst_rew:.2f} | "
                     f"{max_sens:.2f} | {max_fail:.2f} | {track_str} |")

    lines.append("")
    return "\n".join(lines)


def main():
    base_dir = Path("results/ood_v2")

    n_files = len(list(base_dir.rglob("*.json")))
    print(f"<!-- Generated from {n_files} sweep result files -->")
    print()

    # Must-do 1: Existing sweeps with behavioral metrics
    print("### OOD Sensitivity Sweeps v2 (5 seeds, 4 models, full behavioral metrics)")
    print()
    print("We evaluate all four models under OOD perturbations with four metrics:")
    print("reward (higher = better), failure rate (lower = better), ")
    print("tracking error (lower = better), and completion rate (higher = better).")
    print("50 episodes per level per seed, 5 seeds per model.")
    print()

    for sweep in ["friction", "push_magnitude", "action_delay"]:
        table = generate_reward_table(base_dir, sweep, "randomized")
        if table:
            print(table)

    # Must-do 2: Cross-task
    print("### Cross-Task OOD (push + terrain)")
    print()
    print("Push magnitude sweep evaluated on push-task and terrain-task checkpoints.")
    print()
    for task in ["push", "terrain"]:
        table = generate_reward_table(base_dir, "push_magnitude", task,
                                      header_prefix=f"{task.title()} Task — ")
        if table:
            print(table)

    # Must-do 3: Unseen range + combined
    print("### Unseen-Range and Combined-Shift Stress Tests")
    print()
    print("**Unseen action delay:** Training range was [0, 3]; we test up to delay=10.")
    print()
    table = generate_reward_table(base_dir, "action_delay_unseen", "randomized")
    if table:
        print(table)

    print("**Combined shift:** Friction, push, and delay all shifted simultaneously.")
    print()
    table = generate_reward_table(base_dir, "combined_shift", "randomized")
    if table:
        print(table)

    # Overall summary
    print(generate_overall_summary(base_dir))


if __name__ == "__main__":
    main()
