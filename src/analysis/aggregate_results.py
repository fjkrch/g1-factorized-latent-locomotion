#!/usr/bin/env python3
"""
Aggregate results — production aggregation across seeds, methods, tasks, ablations.

Usage:
    # Aggregate all main comparison results
    python src/analysis/aggregate_results.py --base-dir outputs/ --output results/aggregated/

    # Aggregate specific method/task
    python src/analysis/aggregate_results.py --base-dir outputs/ --task randomized --model dynamite

    # Include ablations
    python src/analysis/aggregate_results.py --base-dir outputs/ --include-ablations

    # Export as CSV
    python src/analysis/aggregate_results.py --base-dir outputs/ --output results/aggregated/ --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.utils.metrics_io import discover_run_dirs, is_run_complete, read_run_summary


def aggregate_group(
    run_dirs: list[Path],
    metric_file: str = "eval_metrics.json",
) -> dict[str, Any]:
    """
    Aggregate metrics across seed runs for a single experiment group.

    Returns dict with per-metric mean, std, min, max, n_seeds, per_seed values.
    """
    all_metrics: dict[str, list[float]] = defaultdict(list)
    valid_seeds = 0
    skipped = []

    for rd in run_dirs:
        mpath = rd / metric_file
        if not mpath.exists():
            skipped.append(str(rd))
            continue
        if not is_run_complete(rd):
            skipped.append(f"{rd} (not completed)")
            continue
        try:
            data = read_run_summary(mpath)
        except Exception as e:
            skipped.append(f"{rd} (read error: {e})")
            continue

        valid_seeds += 1
        for k, v in data.items():
            if isinstance(v, (int, float)):
                all_metrics[k].append(float(v))

    result: dict[str, Any] = {
        "n_seeds": valid_seeds,
        "skipped": skipped,
        "metrics": {},
    }

    for k, vals in all_metrics.items():
        arr = np.array(vals)
        result["metrics"][k] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "n": len(vals),
            "per_seed": [round(v, 4) for v in vals],
        }

    return result


def aggregate_main_comparison(
    base_dir: str | Path,
    tasks: list[str],
    models: list[str],
    variant: str = "full",
) -> dict[str, dict[str, Any]]:
    """Aggregate all main comparison groups."""
    results = {}
    base = Path(base_dir)

    for task in tasks:
        for model in models:
            group_key = f"{task}/{model}_{variant}"
            method_dir = base / task / f"{model}_{variant}"
            if not method_dir.exists():
                continue

            seed_runs = []
            for seed_dir in sorted(method_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                    # Take the latest timestamp dir
                    ts_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
                    if ts_dirs:
                        seed_runs.append(ts_dirs[-1])

            if seed_runs:
                results[group_key] = aggregate_group(seed_runs)

    return results


def aggregate_ablations(
    base_dir: str | Path,
    task: str = "randomized",
    model: str = "dynamite",
    ablations: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate ablation experiments."""
    if ablations is None:
        ablations = [
            "full", "seq_len_4", "seq_len_16", "no_latent",
            "single_latent", "no_aux_loss", "depth_1", "depth_4",
        ]

    results = {}
    base = Path(base_dir)

    for abl in ablations:
        method_dir = base / task / f"{model}_{abl}"
        if not method_dir.exists():
            continue

        seed_runs = []
        for seed_dir in sorted(method_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                ts_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
                if ts_dirs:
                    seed_runs.append(ts_dirs[-1])

        if seed_runs:
            results[abl] = aggregate_group(seed_runs)

    return results


def results_to_csv(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    primary_metric: str = "episode_reward/mean",
) -> None:
    """Export aggregated results to CSV."""
    rows = []
    for group_key, data in sorted(results.items()):
        metrics = data.get("metrics", {})
        row = {
            "experiment": group_key,
            "n_seeds": data.get("n_seeds", 0),
        }
        for mname, mstats in metrics.items():
            row[f"{mname}_mean"] = mstats["mean"]
            row[f"{mname}_std"] = mstats["std"]
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    # Ensure all fields are captured
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--output", type=str, default="results/aggregated")
    parser.add_argument("--task", type=str, default=None,
                        help="Aggregate for a specific task only")
    parser.add_argument("--model", type=str, default=None,
                        help="Aggregate for a specific model only")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--csv", action="store_true", help="Also export CSV")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [args.task] if args.task else ["flat", "push", "randomized", "terrain"]
    models = [args.model] if args.model else ["mlp", "lstm", "transformer", "dynamite"]

    # ── Main comparison ──
    print("Aggregating main comparison...")
    main_results = aggregate_main_comparison(args.base_dir, tasks, models)

    with open(output_dir / "main_comparison.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print(f"  Saved: {output_dir / 'main_comparison.json'} ({len(main_results)} groups)")

    if args.csv:
        results_to_csv(main_results, output_dir / "main_comparison.csv")
        print(f"  Saved: {output_dir / 'main_comparison.csv'}")

    # ── Ablations ──
    if args.include_ablations:
        print("Aggregating ablations...")
        abl_results = aggregate_ablations(args.base_dir)
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(abl_results, f, indent=2)
        print(f"  Saved: {output_dir / 'ablation_results.json'} ({len(abl_results)} variants)")

        if args.csv:
            results_to_csv(abl_results, output_dir / "ablation_results.csv")

    # ── Summary table ──
    print("\nSummary:")
    for group_key, data in sorted(main_results.items()):
        metrics = data.get("metrics", {})
        reward = metrics.get("episode_reward/mean", {})
        r_mean = reward.get("mean", "?")
        r_std = reward.get("std", "?")
        n = data.get("n_seeds", 0)
        print(f"  {group_key:40s}  reward={r_mean:>8} ± {r_std:<8}  (n={n})")


if __name__ == "__main__":
    main()
