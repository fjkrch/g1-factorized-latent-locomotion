#!/usr/bin/env python3
"""
Aggregate results across seeds for a given experiment.

Usage:
    python scripts/aggregate_seeds.py \
        --run_dirs outputs/flat/dynamite/seed_42/... outputs/flat/dynamite/seed_43/... \
        --output results/aggregated/dynamite_flat.json

    # Or auto-discover seeds:
    python scripts/aggregate_seeds.py \
        --pattern "outputs/flat/dynamite/seed_*" \
        --output results/aggregated/dynamite_flat.json

Inputs:
    - Multiple run directories (one per seed)
    - Each must contain eval_metrics.json

Outputs:
    - Aggregated JSON with mean/std/min/max per metric across seeds
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import aggregate_seeds, save_aggregated_results


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate results across seeds")
    parser.add_argument("--run_dirs", nargs="+", default=None,
                        help="Explicit list of run directories")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern to find seed directories")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--metric_file", type=str, default="eval_metrics.json",
                        help="Name of metrics file in each run dir")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    elif args.pattern:
        # Find seed directories, then get latest run in each
        seed_dirs = sorted(glob.glob(args.pattern))
        run_dirs = []
        for sd in seed_dirs:
            # Get latest timestamped subdir
            subdirs = sorted(Path(sd).iterdir())
            if subdirs:
                run_dirs.append(subdirs[-1])
    else:
        print("Error: specify --run_dirs or --pattern")
        sys.exit(1)

    print(f"[Aggregate] Found {len(run_dirs)} runs:")
    for rd in run_dirs:
        has_metrics = (rd / args.metric_file).exists()
        print(f"  {rd} {'✓' if has_metrics else '✗ (missing metrics)'}")

    results = aggregate_seeds(run_dirs, metric_file=args.metric_file)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_aggregated_results(results, output_path)

    print(f"\n[Aggregate] Results ({len(results)} metrics):")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n_seeds']})")
    print(f"\n[Aggregate] Saved to: {output_path}")


if __name__ == "__main__":
    main()
