"""
Validate runs — detect failed, incomplete, or problematic experiment runs.

Scans output directories and produces a health report.

Usage:
    python -m src.utils.validate_runs --base-dir outputs/

    from src.utils.validate_runs import validate_all_runs
    report = validate_all_runs("outputs/")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_single_run(run_dir: str | Path) -> dict[str, Any]:
    """
    Validate a single run directory.

    Returns a report dict with:
        run_dir, status, issues, has_config, has_manifest,
        has_metrics, has_best_ckpt, has_latest_ckpt,
        manifest_status, final_step, is_complete
    """
    d = Path(run_dir)
    report: dict[str, Any] = {
        "run_dir": str(d),
        "issues": [],
    }

    # Config
    report["has_config"] = (d / "config.yaml").exists()
    if not report["has_config"]:
        report["issues"].append("missing config.yaml")

    # Manifest
    report["has_manifest"] = (d / "manifest.json").exists()
    manifest_status = "unknown"
    if report["has_manifest"]:
        try:
            with open(d / "manifest.json") as f:
                m = json.load(f)
            manifest_status = m.get("status", "unknown")
        except Exception:
            report["issues"].append("manifest.json corrupted")
            manifest_status = "corrupted"
    else:
        report["issues"].append("missing manifest.json")
    report["manifest_status"] = manifest_status

    # Metrics
    report["has_metrics_csv"] = (d / "metrics.csv").exists()
    report["has_tb"] = (d / "tb").is_dir()
    if not report["has_metrics_csv"] and not report["has_tb"]:
        report["issues"].append("no training metrics found")

    # Check metrics.csv for NaN
    if report["has_metrics_csv"]:
        try:
            content = (d / "metrics.csv").read_text()
            if "nan" in content.lower() or "inf" in content.lower():
                report["issues"].append("NaN or Inf detected in metrics.csv")
        except Exception:
            pass

    # Checkpoints
    ckpt_dir = d / "checkpoints"
    report["has_best_ckpt"] = (ckpt_dir / "best.pt").exists() if ckpt_dir.is_dir() else False
    report["has_latest_ckpt"] = (ckpt_dir / "latest.pt").exists() if ckpt_dir.is_dir() else False

    if not report["has_best_ckpt"] and not report["has_latest_ckpt"]:
        report["issues"].append("no checkpoints found")

    # Eval
    report["has_eval"] = (d / "eval_metrics.json").exists()

    # Overall status
    if manifest_status == "completed" and not report["issues"]:
        report["status"] = "valid"
    elif manifest_status == "completed" and report["issues"]:
        report["status"] = "valid_with_warnings"
    elif manifest_status == "started":
        report["status"] = "interrupted"
        report["issues"].append("manifest status=started (likely interrupted)")
    elif manifest_status == "failed":
        report["status"] = "failed"
    else:
        report["status"] = "incomplete"

    report["is_complete"] = report["status"] == "valid"
    return report


def validate_all_runs(
    base_dir: str | Path,
    task: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Validate all runs under base_dir.

    Returns a summary report with per-run results and aggregate stats.
    """
    base = Path(base_dir)
    runs = []

    for task_dir in sorted(base.iterdir()) if base.exists() else []:
        if not task_dir.is_dir():
            continue
        if task and task_dir.name != task:
            continue
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            if model and not method_dir.name.startswith(model):
                continue
            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                for ts_dir in sorted(seed_dir.iterdir()):
                    if ts_dir.is_dir():
                        runs.append(validate_single_run(ts_dir))

    summary = {
        "total_runs": len(runs),
        "valid": sum(1 for r in runs if r["status"] == "valid"),
        "valid_with_warnings": sum(1 for r in runs if r["status"] == "valid_with_warnings"),
        "interrupted": sum(1 for r in runs if r["status"] == "interrupted"),
        "failed": sum(1 for r in runs if r["status"] == "failed"),
        "incomplete": sum(1 for r in runs if r["status"] == "incomplete"),
        "missing_eval": sum(1 for r in runs if not r["has_eval"]),
        "runs": runs,
    }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    """Pretty-print validation report."""
    print(f"\n{'='*70}")
    print(f"  RUN VALIDATION REPORT")
    print(f"{'='*70}")
    print(f"  Total runs found:     {summary['total_runs']}")
    print(f"  Valid:                 {summary['valid']}")
    print(f"  Valid w/ warnings:     {summary['valid_with_warnings']}")
    print(f"  Interrupted:           {summary['interrupted']}")
    print(f"  Failed:                {summary['failed']}")
    print(f"  Incomplete:            {summary['incomplete']}")
    print(f"  Missing eval:          {summary['missing_eval']}")
    print(f"{'='*70}")

    problem_runs = [r for r in summary["runs"] if r["status"] != "valid"]
    if problem_runs:
        print(f"\n  PROBLEM RUNS ({len(problem_runs)}):")
        for r in problem_runs:
            print(f"\n  [{r['status'].upper()}] {r['run_dir']}")
            for issue in r["issues"]:
                print(f"    - {issue}")
    else:
        print("\n  All runs valid.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate experiment runs")
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    summary = validate_all_runs(args.base_dir, args.task, args.model)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)

    # Exit code: 1 if any runs are failed/interrupted
    if summary["failed"] + summary["interrupted"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
