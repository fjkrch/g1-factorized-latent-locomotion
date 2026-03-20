#!/usr/bin/env python3
"""
Batch 1 Analysis: Checkpoint Selection Results

Reads push_magnitude sweep results for each checkpoint and computes:
1. Per-level mean rewards
2. Worst-case performance
3. Sensitivity (nominal → worst-case degradation)
4. Estimated RobustScore delta vs baseline (best.pt)

Usage:
    python3 scripts/experiments/analyze_batch1.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)


def load_sweep_results(filepath):
    """Load sweep results and return per-level rewards."""
    with open(filepath) as f:
        d = json.load(f)
    return {
        "levels": [r["value"] for r in d["results"]],
        "rewards": [r["episode_reward/mean"] for r in d["results"]],
        "stds": [r.get("episode_reward/std", 0) for r in d["results"]],
    }


def compute_metrics(results):
    """Compute OOD metrics from sweep results."""
    rewards = results["rewards"]
    nominal = rewards[0]  # push [0,0]
    worst = min(rewards)
    sensitivity = nominal - worst  # positive = more degradation
    ood_avg = np.mean(rewards)
    return {
        "nominal": nominal,
        "worst_case": worst,
        "sensitivity": sensitivity,
        "ood_avg": ood_avg,
    }


def main():
    output_base = Path("results/batch1_checkpoint_selection/seed_42")
    checkpoints = ["ckpt_4300800", "ckpt_5529600", "best", "ckpt_7372800", "ckpt_9216000"]

    # Training step mapping for context
    step_map = {
        "ckpt_4300800": 4_300_800,
        "ckpt_5529600": 5_529_600,
        "best": 6_635_520,  # approximate best.pt step
        "ckpt_7372800": 7_372_800,
        "ckpt_9216000": 9_216_000,
    }

    print("=" * 90)
    print("BATCH 1: CHECKPOINT SELECTION ANALYSIS")
    print("=" * 90)

    all_metrics = {}
    for ckpt in checkpoints:
        f = output_base / ckpt / "sweep_push_magnitude.json"
        if not f.exists():
            print(f"[MISSING] {ckpt}")
            continue
        results = load_sweep_results(f)
        metrics = compute_metrics(results)
        all_metrics[ckpt] = {"results": results, "metrics": metrics}

    if not all_metrics:
        print("No results found. Run batch1_checkpoint_selection.sh first.")
        sys.exit(1)

    # ── Table 1: Per-level rewards ──
    print("\n--- Per-Level Push Rewards ---")
    header = f"{'Checkpoint':<18} {'Step':>10}"
    levels = all_metrics[next(iter(all_metrics))]["results"]["levels"]
    for lv in levels:
        header += f" {'push '+str(lv):>12}"
    print(header)
    print("─" * len(header))

    for ckpt in checkpoints:
        if ckpt not in all_metrics:
            continue
        rewards = all_metrics[ckpt]["results"]["rewards"]
        row = f"{ckpt:<18} {step_map[ckpt]:>10,}"
        for r in rewards:
            row += f" {r:>12.4f}"
        print(row)

    # ── Table 2: Summary metrics ──
    print("\n--- OOD Summary Metrics ---")
    print(f"{'Checkpoint':<18} {'Step':>10} {'Nominal':>10} {'Worst':>10} {'Sensitivity':>12} {'OOD Avg':>10}")
    print("─" * 72)

    baseline_metrics = None
    for ckpt in checkpoints:
        if ckpt not in all_metrics:
            continue
        m = all_metrics[ckpt]["metrics"]
        if ckpt == "best":
            baseline_metrics = m
        print(f"{ckpt:<18} {step_map[ckpt]:>10,} {m['nominal']:>10.4f} {m['worst_case']:>10.4f} {m['sensitivity']:>12.4f} {m['ood_avg']:>10.4f}")

    # ── Table 3: Delta vs baseline ──
    if baseline_metrics:
        print("\n--- Delta vs Baseline (best.pt) ---")
        print(f"{'Checkpoint':<18} {'Δ Nominal':>10} {'Δ Worst':>10} {'Δ Sensitivity':>14} {'Δ OOD Avg':>10}")
        print("─" * 64)

        for ckpt in checkpoints:
            if ckpt not in all_metrics or ckpt == "best":
                continue
            m = all_metrics[ckpt]["metrics"]
            dn = m["nominal"] - baseline_metrics["nominal"]
            dw = m["worst_case"] - baseline_metrics["worst_case"]
            ds = m["sensitivity"] - baseline_metrics["sensitivity"]
            da = m["ood_avg"] - baseline_metrics["ood_avg"]
            print(f"{ckpt:<18} {dn:>+10.4f} {dw:>+10.4f} {ds:>+14.4f} {da:>+10.4f}")

    # ── Decision ──
    print("\n" + "=" * 90)
    print("DECISION")
    print("=" * 90)

    if baseline_metrics is None:
        print("Cannot compute decision: baseline (best.pt) results missing.")
        return

    # Find best candidate
    best_candidate = None
    best_score = float("-inf")

    for ckpt in checkpoints:
        if ckpt not in all_metrics or ckpt == "best":
            continue
        m = all_metrics[ckpt]["metrics"]

        # Decision criteria:
        # 1. Worst-case improves ≥ 0.15 (less negative is better)
        wc_improve = m["worst_case"] - baseline_metrics["worst_case"]

        # 2. Sensitivity reduces ≥ 15%
        sens_reduce = 1 - (m["sensitivity"] / baseline_metrics["sensitivity"]) if baseline_metrics["sensitivity"] > 0 else 0

        # 3. Nominal doesn't degrade more than 0.10
        nom_degrade = baseline_metrics["nominal"] - m["nominal"]  # positive = worse

        passed = False
        reason = ""
        if wc_improve >= 0.15:
            passed = True
            reason = f"worst-case improved by {wc_improve:.4f} (≥0.15 threshold)"
        elif sens_reduce >= 0.15:
            passed = True
            reason = f"sensitivity reduced by {sens_reduce:.1%} (≥15% threshold)"

        if nom_degrade > 0.10 and passed:
            reason += f" BUT nominal degraded by {nom_degrade:.4f} (>0.10 limit)"
            passed = False

        status = "✓ PROMOTE" if passed else "✗ reject"

        # Composite improvement score for ranking
        score = 0.30 * wc_improve + 0.20 * (-m["sensitivity"] + baseline_metrics["sensitivity"]) + 0.10 * (m["ood_avg"] - baseline_metrics["ood_avg"])

        print(f"\n  {ckpt}: {status}")
        print(f"    worst-case Δ: {wc_improve:+.4f}, sensitivity Δ: {sens_reduce:+.1%}, nominal Δ: {-nom_degrade:+.4f}")
        if reason:
            print(f"    reason: {reason}")

        if passed and score > best_score:
            best_score = score
            best_candidate = ckpt

    if best_candidate:
        print(f"\n  → PROMOTE {best_candidate} to 3-seed evaluation across all sweep types.")
        print(f"    Next: Run batch1_checkpoint_confirm.sh with CKPT={best_candidate}")
    else:
        print(f"\n  → No checkpoint significantly improves over best.pt for OOD robustness.")
        print(f"    Checkpoint selection is NOT a fruitful intervention for this model.")
        print(f"    Proceed to Batch 2: Randomization Strengthening.")

    # ── Save structured results ──
    output_file = output_base / "batch1_analysis.json"
    analysis = {
        "batch": "1_checkpoint_selection",
        "seed": 42,
        "sweep": "push_magnitude",
        "baseline_checkpoint": "best",
        "decision": "promote" if best_candidate else "reject",
        "promoted_checkpoint": best_candidate,
        "checkpoints": {},
    }
    for ckpt in checkpoints:
        if ckpt in all_metrics:
            analysis["checkpoints"][ckpt] = {
                "step": step_map[ckpt],
                **all_metrics[ckpt]["metrics"],
            }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
