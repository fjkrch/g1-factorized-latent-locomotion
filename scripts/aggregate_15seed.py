#!/usr/bin/env python3
"""
Aggregate all n=15 results after the 15-seed expansion.

This script:
1. Aggregates ID eval metrics (mean reward) for all 4 models × 15 seeds
2. Aggregates OOD sweep results (combined_shift, friction, push_magnitude, action_delay)
3. Aggregates push recovery results
4. Computes paired t-tests, Holm–Bonferroni corrections, Cohen's d_z, bootstrap CIs
5. Outputs comprehensive JSON and human-readable summary

Usage:
    python scripts/aggregate_15seed.py
"""

import json
import sys
import glob
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = ["dynamite", "lstm", "transformer", "mlp"]
SEEDS = list(range(42, 57))  # 42-56 = 15 seeds
SWEEP_TYPES = ["combined_shift", "friction", "push_magnitude", "action_delay", "action_delay_unseen"]
OUTPUT_DIR = PROJECT_ROOT / "results" / "aggregated_15seed"


def find_checkpoint_dir(model, seed):
    """Find the run directory for a model/seed."""
    pattern = f"outputs/randomized/{model}_full/seed_{seed}/*"
    dirs = sorted(glob.glob(str(PROJECT_ROOT / pattern)))
    if dirs:
        return Path(dirs[-1])
    return None


def load_id_reward(model, seed):
    """Load ID eval reward for a model/seed."""
    run_dir = find_checkpoint_dir(model, seed)
    if run_dir is None:
        return None
    # Check multiple possible locations
    for subpath in ["eval_recomputed/eval_metrics.json", "eval_metrics.json"]:
        eval_file = run_dir / subpath
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            for key in ["episode_reward/mean", "eval/reward_mean", "reward_mean", "mean_reward"]:
                if key in data:
                    return data[key]
    # Fallback: extract from metrics.csv (last eval/reward_mean)
    metrics_file = run_dir / "metrics.csv"
    if metrics_file.exists():
        import csv
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row and "eval_reward_mean" in last_row:
                return float(last_row["eval_reward_mean"])
    return None


def load_sweep_results(model, seed, sweep_name):
    """Load sweep results for a model/seed/sweep."""
    result_file = PROJECT_ROOT / f"results/ood_v2/{sweep_name}/randomized/{model}_seed{seed}/sweep_{sweep_name}.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def load_push_recovery(model, seed):
    """Load push recovery results for a model/seed."""
    result_file = PROJECT_ROOT / f"results/push_recovery/{model}_seed{seed}/push_recovery_{model}_seed{seed}.json"
    if not result_file.exists():
        return None
    with open(result_file) as f:
        return json.load(f)


def paired_ttest(x, y):
    """Paired t-test returning t-stat, p-value, Cohen's d_z."""
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    if n < 2:
        return None, None, None
    t_stat, p_val = stats.ttest_rel(x, y)
    d_z = np.mean(diffs) / np.std(diffs, ddof=1)
    return t_stat, p_val, d_z


def bootstrap_ci(data, n_boot=10000, ci=0.95, func=np.mean):
    """Bootstrap confidence interval."""
    data = np.array(data)
    boots = np.array([func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(boots, 100 * alpha), np.percentile(boots, 100 * (1 - alpha))


def holm_bonferroni(p_values):
    """Holm–Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.ones(n)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_indices[i]
        prev_idx = sorted_indices[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def main():
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # 1. ID EVAL AGGREGATION
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PHASE 1: ID EVALUATION AGGREGATION")
    print("=" * 70)

    id_rewards = {}
    for model in MODELS:
        rewards = []
        for seed in SEEDS:
            r = load_id_reward(model, seed)
            if r is not None:
                rewards.append(r)
            else:
                print(f"  [WARN] Missing ID eval: {model} seed {seed}")
        id_rewards[model] = rewards
        n = len(rewards)
        if n > 0:
            mean = np.mean(rewards)
            std = np.std(rewards, ddof=1)
            ci_lo, ci_hi = bootstrap_ci(rewards)
            print(f"  {model:12s}: {mean:.4f} ± {std:.4f}  (n={n})  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    results["id_rewards"] = {
        model: {
            "values": vals,
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "n": len(vals),
            "ci_95": [float(x) for x in bootstrap_ci(vals)] if len(vals) > 1 else None,
        }
        for model, vals in id_rewards.items()
    }

    # Paired t-tests for ID
    print("\n  Paired t-tests (ID, randomized):")
    id_tests = []
    model_pairs = list(combinations(MODELS, 2))
    for m1, m2 in model_pairs:
        # Find matched seeds
        common_seeds = []
        vals1, vals2 = [], []
        for seed in SEEDS:
            r1 = load_id_reward(m1, seed)
            r2 = load_id_reward(m2, seed)
            if r1 is not None and r2 is not None:
                common_seeds.append(seed)
                vals1.append(r1)
                vals2.append(r2)
        if len(vals1) >= 2:
            t, p, d = paired_ttest(vals1, vals2)
            id_tests.append({"pair": f"{m1}_vs_{m2}", "t": t, "p": p, "d_z": d, "n": len(vals1)})
            print(f"    {m1} vs {m2}: t={t:.3f}, p={p:.4f}, d_z={d:.3f} (n={len(vals1)})")

    # Holm–Bonferroni
    if id_tests:
        raw_ps = [t["p"] for t in id_tests]
        adj_ps = holm_bonferroni(np.array(raw_ps))
        for i, t in enumerate(id_tests):
            t["p_adjusted"] = float(adj_ps[i])
        print("\n  After Holm–Bonferroni:")
        for t in sorted(id_tests, key=lambda x: x["p_adjusted"]):
            sig = "***" if t["p_adjusted"] < 0.001 else "**" if t["p_adjusted"] < 0.01 else "*" if t["p_adjusted"] < 0.05 else ""
            print(f"    {t['pair']:30s}: p_adj={t['p_adjusted']:.4f} {sig}")

    results["id_tests"] = id_tests

    # ══════════════════════════════════════════════════════════════
    # 2. OOD SWEEP AGGREGATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 2: OOD SWEEP AGGREGATION")
    print("=" * 70)

    ood_results = {}
    all_ood_tests = []

    for sweep in SWEEP_TYPES:
        print(f"\n  ── {sweep} ──")
        sweep_data = {}

        for model in MODELS:
            level_rewards = {}  # level -> [seed_values]
            for seed in SEEDS:
                data = load_sweep_results(model, seed, sweep)
                if data is None:
                    continue
                # Parse sweep results — results is a list of dicts
                results_list = data.get("results", data.get("sweep_results", data.get("levels", [])))
                if isinstance(results_list, list):
                    for i, entry in enumerate(results_list):
                        if not isinstance(entry, dict):
                            continue
                        # Level identifier: "value" field (may be scalar or list)
                        level = entry.get("value", entry.get("level", entry.get("name", i)))
                        if isinstance(level, list):
                            level = str(level)
                        else:
                            level = str(level)
                        reward = entry.get("episode_reward/mean", entry.get("reward_mean", entry.get("mean_reward")))
                        if reward is not None:
                            level_rewards.setdefault(level, []).append(reward)
                elif isinstance(results_list, dict):
                    for level_key, level_data in results_list.items():
                        reward = None
                        if isinstance(level_data, dict):
                            reward = level_data.get("episode_reward/mean", level_data.get("reward_mean"))
                        elif isinstance(level_data, (int, float)):
                            reward = level_data
                        if reward is not None:
                            level_rewards.setdefault(str(level_key), []).append(reward)

            sweep_data[model] = {}
            for level, values in sorted(level_rewards.items()):
                n = len(values)
                mean = np.mean(values)
                std = np.std(values, ddof=1) if n > 1 else 0
                sweep_data[model][level] = {
                    "values": values,
                    "mean": float(mean),
                    "std": float(std),
                    "n": n,
                }
                print(f"    {model:12s} L{level}: {mean:.4f} ± {std:.4f} (n={n})")

        ood_results[sweep] = sweep_data

        # Paired t-tests per level
        for level in sorted(set().union(*[set(sweep_data.get(m, {}).keys()) for m in MODELS])):
            for m1, m2 in model_pairs:
                v1 = sweep_data.get(m1, {}).get(level, {}).get("values", [])
                v2 = sweep_data.get(m2, {}).get(level, {}).get("values", [])
                # Match by seed order (both should be in same order)
                n_common = min(len(v1), len(v2))
                if n_common >= 2:
                    t, p, d = paired_ttest(v1[:n_common], v2[:n_common])
                    all_ood_tests.append({
                        "sweep": sweep, "level": level,
                        "pair": f"{m1}_vs_{m2}",
                        "t": float(t), "p": float(p), "d_z": float(d),
                        "n": n_common
                    })

    # Holm–Bonferroni across ALL OOD tests
    if all_ood_tests:
        raw_ps = np.array([t["p"] for t in all_ood_tests])
        adj_ps = holm_bonferroni(raw_ps)
        for i, t in enumerate(all_ood_tests):
            t["p_adjusted"] = float(adj_ps[i])

        print(f"\n  Total OOD pairwise tests: {len(all_ood_tests)}")
        significant = [t for t in all_ood_tests if t["p_adjusted"] < 0.05]
        print(f"  Significant after Holm–Bonferroni (p < 0.05): {len(significant)}/{len(all_ood_tests)}")
        for t in sorted(significant, key=lambda x: x["p_adjusted"]):
            print(f"    {t['sweep']:20s} L{t['level']:3s} {t['pair']:30s}: p_adj={t['p_adjusted']:.4f}, d_z={t['d_z']:.3f}")

    results["ood_sweeps"] = ood_results
    results["ood_tests"] = all_ood_tests

    # ══════════════════════════════════════════════════════════════
    # 3. PUSH RECOVERY AGGREGATION
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 3: PUSH RECOVERY AGGREGATION")
    print("=" * 70)

    push_results = {}
    push_tests = []

    for model in MODELS:
        mag_data = {}  # magnitude -> {recovery_rate: [...], ...}
        for seed in SEEDS:
            data = load_push_recovery(model, seed)
            if data is None:
                continue
            if "results" in data:
                for mag_key, mag_data_inner in data["results"].items():
                    mag = str(mag_key)
                    if isinstance(mag_data_inner, dict):
                        rr = mag_data_inner.get("recovery_rate", mag_data_inner.get("non_fall_rate"))
                        if rr is not None:
                            mag_data.setdefault(mag, []).append(rr)

        push_results[model] = {}
        for mag in sorted(mag_data.keys(), key=lambda x: float(x)):
            values = mag_data[mag]
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0
            push_results[model][mag] = {
                "values": values,
                "mean": float(mean),
                "std": float(std),
                "n": n,
            }
            print(f"  {model:12s} mag={mag:4s}: {mean:.4f} ± {std:.4f} (n={n})")

    # Push recovery paired tests
    for mag in sorted(set().union(*[set(push_results.get(m, {}).keys()) for m in MODELS]), key=lambda x: float(x)):
        for m1, m2 in model_pairs:
            v1 = push_results.get(m1, {}).get(mag, {}).get("values", [])
            v2 = push_results.get(m2, {}).get(mag, {}).get("values", [])
            n_common = min(len(v1), len(v2))
            if n_common >= 2:
                t, p, d = paired_ttest(v1[:n_common], v2[:n_common])
                push_tests.append({
                    "test": "push_recovery", "magnitude": mag,
                    "pair": f"{m1}_vs_{m2}",
                    "t": float(t), "p": float(p), "d_z": float(d),
                    "n": n_common,
                })

    if push_tests:
        raw_ps = np.array([t["p"] for t in push_tests])
        adj_ps = holm_bonferroni(raw_ps)
        for i, t in enumerate(push_tests):
            t["p_adjusted"] = float(adj_ps[i])
        significant = [t for t in push_tests if t["p_adjusted"] < 0.05]
        print(f"\n  Push recovery tests: {len(push_tests)}, significant: {len(significant)}")
        for t in sorted(significant, key=lambda x: x["p_adjusted"]):
            print(f"    mag={t['magnitude']:4s} {t['pair']:30s}: p_adj={t['p_adjusted']:.4f}, d_z={t['d_z']:.3f}")

    results["push_recovery"] = push_results
    results["push_tests"] = push_tests

    # ══════════════════════════════════════════════════════════════
    # 4. SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_tests = len(id_tests) + len(all_ood_tests) + len(push_tests)
    all_tests = id_tests + all_ood_tests + push_tests
    sig_count = sum(1 for t in all_tests if t.get("p_adjusted", 1) < 0.05)
    print(f"  Total statistical tests: {total_tests}")
    print(f"  Significant (Holm–Bonferroni p < 0.05): {sig_count}")
    print(f"  ID seeds per model: {[len(id_rewards[m]) for m in MODELS]}")

    results["summary"] = {
        "total_tests": total_tests,
        "significant_tests": sig_count,
        "seeds_per_model": {m: len(id_rewards[m]) for m in MODELS},
        "target_seeds": 15,
    }

    # Save
    output_file = OUTPUT_DIR / "full_15seed_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")

    # Also save a compact summary
    summary = {
        "id_comparison": {
            model: {
                "mean": results["id_rewards"][model]["mean"],
                "std": results["id_rewards"][model]["std"],
                "n": results["id_rewards"][model]["n"],
                "ci_95": results["id_rewards"][model]["ci_95"],
            }
            for model in MODELS
        },
        "id_paired_tests": [
            {k: v for k, v in t.items() if k != "values"}
            for t in id_tests
        ],
        "ood_significant_tests": [
            {k: v for k, v in t.items()}
            for t in all_ood_tests if t.get("p_adjusted", 1) < 0.05
        ],
        "push_significant_tests": [
            {k: v for k, v in t.items()}
            for t in push_tests if t.get("p_adjusted", 1) < 0.05
        ],
        "total_tests": total_tests,
        "significant_count": sig_count,
    }
    summary_file = OUTPUT_DIR / "summary_15seed.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
