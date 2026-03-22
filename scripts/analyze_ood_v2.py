#!/usr/bin/env python3
"""
Comprehensive OOD Analysis (Must-do 5).

Reads all results from results/ood_v2/ and produces:
  1. Per-sweep tables with all 4 metrics (reward, failure_rate, tracking_error, completion_rate)
  2. Severe-level mean (worst 2 levels per sweep)
  3. Worst-case score across all sweeps
  4. AUC degradation summary (area under reward vs perturbation curve)
  5. 95% CIs and Cohen's d effect sizes (DynaMITE vs each baseline)
  6. Holm-Bonferroni corrected p-values for targeted comparisons

Outputs:
  results/aggregated/ood_analysis_v2.json
  Console markdown tables for README
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats


MODELS = ["dynamite", "lstm", "transformer", "mlp"]
MODEL_LABELS = {
    "dynamite": "DynaMITE (Ours)",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "mlp": "MLP",
}
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# Sweep configs
SWEEPS_RANDOMIZED = ["friction", "push_magnitude", "action_delay",
                     "action_delay_unseen", "combined_shift"]
SWEEPS_CROSS_TASK = {"push": ["push_magnitude"], "terrain": ["push_magnitude"]}

METRICS = ["episode_reward/mean", "failure_rate", "tracking_error/mean", "completion_rate"]
METRIC_LABELS = {
    "episode_reward/mean": "Reward",
    "failure_rate": "Fail Rate",
    "tracking_error/mean": "Track Err",
    "completion_rate": "Compl %",
}


def load_sweep(base_dir, sweep_name, task, model, seeds):
    """Load all seed results for a model/sweep/task combination."""
    all_data = []
    found_seeds = []
    for seed in seeds:
        path = base_dir / sweep_name / task / f"{model}_seed{seed}" / f"sweep_{sweep_name}.json"
        if not path.exists():
            # Also try old format (results/sweeps_multiseed/)
            alt_path = Path("results/sweeps_multiseed") / sweep_name / f"{model}_seed{seed}" / f"sweep_{sweep_name}.json"
            if alt_path.exists():
                path = alt_path
            else:
                continue
        with open(path) as f:
            data = json.load(f)
        all_data.append(data)
        found_seeds.append(seed)
    return all_data, found_seeds


def extract_metric_array(sweep_data_list, metric_key):
    """Extract metric across seeds × levels → (n_seeds, n_levels)."""
    arrays = []
    for data in sweep_data_list:
        vals = []
        for r in data["results"]:
            v = r.get(metric_key, None)
            if v is None:
                v = float("nan")
            vals.append(float(v))
        arrays.append(vals)
    return np.array(arrays)  # (n_seeds, n_levels)


def compute_auc(x_vals, y_vals):
    """Compute AUC using trapezoidal rule (normalized by x-range)."""
    if len(x_vals) < 2:
        return float("nan")
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    # Sort by x
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    auc = np.trapz(y, x)
    x_range = x[-1] - x[0]
    return auc / x_range if x_range > 0 else float("nan")


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction. Returns list of (original_p, adjusted_p, reject)."""
    n = len(p_values)
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    results = [None] * n
    for rank, (p, orig_idx) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = min(adjusted, 1.0)
        results[orig_idx] = (p, adjusted, adjusted < alpha)
    return results


def extract_x_vals(values):
    """Convert sweep values to numeric x-axis values."""
    x = []
    for v in values:
        if isinstance(v, list):
            x.append(v[0] if v[0] == v[1] else (v[0] + v[1]) / 2)
        else:
            x.append(float(v))
    return x


def analyze_sweep(base_dir, sweep_name, task, models, seeds):
    """Full analysis of one sweep across all models."""
    result = {"sweep": sweep_name, "task": task, "models": {}}

    for model in models:
        data_list, found_seeds = load_sweep(base_dir, sweep_name, task, model, seeds)
        if not data_list:
            continue

        n_seeds = len(found_seeds)
        values = data_list[0]["values"]
        x_vals = extract_x_vals(values)

        model_result = {
            "n_seeds": n_seeds,
            "seeds": found_seeds,
            "values": values,
            "x_values": x_vals,
        }

        for metric in METRICS:
            arr = extract_metric_array(data_list, metric)  # (n_seeds, n_levels)
            if np.all(np.isnan(arr)):
                continue

            mean_per_level = np.nanmean(arr, axis=0)
            std_per_level = np.nanstd(arr, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean_per_level)

            # Overall mean across all levels
            seed_means = np.nanmean(arr, axis=1)
            overall_mean = float(np.mean(seed_means))
            overall_std = float(np.std(seed_means, ddof=1)) if n_seeds > 1 else 0.0

            # Sensitivity (max - min of level means)
            sensitivity = float(np.nanmax(mean_per_level) - np.nanmin(mean_per_level))

            # Severe-level mean (last 2 levels — typically the hardest)
            n_severe = min(2, len(mean_per_level))
            if "reward" in metric:
                severe_idx = np.argsort(mean_per_level)[:n_severe]  # lowest reward
            else:
                severe_idx = np.argsort(mean_per_level)[-n_severe:]  # highest error/failure
            severe_mean = float(np.mean(mean_per_level[severe_idx]))

            # Worst single level
            if "reward" in metric:
                worst_level = float(np.nanmin(mean_per_level))
            else:
                worst_level = float(np.nanmax(mean_per_level))

            # AUC (reward only, using x_vals)
            auc = float("nan")
            if "reward" in metric and len(x_vals) == len(mean_per_level):
                auc = compute_auc(x_vals, mean_per_level)

            # 95% CI on overall mean
            ci_lo, ci_hi = float("nan"), float("nan")
            if n_seeds >= 3:
                se = overall_std / np.sqrt(n_seeds)
                t_crit = stats.t.ppf(0.975, n_seeds - 1)
                ci_lo = overall_mean - t_crit * se
                ci_hi = overall_mean + t_crit * se

            mkey = metric.replace("/", "_")
            model_result[mkey] = {
                "per_level_mean": mean_per_level.tolist(),
                "per_level_std": std_per_level.tolist(),
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "sensitivity": sensitivity,
                "severe_mean": severe_mean,
                "worst_level": worst_level,
                "auc": auc,
                "ci_95": [ci_lo, ci_hi],
            }

        result["models"][model] = model_result

    # Pairwise comparisons: DynaMITE vs each other model (reward)
    if "dynamite" in result["models"]:
        dyn_data, _ = load_sweep(base_dir, sweep_name, task, "dynamite", seeds)
        dyn_reward = extract_metric_array(dyn_data, "episode_reward/mean")
        dyn_seed_means = np.nanmean(dyn_reward, axis=1)

        comparisons = []
        p_values_raw = []
        for other in ["lstm", "transformer", "mlp"]:
            if other not in result["models"]:
                continue
            other_data, _ = load_sweep(base_dir, sweep_name, task, other, seeds)
            other_reward = extract_metric_array(other_data, "episode_reward/mean")
            other_seed_means = np.nanmean(other_reward, axis=1)

            n = min(len(dyn_seed_means), len(other_seed_means))
            if n < 3:
                continue

            t_stat, p_val = stats.ttest_rel(dyn_seed_means[:n], other_seed_means[:n])
            d = cohens_d(dyn_seed_means[:n], other_seed_means[:n])
            comparisons.append({
                "comparison": f"DynaMITE_vs_{other}",
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "mean_diff": float(np.mean(dyn_seed_means[:n]) - np.mean(other_seed_means[:n])),
            })
            p_values_raw.append(float(p_val))

        # Apply Holm-Bonferroni correction
        if p_values_raw:
            corrections = holm_bonferroni(p_values_raw)
            for comp, (orig_p, adj_p, reject) in zip(comparisons, corrections):
                comp["p_adjusted"] = adj_p
                comp["reject_h0"] = reject

        result["comparisons"] = comparisons

    return result


def print_sweep_table(analysis, metric_key="episode_reward_mean"):
    """Print a markdown table for one sweep."""
    sweep = analysis["sweep"]
    task = analysis["task"]

    # Get values from first model that has data
    ref_model = next((m for m in MODELS if m in analysis["models"]), None)
    if ref_model is None:
        return
    ref = analysis["models"][ref_model]
    values = ref["values"]

    # Format column headers
    def fmt_val(v):
        if isinstance(v, list):
            return str(v[0]) if v[0] == v[1] else f"{v[0]}-{v[1]}"
        return str(v)

    col_labels = [fmt_val(v) for v in values]

    mdata = metric_key
    metric_label = METRIC_LABELS.get(metric_key.replace("_", "/"), metric_key)

    print(f"\n#### {sweep} ({task}) — {metric_label}")
    header = "| Model | " + " | ".join(col_labels) + " | Sens | Severe | AUC |"
    sep = "|" + "---|" * (len(col_labels) + 4)
    print(header)
    print(sep)

    for model in MODELS:
        if model not in analysis["models"]:
            continue
        md = analysis["models"][model].get(mdata, None)
        if md is None:
            continue
        cells = []
        for i in range(len(col_labels)):
            m = md["per_level_mean"][i]
            s = md["per_level_std"][i]
            cells.append(f"{m:.2f}±{s:.2f}")
        sens = md["sensitivity"]
        severe = md["severe_mean"]
        auc = md["auc"]
        auc_str = f"{auc:.2f}" if not np.isnan(auc) else "—"
        row = f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + f" | {sens:.2f} | {severe:.2f} | {auc_str} |"
        print(row)


def main():
    base_dir = Path("results/ood_v2")
    out_dir = Path("results/aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_analyses = {}

    # Randomized task sweeps
    print("=" * 80)
    print("OOD ANALYSIS v2 — Full Behavioral Metrics")
    print("=" * 80)

    for sweep in SWEEPS_RANDOMIZED:
        analysis = analyze_sweep(base_dir, sweep, "randomized", MODELS, SEEDS)
        if analysis["models"]:
            key = f"{sweep}_randomized"
            all_analyses[key] = analysis
            print_sweep_table(analysis, "episode_reward_mean")
            print_sweep_table(analysis, "failure_rate")
            print_sweep_table(analysis, "tracking_error_mean")

    # Cross-task sweeps
    for task, sweeps in SWEEPS_CROSS_TASK.items():
        for sweep in sweeps:
            analysis = analyze_sweep(base_dir, sweep, task, MODELS, SEEDS)
            if analysis["models"]:
                key = f"{sweep}_{task}"
                all_analyses[key] = analysis
                print_sweep_table(analysis, "episode_reward_mean")
                print_sweep_table(analysis, "failure_rate")

    # Print overall robustness summary
    print("\n" + "=" * 80)
    print("OVERALL ROBUSTNESS SUMMARY")
    print("=" * 80)
    print("\n| Model | Avg Reward | Worst Case | Max Sensitivity | Max Fail Rate | "
          "Mean Track Err |")
    print("|---|---|---|---|---|---|")

    for model in MODELS:
        all_rewards = []
        worst_reward = float("inf")
        max_sens = 0
        max_fail = 0
        all_track = []

        for key, analysis in all_analyses.items():
            if model not in analysis["models"]:
                continue
            md = analysis["models"][model]
            rew = md.get("episode_reward_mean", {})
            fail = md.get("failure_rate", {})
            track = md.get("tracking_error_mean", {})

            if rew:
                all_rewards.append(rew["overall_mean"])
                worst_reward = min(worst_reward, rew["worst_level"])
                max_sens = max(max_sens, rew["sensitivity"])
            if fail:
                max_fail = max(max_fail, fail.get("worst_level", 0))
            if track:
                all_track.append(track["overall_mean"])

        if not all_rewards:
            continue
        avg_rew = np.mean(all_rewards)
        avg_track = np.mean(all_track) if all_track else float("nan")
        track_str = f"{avg_track:.2f}" if not np.isnan(avg_track) else "—"

        print(f"| {MODEL_LABELS[model]} | {avg_rew:.2f} | {worst_reward:.2f} | "
              f"{max_sens:.2f} | {max_fail:.2f} | {track_str} |")

    # Print statistical comparisons
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS (DynaMITE vs others, Holm-Bonferroni corrected)")
    print("=" * 80)
    print("\n| Sweep (task) | vs Model | Mean Diff | Cohen's d | p (raw) | p (adj) | Sig? |")
    print("|---|---|---|---|---|---|---|")

    for key, analysis in all_analyses.items():
        comps = analysis.get("comparisons", [])
        for c in comps:
            sig = "**Yes**" if c.get("reject_h0", False) else "No"
            other = c["comparison"].replace("DynaMITE_vs_", "")
            p_adj = c.get("p_adjusted", c["p_value"])
            print(f"| {key} | {other} | {c['mean_diff']:+.2f} | "
                  f"{c['cohens_d']:.2f} | {c['p_value']:.4f} | {p_adj:.4f} | {sig} |")

    # Save JSON
    # Convert numpy types
    def jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [jsonable(v) for v in obj]
        return obj

    out_path = out_dir / "ood_analysis_v2.json"
    with open(out_path, "w") as f:
        json.dump(jsonable(all_analyses), f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
