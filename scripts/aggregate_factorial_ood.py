#!/usr/bin/env python3
"""Aggregate 2×2 factorial OOD results from combined-shift severe evaluations.

Reads sweep_combined_shift_severe.json files for 4 ablation variants × 10 seeds.
Computes: per-level means, severe mean (avg of L3+L4), 2×2 main effects, paired t-tests.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

BASE_DIR = Path("/home/chyanin/robotpaper")
RESULTS_DIR = BASE_DIR / "results" / "factorial_ood" / "combined_shift_severe"

VARIANTS = ["full", "no_aux_loss", "no_latent", "aux_only"]
SEEDS = list(range(42, 52))

# 2×2 factorial mapping:
#   full       = Bottleneck + Aux
#   no_aux_loss = Bottleneck + No Aux
#   no_latent   = No Bottleneck + No Aux
#   aux_only    = No Bottleneck + Aux
FACTORIAL_LABELS = {
    "full":        ("Bottleneck", "Aux"),
    "no_aux_loss": ("Bottleneck", "No Aux"),
    "no_latent":   ("No Bottleneck", "No Aux"),
    "aux_only":    ("No Bottleneck", "Aux"),
}


def load_results():
    """Load all sweep results. Returns dict[variant][seed] = {level3, level4, severe_mean}."""
    data = {}
    for variant in VARIANTS:
        data[variant] = {}
        for seed in SEEDS:
            fpath = RESULTS_DIR / variant / f"seed_{seed}" / "sweep_combined_shift_severe.json"
            if not fpath.exists():
                print(f"WARNING: missing {fpath}")
                continue
            with open(fpath) as f:
                sweep = json.load(f)
            results = sweep["results"]
            # results[0] = level 3, results[1] = level 4
            l3 = results[0]["episode_reward/mean"]
            l4 = results[1]["episode_reward/mean"]
            severe = (l3 + l4) / 2.0
            data[variant][seed] = {
                "level3": l3,
                "level4": l4,
                "severe_mean": severe,
                "level3_tracking": results[0].get("tracking_error/mean", None),
                "level4_tracking": results[1].get("tracking_error/mean", None),
                "level3_failure": results[0].get("failure_rate", None),
                "level4_failure": results[1].get("failure_rate", None),
            }
    return data


def compute_stats(data, metric="severe_mean"):
    """Compute mean ± std for each variant."""
    stats_dict = {}
    for variant in VARIANTS:
        vals = [data[variant][s][metric] for s in SEEDS if s in data[variant]]
        stats_dict[variant] = {
            "mean": np.mean(vals),
            "std": np.std(vals, ddof=1),
            "values": vals,
            "n": len(vals),
        }
    return stats_dict


def paired_ttest(vals_a, vals_b):
    """Paired t-test, return t-stat and p-value."""
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return t_stat, p_val


def compute_factorial_effects(data, metric="severe_mean"):
    """Compute 2×2 main effects and interaction.

    Cells:
        full       (B+, A+)
        no_aux_loss (B+, A-)
        no_latent   (B-, A-)
        aux_only    (B-, A+)

    Main effect of Bottleneck = mean(B+) - mean(B-)
        = [mean(full) + mean(no_aux_loss)] / 2 - [mean(no_latent) + mean(aux_only)] / 2
    Main effect of Aux = mean(A+) - mean(A-)
        = [mean(full) + mean(aux_only)] / 2 - [mean(no_aux_loss) + mean(no_latent)] / 2
    Interaction = [full - no_aux_loss] - [aux_only - no_latent]
    """
    full_vals = np.array([data["full"][s][metric] for s in SEEDS if s in data["full"]])
    noaux_vals = np.array([data["no_aux_loss"][s][metric] for s in SEEDS if s in data["no_aux_loss"]])
    nolat_vals = np.array([data["no_latent"][s][metric] for s in SEEDS if s in data["no_latent"]])
    auxonly_vals = np.array([data["aux_only"][s][metric] for s in SEEDS if s in data["aux_only"]])

    # Per-seed effects (for paired analysis)
    n = min(len(full_vals), len(noaux_vals), len(nolat_vals), len(auxonly_vals))

    # Bottleneck effect (higher = better, since rewards are negative penalties)
    # B+ cells: full, no_aux_loss  |  B- cells: no_latent, aux_only
    bottleneck_per_seed = ((full_vals[:n] + noaux_vals[:n]) / 2) - ((nolat_vals[:n] + auxonly_vals[:n]) / 2)
    bottleneck_effect = np.mean(bottleneck_per_seed)
    _, bottleneck_p = stats.ttest_1samp(bottleneck_per_seed, 0)

    # Aux effect
    # A+ cells: full, aux_only  |  A- cells: no_aux_loss, no_latent
    aux_per_seed = ((full_vals[:n] + auxonly_vals[:n]) / 2) - ((noaux_vals[:n] + nolat_vals[:n]) / 2)
    aux_effect = np.mean(aux_per_seed)
    _, aux_p = stats.ttest_1samp(aux_per_seed, 0)

    # Interaction: (full - no_aux_loss) - (aux_only - no_latent)
    interaction_per_seed = (full_vals[:n] - noaux_vals[:n]) - (auxonly_vals[:n] - nolat_vals[:n])
    interaction_effect = np.mean(interaction_per_seed)
    _, interaction_p = stats.ttest_1samp(interaction_per_seed, 0)

    return {
        "bottleneck": {"effect": bottleneck_effect, "p": bottleneck_p, "per_seed": bottleneck_per_seed},
        "aux": {"effect": aux_effect, "p": aux_p, "per_seed": aux_per_seed},
        "interaction": {"effect": interaction_effect, "p": interaction_p, "per_seed": interaction_per_seed},
    }


def main():
    data = load_results()

    # Count results
    total = sum(len(data[v]) for v in VARIANTS)
    print(f"Loaded {total} sweep results ({len(VARIANTS)} variants × seeds)\n")

    print("=" * 72)
    print("  2×2 Factorial OOD Results — Combined-Shift Severe (Levels 3 & 4)")
    print("=" * 72)

    # Per-variant stats for each metric
    for metric_name, metric_key in [("Level 3 Reward", "level3"), ("Level 4 Reward", "level4"),
                                     ("Severe Mean (L3+L4)/2", "severe_mean"),
                                     ("Level 3 Tracking Error", "level3_tracking"),
                                     ("Level 4 Tracking Error", "level4_tracking")]:
        print(f"\n--- {metric_name} ---")
        s = compute_stats(data, metric_key)
        for variant in VARIANTS:
            label = f"{FACTORIAL_LABELS[variant][0]:15s} + {FACTORIAL_LABELS[variant][1]:6s}"
            print(f"  {variant:15s} ({label}): {s[variant]['mean']:7.4f} ± {s[variant]['std']:.4f}  (n={s[variant]['n']})")

    # Paired t-tests vs Full
    print(f"\n--- Paired t-tests vs Full (Severe Mean) ---")
    s = compute_stats(data, "severe_mean")
    full_vals = s["full"]["values"]
    for variant in ["no_aux_loss", "no_latent", "aux_only"]:
        vals = s[variant]["values"]
        delta = np.mean(vals) - np.mean(full_vals)
        t_stat, p_val = paired_ttest(full_vals, vals)
        n_worse = sum(1 for a, b in zip(full_vals, vals) if b < a)
        print(f"  {variant:15s}: Δ={delta:+.4f}  t={t_stat:.3f}  p={p_val:.4f}  {n_worse}/10 worse")

    # 2×2 Factorial effects
    for metric_name, metric_key in [("Severe Mean (L3+L4)/2", "severe_mean"),
                                     ("Level 3 Reward", "level3"),
                                     ("Level 4 Reward", "level4")]:
        print(f"\n--- 2×2 Factorial Effects: {metric_name} ---")
        effects = compute_factorial_effects(data, metric_key)
        print(f"  Bottleneck effect: {effects['bottleneck']['effect']:+.4f}  (p={effects['bottleneck']['p']:.4f})")
        print(f"  Aux loss effect:   {effects['aux']['effect']:+.4f}  (p={effects['aux']['p']:.4f})")
        print(f"  Interaction:       {effects['interaction']['effect']:+.4f}  (p={effects['interaction']['p']:.4f})")

    # 2×2 Table format
    print(f"\n--- 2×2 Table (Severe Mean) ---")
    s = compute_stats(data, "severe_mean")
    print(f"                  | No Aux Loss         | Aux Loss            |")
    print(f"  Bottleneck      | {s['no_aux_loss']['mean']:.2f} ± {s['no_aux_loss']['std']:.2f}        | {s['full']['mean']:.2f} ± {s['full']['std']:.2f}        |")
    print(f"  No Bottleneck   | {s['no_latent']['mean']:.2f} ± {s['no_latent']['std']:.2f}        | {s['aux_only']['mean']:.2f} ± {s['aux_only']['std']:.2f}        |")

    # Degradation from ID (using known ID means)
    print(f"\n--- OOD Degradation from ID baseline ---")
    ID_MEANS = {
        "full": -4.4568,
        "no_aux_loss": -4.5065,
        "no_latent": -4.6398,
        "aux_only": -4.6353,
    }
    for variant in VARIANTS:
        ood_mean = s[variant]["mean"]
        id_mean = ID_MEANS[variant]
        degradation = ood_mean - id_mean
        pct_deg = (degradation / abs(id_mean)) * 100
        print(f"  {variant:15s}: ID={id_mean:.4f}  OOD={ood_mean:.4f}  Δ={degradation:+.4f}  ({pct_deg:+.1f}%)")

    # Per-seed data dump for reference
    print(f"\n--- Per-seed Severe Mean ---")
    for variant in VARIANTS:
        vals = [f"{data[variant][s]['severe_mean']:.2f}" for s in SEEDS if s in data[variant]]
        print(f"  {variant:15s}: {', '.join(vals)}")

    # Save aggregated results as JSON
    output = {
        "description": "2x2 Factorial OOD — Combined-Shift Severe (Levels 3 & 4)",
        "variants": {},
        "factorial_effects": {},
    }
    for metric_key in ["level3", "level4", "severe_mean"]:
        s = compute_stats(data, metric_key)
        effects = compute_factorial_effects(data, metric_key)
        for variant in VARIANTS:
            if variant not in output["variants"]:
                output["variants"][variant] = {"label": FACTORIAL_LABELS[variant]}
            output["variants"][variant][metric_key] = {
                "mean": round(s[variant]["mean"], 4),
                "std": round(s[variant]["std"], 4),
                "values": [round(v, 4) for v in s[variant]["values"]],
            }
        output["factorial_effects"][metric_key] = {
            "bottleneck": {"effect": round(effects["bottleneck"]["effect"], 4), "p": round(effects["bottleneck"]["p"], 4)},
            "aux": {"effect": round(effects["aux"]["effect"], 4), "p": round(effects["aux"]["p"], 4)},
            "interaction": {"effect": round(effects["interaction"]["effect"], 4), "p": round(effects["interaction"]["p"], 4)},
        }

    out_path = RESULTS_DIR / "factorial_ood_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
