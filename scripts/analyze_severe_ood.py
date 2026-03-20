#!/usr/bin/env python3
"""
Severe OOD Analysis & Pareto Plot.

Computes the metrics that matter for honest OOD reporting:
  - Severe-level mean (worst 2 levels of each sweep)
  - Worst-case reward
  - AUC (area under reward-vs-perturbation curve)
  - ID-to-OOD degradation ratio
  - Crossover point (where LSTM loses to DynaMITE)
  - Normalized robust performance (OOD/ID ratio, worst/ID ratio)

Generates:
  - Pareto plot: ID reward vs severe OOD reward
  - Combined-shift degradation figure (main benchmark figure)
  - Degradation ratio table

Usage:
    python scripts/analyze_severe_ood.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_data():
    """Load OOD analysis and main comparison data."""
    ood = json.load(open("results/aggregated/ood_analysis_v2.json"))
    main = json.load(open("results/aggregated/main_comparison.json"))
    return ood, main


def compute_severe_metrics(ood_data, main_data):
    """Compute severe OOD metrics for each model × sweep."""
    models = ["dynamite", "lstm", "transformer", "mlp"]
    model_labels = {"dynamite": "DynaMITE", "lstm": "LSTM",
                    "transformer": "Transformer", "mlp": "MLP"}

    # Focus sweeps (excluding action delay which is uninformative)
    focus_sweeps = [
        "combined_shift_randomized",
        "push_magnitude_randomized",
        "push_magnitude_push",
        "push_magnitude_terrain",
        "friction_randomized",
    ]

    # ID rewards per task
    id_rewards = {}
    for model in models:
        id_rewards[model] = {}
        for task in main_data:
            id_rewards[model][task] = main_data[task][model]["mean"]

    results = {}
    for sweep_key in focus_sweeps:
        if sweep_key not in ood_data:
            continue

        sweep_info = ood_data[sweep_key]
        task = sweep_info["task"]
        sweep_name = sweep_info["sweep"]

        sweep_results = {}
        for model in models:
            md = sweep_info["models"][model]
            rew = md["episode_reward_mean"]
            track = md["tracking_error_mean"]

            id_rew = id_rewards[model].get(task, None)
            if id_rew is None:
                id_rew = rew["per_level_mean"][0]  # Use lowest perturbation as proxy

            severe_mean = rew["severe_mean"]
            worst = rew["worst_level"]
            sensitivity = rew["sensitivity"]
            auc = rew["auc"]

            # Degradation from ID to severe OOD
            degradation = id_rew - severe_mean  # positive = how much worse
            degradation_pct = (degradation / abs(id_rew)) * 100 if id_rew != 0 else 0

            # Normalized: OOD/ID ratio (closer to 1 = better retention)
            retention_ratio = severe_mean / id_rew if id_rew != 0 else 0

            # Worst-case retention
            worst_retention = worst / id_rew if id_rew != 0 else 0

            sweep_results[model] = {
                "id_reward": id_rew,
                "severe_mean": severe_mean,
                "worst_case": worst,
                "sensitivity": sensitivity,
                "auc": auc if not (isinstance(auc, float) and np.isnan(auc)) else None,
                "degradation": degradation,
                "degradation_pct": degradation_pct,
                "retention_ratio": retention_ratio,
                "worst_retention": worst_retention,
                "track_err_severe": track["severe_mean"],
                "track_err_worst": track["worst_level"],
            }

        # Find crossover point (where DynaMITE overtakes LSTM)
        dyn_levels = ood_data[sweep_key]["models"]["dynamite"]["episode_reward_mean"]["per_level_mean"]
        lstm_levels = ood_data[sweep_key]["models"]["lstm"]["episode_reward_mean"]["per_level_mean"]
        x_vals = ood_data[sweep_key]["models"]["dynamite"]["x_values"]

        crossover = None
        for i in range(len(dyn_levels)):
            if dyn_levels[i] > lstm_levels[i]:
                crossover = {"level_index": i, "x_value": x_vals[i] if x_vals else i}
                break

        results[sweep_key] = {
            "task": task,
            "sweep": sweep_name,
            "models": sweep_results,
            "crossover_dynamite_beats_lstm": crossover,
        }

    return results


def compute_aggregate_pareto(severe_results, main_data):
    """Compute aggregate ID reward vs severe OOD reward for Pareto plot."""
    models = ["dynamite", "lstm", "transformer", "mlp"]
    model_labels = {"dynamite": "DynaMITE", "lstm": "LSTM",
                    "transformer": "Transformer", "mlp": "MLP"}

    # Aggregate across randomized-task sweeps (combined_shift + push_magnitude + friction)
    randomized_sweeps = [k for k in severe_results
                         if k.endswith("_randomized") and "delay" not in k]

    aggregate = {}
    for model in models:
        # ID reward = average across all 4 tasks
        id_rew = np.mean([main_data[t][model]["mean"]
                          for t in main_data if model in main_data[t]])

        # Severe OOD = average of severe_mean across randomized sweeps
        severe_rews = [severe_results[s]["models"][model]["severe_mean"]
                       for s in randomized_sweeps
                       if model in severe_results[s]["models"]]
        worst_rews = [severe_results[s]["models"][model]["worst_case"]
                      for s in randomized_sweeps
                      if model in severe_results[s]["models"]]

        aggregate[model] = {
            "id_reward": float(id_rew),
            "severe_ood_mean": float(np.mean(severe_rews)),
            "worst_case": float(np.min(worst_rews)),
            "label": model_labels[model],
        }

    return aggregate


def plot_pareto(aggregate, output_path):
    """Generate Pareto plot: ID reward (x) vs Severe OOD reward (y)."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    colors = {"dynamite": "#2196F3", "lstm": "#4CAF50",
              "transformer": "#FF9800", "mlp": "#F44336"}
    markers = {"dynamite": "D", "lstm": "s", "transformer": "^", "mlp": "o"}

    for model, data in aggregate.items():
        ax.scatter(data["id_reward"], data["severe_ood_mean"],
                   c=colors[model], marker=markers[model], s=150, zorder=5,
                   label=data["label"], edgecolors="black", linewidth=0.8)

    # Add reference line y=x (perfect OOD retention)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5, label="Perfect retention")

    ax.set_xlabel("ID Reward (nominal, higher = better)", fontsize=12)
    ax.set_ylabel("Severe OOD Reward (mean of worst 2 levels)", fontsize=12)
    ax.set_title("ID vs Severe OOD: The DynaMITE-LSTM Tradeoff", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the tradeoff
    d = aggregate["dynamite"]
    l = aggregate["lstm"]
    ax.annotate("", xy=(d["id_reward"], d["severe_ood_mean"]),
                xytext=(l["id_reward"], l["severe_ood_mean"]),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_combined_shift_main(ood_data, output_path):
    """Generate the main combined-shift benchmark figure with degradation."""
    if not HAS_MPL:
        return

    sweep_key = "combined_shift_randomized"
    if sweep_key not in ood_data:
        return

    models_order = ["lstm", "dynamite", "transformer", "mlp"]
    model_labels = {"dynamite": "DynaMITE", "lstm": "LSTM",
                    "transformer": "Transformer", "mlp": "MLP"}
    colors = {"dynamite": "#2196F3", "lstm": "#4CAF50",
              "transformer": "#FF9800", "mlp": "#F44336"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sweep = ood_data[sweep_key]
    x_labels = ["None", "Low", "Med", "High", "Extreme"]

    # Left: Reward curves with crossover annotation
    for model in models_order:
        md = sweep["models"][model]
        means = md["episode_reward_mean"]["per_level_mean"]
        stds = md["episode_reward_mean"]["per_level_std"]
        x = range(len(means))
        ax1.plot(x, means, '-o', color=colors[model], label=model_labels[model],
                 linewidth=2, markersize=6)
        ax1.fill_between(x, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=colors[model])

    # Find and annotate crossover
    dyn = sweep["models"]["dynamite"]["episode_reward_mean"]["per_level_mean"]
    lstm = sweep["models"]["lstm"]["episode_reward_mean"]["per_level_mean"]
    for i in range(len(dyn) - 1):
        if lstm[i] < dyn[i]:
            break
        if lstm[i + 1] < dyn[i + 1] and lstm[i] >= dyn[i]:
            # Interpolate crossover
            frac = (dyn[i] - lstm[i]) / ((lstm[i + 1] - lstm[i]) - (dyn[i + 1] - dyn[i]))
            cross_x = i + frac
            cross_y = lstm[i] + frac * (lstm[i + 1] - lstm[i])
            ax1.axvline(cross_x, color='red', linestyle=':', alpha=0.7)
            ax1.annotate("LSTM ≤ DynaMITE",
                         xy=(cross_x, cross_y), fontsize=9,
                         xytext=(cross_x + 0.3, cross_y + 0.15),
                         arrowprops=dict(arrowstyle="->", color="red"))

    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Combined Perturbation Severity", fontsize=12)
    ax1.set_ylabel("Reward (higher = better)", fontsize=12)
    ax1.set_title("Combined-Shift Stress Test: Reward", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Tracking error
    for model in models_order:
        md = sweep["models"][model]
        means = md["tracking_error_mean"]["per_level_mean"]
        stds = md["tracking_error_mean"]["per_level_std"]
        x = range(len(means))
        ax2.plot(x, means, '-o', color=colors[model], label=model_labels[model],
                 linewidth=2, markersize=6)
        ax2.fill_between(x, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.15, color=colors[model])

    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Combined Perturbation Severity", fontsize=12)
    ax2.set_ylabel("Tracking Error (lower = better)", fontsize=12)
    ax2.set_title("Combined-Shift Stress Test: Tracking Error", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_degradation_bar(severe_results, output_path):
    """Bar chart of reward degradation from ID → severe OOD per model."""
    if not HAS_MPL:
        return

    # Use combined_shift as the headline metric
    sweep_key = "combined_shift_randomized"
    if sweep_key not in severe_results:
        return

    models = ["dynamite", "lstm", "transformer", "mlp"]
    labels = ["DynaMITE", "LSTM", "Transformer", "MLP"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    degs = [severe_results[sweep_key]["models"][m]["degradation"] for m in models]
    id_rews = [severe_results[sweep_key]["models"][m]["id_reward"] for m in models]
    severe_rews = [severe_results[sweep_key]["models"][m]["severe_mean"] for m in models]

    x = range(len(models))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], [-r for r in id_rews], width,
                   label="ID Reward (neg.)", color=colors, alpha=0.4, edgecolor="black")
    bars2 = ax.bar([i + width / 2 for i in x], [-r for r in severe_rews], width,
                   label="Severe OOD (neg.)", color=colors, alpha=0.9, edgecolor="black")

    ax.set_ylabel("Positive Penalty (lower = better)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Combined-Shift: ID vs Severe OOD Penalty", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate degradation
    for i, model in enumerate(models):
        deg_pct = severe_results[sweep_key]["models"][model]["degradation_pct"]
        ax.annotate(f"↓{deg_pct:.0f}%",
                    xy=(i + width / 2, -severe_rews[i]),
                    xytext=(i + width / 2, -severe_rews[i] + 0.15),
                    ha='center', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    ood_data, main_data = load_data()

    print("Computing severe OOD metrics...")
    severe = compute_severe_metrics(ood_data, main_data)

    print("\n" + "=" * 70)
    print("SEVERE OOD SUMMARY")
    print("=" * 70)

    for sweep_key, sdata in severe.items():
        print(f"\n### {sdata['sweep']} ({sdata['task']})")
        print(f"{'Model':<14} {'ID Reward':>10} {'Severe Mean':>12} {'Worst':>10} "
              f"{'Degradation':>12} {'Retention':>10} {'Track Err':>10}")
        print("-" * 80)
        for model in ["dynamite", "lstm", "transformer", "mlp"]:
            m = sdata["models"][model]
            print(f"{model:<14} {m['id_reward']:>10.2f} {m['severe_mean']:>12.2f} "
                  f"{m['worst_case']:>10.2f} {m['degradation']:>10.2f} "
                  f"({m['degradation_pct']:>5.1f}%) {m['retention_ratio']:>8.2f}  "
                  f"{m['track_err_severe']:>8.2f}")

        xover = sdata.get("crossover_dynamite_beats_lstm")
        if xover:
            print(f"  ** DynaMITE overtakes LSTM at level index {xover['level_index']} "
                  f"(x={xover['x_value']})")
        else:
            print(f"  ** LSTM leads DynaMITE at all levels")

    # Aggregate Pareto
    aggregate = compute_aggregate_pareto(severe, main_data)
    print("\n" + "=" * 70)
    print("PARETO SUMMARY (ID vs Severe OOD)")
    print(f"{'Model':<14} {'ID Reward':>10} {'Severe OOD':>12} {'Worst Case':>12}")
    print("-" * 50)
    for model in ["dynamite", "lstm", "transformer", "mlp"]:
        a = aggregate[model]
        print(f"{a['label']:<14} {a['id_reward']:>10.2f} {a['severe_ood_mean']:>12.2f} "
              f"{a['worst_case']:>12.2f}")

    # Save results
    output = {
        "severe_analysis": severe,
        "pareto_aggregate": aggregate,
    }
    out_path = Path("results/aggregated/severe_ood_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Generate figures
    if HAS_MPL:
        print("\nGenerating figures...")
        plot_pareto(aggregate, "figures/pareto_id_vs_ood.png")
        plot_combined_shift_main(ood_data, "figures/combined_shift_main.png")
        plot_degradation_bar(severe, "figures/degradation_bar.png")


if __name__ == "__main__":
    main()
