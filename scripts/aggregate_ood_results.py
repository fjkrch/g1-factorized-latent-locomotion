#!/usr/bin/env python3
"""
Aggregate OOD sweep results from results/sweeps_multiseed/.

Produces:
  1. results/aggregated/ood_summary.json — full data for all models/sweeps
  2. Console markdown tables suitable for README
  3. Sensitivity analysis (reward drop from nominal to worst)

Expected layout:
  results/sweeps_multiseed/{sweep_type}/{model}_seed{N}/sweep_{sweep_type}.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required")
    sys.exit(1)


MODELS = ["dynamite", "lstm", "transformer", "mlp"]
MODEL_LABELS = {
    "dynamite": "DynaMITE (Ours)",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "mlp": "MLP",
}
SEEDS = [42, 43, 44, 45, 46]
SWEEP_TYPES = ["friction", "push_magnitude", "action_delay"]

SWEEP_DISPLAY = {
    "friction": "Friction",
    "push_magnitude": "Push Magnitude",
    "action_delay": "Action Delay",
}


def extract_x(values):
    """Convert sweep values (may be ranges) to x-axis values."""
    x = []
    for v in values:
        if isinstance(v, list):
            x.append(v[0] if v[0] == v[1] else (v[0] + v[1]) / 2)
        else:
            x.append(v)
    return x


def load_sweep_data(base_dir: Path, sweep_type: str, model: str, seeds: list):
    """Load all seed results for a model/sweep combination."""
    all_means = []
    found_seeds = []
    x_vals = None
    raw_values = None

    for seed in seeds:
        path = base_dir / sweep_type / f"{model}_seed{seed}" / f"sweep_{sweep_type}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        means = [r["episode_reward/mean"] for r in data["results"]]
        all_means.append(means)
        found_seeds.append(seed)
        if x_vals is None:
            x_vals = extract_x(data["values"])
            raw_values = data["values"]

    if not all_means:
        return None

    arr = np.array(all_means)
    n = arr.shape[0]
    return {
        "model": model,
        "sweep": sweep_type,
        "n_seeds": n,
        "seeds": found_seeds,
        "x_values": x_vals,
        "raw_values": raw_values,
        "per_level_mean": np.mean(arr, axis=0).tolist(),
        "per_level_std": (np.std(arr, axis=0, ddof=1) if n > 1 else np.zeros(arr.shape[1])).tolist(),
        "overall_mean": float(np.mean(arr)),
        "overall_std": float(np.std(np.mean(arr, axis=1), ddof=1)) if n > 1 else 0.0,
        "nominal_mean": float(np.mean(arr[:, 0])),  # first level = nominal
        "nominal_std": float(np.std(arr[:, 0], ddof=1)) if n > 1 else 0.0,
        "worst_level_mean": float(np.min(np.mean(arr, axis=0))),
        "worst_level_idx": int(np.argmin(np.mean(arr, axis=0))),
    }


def sensitivity(data):
    """Compute reward drop from nominal (level 0) to worst level."""
    if data is None:
        return None
    drop = data["nominal_mean"] - data["worst_level_mean"]
    pct = (drop / abs(data["nominal_mean"]) * 100) if data["nominal_mean"] != 0 else 0
    return {"absolute_drop": drop, "percentage_drop": pct}


def main():
    base_dir = Path("results/sweeps_multiseed")
    out_dir = Path("results/aggregated")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_results = {}
    for sweep_type in SWEEP_TYPES:
        for model in MODELS:
            data = load_sweep_data(base_dir, sweep_type, model, SEEDS)
            if data is not None:
                all_results[f"{model}_{sweep_type}"] = data

    if not all_results:
        print("ERROR: No sweep results found")
        sys.exit(1)

    # Save full JSON
    out_path = out_dir / "ood_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    # Print markdown tables
    print("\n" + "=" * 80)
    print("OOD SWEEP RESULTS (Mean ± Std across seeds)")
    print("=" * 80)

    for sweep_type in SWEEP_TYPES:
        # Get x values from first available model
        ref_key = next((k for k in all_results if k.endswith(f"_{sweep_type}")), None)
        if ref_key is None:
            continue
        ref = all_results[ref_key]
        x_vals = ref["x_values"]
        raw_vals = ref["raw_values"]

        # Format header
        x_labels = []
        for rv in raw_vals:
            if isinstance(rv, list):
                if rv[0] == rv[1]:
                    x_labels.append(str(rv[0]))
                else:
                    x_labels.append(f"{rv[0]}-{rv[1]}")
            else:
                x_labels.append(str(rv))

        print(f"\n### {SWEEP_DISPLAY[sweep_type]} Sweep")
        header = "| Model | " + " | ".join(x_labels) + " | Sensitivity |"
        sep = "|" + "---|" * (len(x_labels) + 2)
        print(header)
        print(sep)

        for model in MODELS:
            key = f"{model}_{sweep_type}"
            if key not in all_results:
                continue
            d = all_results[key]
            sens = sensitivity(d)
            cells = []
            for i in range(len(x_labels)):
                m = d["per_level_mean"][i]
                s = d["per_level_std"][i]
                cells.append(f"{m:.1f}±{s:.1f}")

            sens_str = f"{sens['absolute_drop']:.1f} ({sens['percentage_drop']:.1f}%)" if sens else "N/A"
            row = f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + f" | {sens_str} |"
            print(row)

    # Summary table: overall robustness ranking
    print(f"\n### Overall Robustness Summary")
    print("| Model | Seeds | Avg Reward (all OOD) | Worst Case | Nominal | Max Sensitivity |")
    print("|---|---|---|---|---|---|")
    for model in MODELS:
        model_data = [all_results[k] for k in all_results if k.startswith(f"{model}_")]
        if not model_data:
            continue
        n_seeds = model_data[0]["n_seeds"]
        all_overall = [d["overall_mean"] for d in model_data]
        all_worst = [d["worst_level_mean"] for d in model_data]
        all_nominal = [d["nominal_mean"] for d in model_data]
        all_sens = [sensitivity(d)["absolute_drop"] for d in model_data]

        print(f"| {MODEL_LABELS[model]} | {n_seeds} | "
              f"{np.mean(all_overall):.1f} | "
              f"{min(all_worst):.1f} | "
              f"{np.mean(all_nominal):.1f} | "
              f"{max(all_sens):.1f} |")

    print(f"\nTotal sweep evaluations: {sum(d['n_seeds'] for d in all_results.values())}")


if __name__ == "__main__":
    main()
