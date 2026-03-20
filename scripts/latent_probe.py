#!/usr/bin/env python3
"""
Latent Probe Analysis: DynaMITE latent vs LSTM hidden state.

Compares how well a simple linear probe can predict ground-truth dynamics
parameters from:
  1. DynaMITE's 24-d factored latent
  2. LSTM's 128-d hidden state

Protocol:
  1. Run rollouts collecting representations + ground-truth DR params
  2. Train Ridge regression probes for each factor
  3. Report R² for each factor x model

Usage (one model per process due to SimulationApp constraint):
    python scripts/latent_probe.py --model_type dynamite --ckpt <path> --seed 42
    python scripts/latent_probe.py --model_type lstm --ckpt <path> --seed 42

Then aggregate:
    python scripts/latent_probe.py --aggregate --output_dir results/latent_probe
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.utils.history_buffer import HistoryBuffer

# Factor structure: name -> number of GT target dimensions
FACTOR_TARGETS = {
    "friction": 2,   # [static, dynamic]
    "mass": 2,       # [added_mass, com_disp]
    "motor": 2,      # [motor_strength_1, motor_strength_2]
    "contact": 1,    # [restitution]
    "delay": 1,      # [action_delay]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Probe Analysis")
    parser.add_argument("--model_type", type=str, choices=["dynamite", "lstm"],
                        help="Model type to probe")
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="Collection episodes for probe training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/latent_probe")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate saved probe results instead of running")
    return parser.parse_args()


def collect_representations(cfg, model, device, num_episodes, model_type):
    """Run rollouts and collect model representations + GT dynamics params.

    For DynaMITE: collects the 24-d latent z vector
    For LSTM: collects the 128-d hidden state h

    Returns:
        representations: np.array (N, repr_dim)
        dynamics_gt: dict of factor_name -> np.array (N, factor_dim)
    """
    model.eval()
    env = make_env(cfg, device=device)

    uses_history = getattr(model, 'uses_history', False)
    history_buf = None
    if uses_history and model_type == "dynamite":
        task_cfg = cfg["task"]["observation"]
        history_buf = HistoryBuffer(
            num_envs=cfg["task"]["num_envs"],
            history_len=task_cfg["history_len"],
            obs_dim=task_cfg["proprioception_dim"],
            act_dim=task_cfg["action_dim"],
            device=device,
        )

    all_reprs = []
    all_gt = {f: [] for f in FACTOR_TARGETS}

    completed = 0
    episode_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data.get("cmd", torch.zeros(env.num_envs, env.cmd_dim, device=device))
    prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)

    if history_buf is not None:
        history_buf.insert(obs, prev_action)

    # For LSTM, maintain hidden state
    hidden = None
    if model_type == "lstm":
        hidden = model.init_hidden(env.num_envs, device)

    # Collect enough samples: each step yields num_envs samples
    # We want at least num_episodes * ~100 steps total
    target_samples = num_episodes * 100
    max_steps = target_samples // env.num_envs + 500
    step_count = 0

    print(f"  Collecting ~{target_samples} samples ({max_steps} steps x {env.num_envs} envs)...")

    with torch.no_grad():
        while step_count < max_steps:
            step_count += 1
            episode_step += 1

            # Build model input
            model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}

            if model_type == "dynamite":
                if history_buf is not None:
                    o, a, m = history_buf.get()
                    model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})
                output = model(**model_input)
                latent_z = output.get("latent_z", None)
                if latent_z is not None:
                    all_reprs.append(latent_z.cpu().numpy())
                else:
                    step_data = env.step(output["action_mean"])
                    obs = step_data["obs"]
                    cmd = step_data.get("cmd", cmd)
                    prev_action = output["action_mean"]
                    continue

            elif model_type == "lstm":
                model_input["hidden"] = hidden
                output = model(**model_input)
                hidden = output["hidden"]
                # hidden = (h, c), each (num_layers, batch, hidden_dim)
                h_state = hidden[0]  # (num_layers, batch, hidden_dim)
                h_last = h_state[-1]  # (batch, hidden_dim) - last layer
                all_reprs.append(h_last.cpu().numpy())

            action = output["action_mean"]

            # Get ground-truth dynamics params (per-env tensors)
            dynamics = env._dynamics_params
            for factor in FACTOR_TARGETS:
                if factor in dynamics:
                    all_gt[factor].append(dynamics[factor].cpu().numpy())

            # Step environment
            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            # Update history buffer for DynaMITE
            if history_buf is not None:
                history_buf.insert(obs, action)
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if isinstance(reset_ids, torch.Tensor) and len(reset_ids) > 0:
                    history_buf.reset_envs(reset_ids)
                    history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                    history_buf.lengths[reset_ids] = 1

            # Handle resets for LSTM hidden state
            if model_type == "lstm":
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if isinstance(reset_ids, torch.Tensor) and len(reset_ids) > 0:
                    h, c = hidden
                    h[:, reset_ids] = 0
                    c[:, reset_ids] = 0
                    hidden = (h, c)

            prev_action = action

            if step_count % 100 == 0:
                n_samples = step_count * env.num_envs
                print(f"    Step {step_count}/{max_steps}, samples: {n_samples}")

    # Don't close env - main() uses os._exit(0)

    # Stack collected data: each element is (num_envs, repr_dim)
    reprs = np.concatenate(all_reprs, axis=0)
    gt = {}
    for factor in FACTOR_TARGETS:
        if all_gt[factor]:
            gt[factor] = np.concatenate(all_gt[factor], axis=0)

    return reprs, gt


def train_probes(reprs, gt, test_frac=0.2, n_splits=5):
    """Train linear AND non-linear probes to predict each GT factor from representation.

    Uses k-fold cross-validation for robust R2 estimates.
    - Ridge: linear probe (R2_linear)
    - MLP: 2-layer MLP probe (R2_mlp) — fairer for non-linear encodings (e.g. tanh bottleneck)

    Returns:
        dict of factor_name -> {"r2_linear_mean", "r2_linear_std", "r2_mlp_mean", "r2_mlp_std", ...}
    """
    n = reprs.shape[0]
    idx = np.random.permutation(n)
    reprs = reprs[idx]

    results = {}
    for factor, target_dim in FACTOR_TARGETS.items():
        if factor not in gt:
            results[factor] = {
                "r2_linear_mean": float('nan'), "r2_linear_std": float('nan'),
                "r2_mlp_mean": float('nan'), "r2_mlp_std": float('nan'),
                "r2_per_dim_linear": [], "r2_per_dim_mlp": [],
            }
            continue
        y = gt[factor][idx]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_dims = y.shape[1]

        # K-fold cross-validation
        fold_size = n // n_splits
        r2_linear_folds = []
        r2_mlp_folds = []
        r2_per_dim_linear_folds = []
        r2_per_dim_mlp_folds = []

        for fold in range(n_splits):
            test_start = fold * fold_size
            test_end = test_start + fold_size
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n)])

            X_train, X_test = reprs[train_idx], reprs[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- Linear probe (Ridge) ---
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred_lin = ridge.predict(X_test)
            if y_pred_lin.ndim == 1:
                y_pred_lin = y_pred_lin.reshape(-1, 1)
            if y_test.ndim == 1:
                y_test = y_test.reshape(-1, 1)

            r2_lin = float(r2_score(y_test, y_pred_lin))
            r2_linear_folds.append(r2_lin)

            per_dim_lin = []
            for d in range(n_dims):
                r2 = r2_score(y_test[:, d], y_pred_lin[:, d])
                per_dim_lin.append(float(r2))
            r2_per_dim_linear_folds.append(per_dim_lin)

            # --- MLP probe (non-linear) ---
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                learning_rate_init=0.001,
            )
            mlp.fit(X_train, y_train.ravel() if n_dims == 1 else y_train)
            y_pred_mlp = mlp.predict(X_test)
            if y_pred_mlp.ndim == 1:
                y_pred_mlp = y_pred_mlp.reshape(-1, 1)

            r2_mlp = float(r2_score(y_test, y_pred_mlp))
            r2_mlp_folds.append(r2_mlp)

            per_dim_mlp = []
            for d in range(n_dims):
                r2 = r2_score(y_test[:, d], y_pred_mlp[:, d])
                per_dim_mlp.append(float(r2))
            r2_per_dim_mlp_folds.append(per_dim_mlp)

        # Mean across folds
        r2_linear_mean = float(np.mean(r2_linear_folds))
        r2_linear_std = float(np.std(r2_linear_folds))
        r2_mlp_mean = float(np.mean(r2_mlp_folds))
        r2_mlp_std = float(np.std(r2_mlp_folds))

        r2_per_dim_linear_avg = [float(np.mean([f[d] for f in r2_per_dim_linear_folds]))
                                  for d in range(n_dims)]
        r2_per_dim_mlp_avg = [float(np.mean([f[d] for f in r2_per_dim_mlp_folds]))
                               for d in range(n_dims)]

        results[factor] = {
            "r2_linear_mean": r2_linear_mean,
            "r2_linear_std": r2_linear_std,
            "r2_mlp_mean": r2_mlp_mean,
            "r2_mlp_std": r2_mlp_std,
            "r2_per_dim_linear": r2_per_dim_linear_avg,
            "r2_per_dim_mlp": r2_per_dim_mlp_avg,
            "r2_linear_folds": r2_linear_folds,
            "r2_mlp_folds": r2_mlp_folds,
        }

    return results


def aggregate_results(output_dir):
    """Aggregate probe results across seeds and models."""
    output_dir = Path(output_dir)
    result_files = sorted(output_dir.glob("probe_*.json"))

    if not result_files:
        print(f"No probe result files found in {output_dir}")
        return None

    # Organize by model_type -> seed -> results
    by_model = {}
    for f in result_files:
        if f.name == "probe_aggregate.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        model_type = data["model_type"]
        seed = data["seed"]
        if model_type not in by_model:
            by_model[model_type] = {}
        by_model[model_type][seed] = data["probe_r2"]

    factors = list(FACTOR_TARGETS.keys())

    # Print comparison table
    print("\n" + "=" * 70)
    print("Latent Probe R2 Comparison: DynaMITE (24-d) vs LSTM (128-d)")
    print("=" * 70)

    # Aggregate across seeds for each probe type
    aggregate = {}

    for probe_type in ["linear", "mlp"]:
        key = f"r2_{probe_type}_mean"
        print(f"\n--- {probe_type.upper()} Probe ---")
        print(f"{'Factor':<12} ", end="")
        for mt in sorted(by_model.keys()):
            print(f"  {mt:>20}", end="")
        print()
        print("-" * 55)

        for factor in factors:
            print(f"{factor:<12} ", end="")
            for mt in sorted(by_model.keys()):
                seeds = by_model[mt]
                vals = [seeds[s].get(factor, {}).get(key, float("nan")) for s in seeds]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    mean_r2 = np.mean(vals)
                    std_r2 = np.std(vals)
                    print(f"  {mean_r2:>6.3f} +/- {std_r2:.3f}   ", end="")
                    if mt not in aggregate:
                        aggregate[mt] = {}
                    if probe_type not in aggregate[mt]:
                        aggregate[mt][probe_type] = {}
                    aggregate[mt][probe_type][factor] = {"mean": float(mean_r2), "std": float(std_r2)}
                else:
                    print(f"  {'N/A':>15}   ", end="")
            print()

        # Overall
        print(f"{'OVERALL':<12} ", end="")
        for mt in sorted(by_model.keys()):
            if mt in aggregate and probe_type in aggregate[mt]:
                all_means = [aggregate[mt][probe_type][f]["mean"]
                            for f in factors if f in aggregate[mt][probe_type]]
                overall = np.mean(all_means)
                print(f"  {overall:>6.3f}             ", end="")
                aggregate[mt][probe_type]["overall"] = float(overall)
        print()

    # Winner comparison
    print("\n" + "=" * 70)
    print("Per-factor winner (MLP probe):")
    for factor in factors:
        d_r2 = aggregate.get("dynamite", {}).get("mlp", {}).get(factor, {}).get("mean", float("nan"))
        l_r2 = aggregate.get("lstm", {}).get("mlp", {}).get(factor, {}).get("mean", float("nan"))
        if not np.isnan(d_r2) and not np.isnan(l_r2):
            winner = "DynaMITE" if d_r2 > l_r2 else "LSTM"
            diff = abs(d_r2 - l_r2)
            print(f"  {factor:<12} -> {winner} (by {diff:.3f})")

    # Save aggregate
    agg_file = output_dir / "probe_aggregate.json"
    with open(agg_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved aggregate to: {agg_file}")

    return aggregate


def main():
    args = parse_args()

    if args.aggregate:
        aggregate_results(args.output_dir)
        return

    if not args.model_type or not args.ckpt:
        print("ERROR: --model_type and --ckpt required when not aggregating")
        sys.exit(1)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Probe] Model: {args.model_type}, Seed: {args.seed}")
    print(f"[Probe] Checkpoint: {args.ckpt}")
    print(f"[Probe] Episodes: {args.num_episodes}")

    init_sim(headless=args.headless)

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Force randomized task with DR enabled for varied dynamics
    cfg["task"]["name"] = "randomized"
    cfg["task"]["domain_randomization"]["enabled"] = True
    cfg["task"]["num_envs"] = 32

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    reprs, gt = collect_representations(
        cfg, model, device, args.num_episodes, args.model_type
    )
    print(f"  Collected {reprs.shape[0]} samples, repr_dim={reprs.shape[1]}")

    probe_results = train_probes(reprs, gt)

    # Print results
    print(f"\n  Probe R2 results for {args.model_type} (seed {args.seed}):")
    print(f"    {'Factor':<12} {'Linear R2':<20} {'MLP R2':<20}")
    print(f"    {'-'*52}")
    for factor, res in probe_results.items():
        lin_str = f"{res['r2_linear_mean']:.3f} +/- {res['r2_linear_std']:.3f}"
        mlp_str = f"{res['r2_mlp_mean']:.3f} +/- {res['r2_mlp_std']:.3f}"
        print(f"    {factor:<12} {lin_str:<20} {mlp_str:<20}")

    # Save
    result = {
        "model_type": args.model_type,
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "repr_dim": int(reprs.shape[1]),
        "n_samples": int(reprs.shape[0]),
        "num_episodes": args.num_episodes,
        "probe_r2": probe_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_file = output_dir / f"probe_{args.model_type}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Probe] Saved to: {out_file}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
