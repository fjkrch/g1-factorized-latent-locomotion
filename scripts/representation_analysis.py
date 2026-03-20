#!/usr/bin/env python3
"""
Analysis 2 + 3: Representation Geometry and Mutual Information (MINE).

Collects representations from DynaMITE (24-d latent) or LSTM (128-d hidden
state) and computes:
  - Effective rank, condition number, participation ratio, PCA variance curve
  - Mutual information via MINE estimator (with KNN fallback)

One process per model/seed pair (SimulationApp constraint).

Usage:
    # Per model/seed:
    python scripts/representation_analysis.py --model_type dynamite --ckpt <path> --seed 42
    python scripts/representation_analysis.py --model_type lstm --ckpt <path> --seed 42

    # Aggregate all results:
    python scripts/representation_analysis.py --aggregate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.utils.history_buffer import HistoryBuffer

FACTOR_TARGETS = {
    "friction": 2,
    "mass": 2,
    "motor": 2,
    "contact": 1,
    "delay": 1,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Collection (reused from latent_probe.py)
# ═══════════════════════════════════════════════════════════════════════════

def collect_representations(cfg, model, device, num_episodes, model_type):
    """Collect representations and GT dynamics params via rollouts."""
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

    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data.get("cmd", torch.zeros(env.num_envs, env.cmd_dim, device=device))
    prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)

    if history_buf is not None:
        history_buf.insert(obs, prev_action)

    hidden = None
    if model_type == "lstm":
        hidden = model.init_hidden(env.num_envs, device)

    target_samples = num_episodes * 100
    max_steps = target_samples // env.num_envs + 500
    step_count = 0

    print(f"  Collecting ~{target_samples} samples ({max_steps} steps x {env.num_envs} envs)...")

    with torch.no_grad():
        while step_count < max_steps:
            step_count += 1
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
                h_state = hidden[0]
                h_last = h_state[-1]
                all_reprs.append(h_last.cpu().numpy())

            action = output["action_mean"]

            dynamics = env._dynamics_params
            for factor in FACTOR_TARGETS:
                if factor in dynamics:
                    all_gt[factor].append(dynamics[factor].cpu().numpy())

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            if history_buf is not None:
                history_buf.insert(obs, action)
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if isinstance(reset_ids, torch.Tensor) and len(reset_ids) > 0:
                    history_buf.reset_envs(reset_ids)
                    history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                    history_buf.lengths[reset_ids] = 1

            if model_type == "lstm":
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if isinstance(reset_ids, torch.Tensor) and len(reset_ids) > 0:
                    h, c = hidden
                    h[:, reset_ids] = 0
                    c[:, reset_ids] = 0
                    hidden = (h, c)

            prev_action = action

            if step_count % 200 == 0:
                n_samples = step_count * env.num_envs
                print(f"    Step {step_count}/{max_steps}, samples: {n_samples}")

    reprs = np.concatenate(all_reprs, axis=0)
    gt = {}
    for factor in FACTOR_TARGETS:
        if all_gt[factor]:
            gt[factor] = np.concatenate(all_gt[factor], axis=0)

    return reprs, gt


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: Representation Geometry
# ═══════════════════════════════════════════════════════════════════════════

def compute_geometry(reprs):
    """Compute representation geometry metrics from collected representations."""
    reprs_centered = reprs - reprs.mean(axis=0)
    n, d = reprs_centered.shape

    # SVD
    U, sigma, Vt = np.linalg.svd(reprs_centered, full_matrices=False)

    # 1. Effective rank = (sum sigma)^2 / sum(sigma^2)
    sigma_sum = sigma.sum()
    sigma_sq_sum = (sigma ** 2).sum()
    effective_rank = float((sigma_sum ** 2) / (sigma_sq_sum + 1e-10))

    # 2. Condition number = sigma_max / sigma_min
    condition_number = float(sigma[0] / (sigma[-1] + 1e-10))

    # 3. Participation ratio via eigenvalues of covariance
    eigenvalues = (sigma ** 2) / (n - 1)
    eig_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
    participation_ratio = float(1.0 / ((eig_norm ** 2).sum() + 1e-10))

    # 4. Cumulative explained variance
    explained_variance = eigenvalues / (eigenvalues.sum() + 1e-10)
    cumulative_variance = np.cumsum(explained_variance)

    return {
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "participation_ratio": participation_ratio,
        "singular_values": sigma.tolist(),
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "repr_dim": d,
        "n_samples": n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3: MINE (Mutual Information Neural Estimation)
# ═══════════════════════════════════════════════════════════════════════════

class MINECritic(nn.Module):
    """3-layer MLP critic for MINE estimation."""
    def __init__(self, x_dim, y_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


def estimate_mine(
    x_np, y_np, steps=5000, lr=1e-4, batch_size=1024, ema_alpha=0.01
):
    """
    Estimate MI(X; Y) using MINE with EMA baseline.

    Returns:
        mi_estimate: float
        mi_history: list of per-step MI estimates
        stable: bool
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    x_dim, y_dim = x.shape[1], y.shape[1]
    n = len(x)

    critic = MINECritic(x_dim, y_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    running_mean = None
    mi_history = []

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,))
        x_batch = x[idx]
        y_batch = y[idx]

        # Joint
        t_joint = critic(x_batch, y_batch)

        # Marginal (shuffle y)
        y_marginal = y[torch.randint(0, n, (batch_size,))]
        t_marginal = critic(x_batch, y_marginal)

        # EMA baseline
        et = torch.exp(t_marginal)
        if running_mean is None:
            running_mean = et.mean().detach()
        else:
            running_mean = (
                (1 - ema_alpha) * running_mean + ema_alpha * et.mean().detach()
            )

        # MINE objective with EMA variance reduction
        loss = -(t_joint.mean() - (et / running_mean.clamp(min=1e-8)).mean().log())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()

        with torch.no_grad():
            mi_est = t_joint.mean().item() - torch.log(et.mean() + 1e-8).item()
            mi_history.append(float(mi_est))

    # Stability check
    last_500 = mi_history[-500:] if len(mi_history) >= 500 else mi_history
    mi_mean = np.mean(last_500)
    mi_std = np.std(last_500)
    negative_frac = sum(1 for v in last_500 if v < 0) / len(last_500)
    stable = negative_frac < 0.5 and mi_std < 2.0 * abs(mi_mean) + 0.1

    final_mi = max(0.0, float(mi_mean))
    return final_mi, mi_history, bool(stable)


def knn_mi_fallback(x_np, y_np, n_neighbors=5):
    """
    KNN-based MI estimator fallback using sklearn.
    Estimates I(X; Y) by averaging per-target-dim MI(X; y_d).
    """
    from sklearn.feature_selection import mutual_info_regression

    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    # Subsample if too large for KNN
    n = x_np.shape[0]
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        x_np = x_np[idx]
        y_np = y_np[idx]

    total_mi = 0.0
    for d in range(y_np.shape[1]):
        # MI between each feature of X and y_d, then take max (best single feature)
        mi_vals = mutual_info_regression(
            x_np, y_np[:, d], n_neighbors=n_neighbors, random_state=42
        )
        total_mi += float(mi_vals.max())

    return total_mi


def run_mine_analysis(reprs, gt, model_type, seed):
    """
    Run MINE for all factors + overall.

    Returns dict of factor -> {mi, stable, method}
    """
    results = {}

    # Construct overall target: concatenate all factor targets
    all_targets = []
    factor_order = []
    for fname in FACTOR_TARGETS:
        if fname in gt:
            all_targets.append(gt[fname])
            factor_order.append(fname)
    if all_targets:
        overall_target = np.concatenate(all_targets, axis=1)

        print(f"  MINE overall: repr({reprs.shape[1]}d) vs params({overall_target.shape[1]}d)")
        mi, history, stable = estimate_mine(reprs, overall_target, steps=5000)

        if not stable:
            print(f"    MINE unstable, falling back to KNN estimator")
            mi = knn_mi_fallback(reprs, overall_target)
            method = "knn_fallback"
        else:
            method = "mine"

        results["overall"] = {
            "mi": float(mi),
            "stable": stable,
            "method": method,
        }
        print(f"    Overall MI = {mi:.4f} ({method})")

    # Per-factor MI
    for fname in FACTOR_TARGETS:
        if fname not in gt:
            continue
        y = gt[fname]
        print(f"  MINE factor={fname}: repr({reprs.shape[1]}d) vs target({y.shape[1] if y.ndim > 1 else 1}d)")

        mi, history, stable = estimate_mine(reprs, y, steps=5000)

        if not stable:
            print(f"    MINE unstable for {fname}, falling back to KNN")
            mi = knn_mi_fallback(reprs, y)
            method = "knn_fallback"
        else:
            method = "mine"

        results[fname] = {
            "mi": float(mi),
            "stable": stable,
            "method": method,
        }
        print(f"    MI({fname}) = {mi:.4f} ({method})")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(base_dir):
    """Aggregate geometry and MINE results across seeds and models."""
    base_dir = Path(base_dir)
    geo_dir = base_dir / "geometry_analysis"
    mi_dir = base_dir / "mi_analysis"

    # Aggregate geometry
    geo_files = sorted(geo_dir.glob("geometry_*.json")) if geo_dir.exists() else []
    geo_by_model = {}
    for f in geo_files:
        with open(f) as fh:
            data = json.load(fh)
        mt = data["model_type"]
        if mt not in geo_by_model:
            geo_by_model[mt] = []
        geo_by_model[mt].append(data["geometry"])

    geo_agg = {}
    for mt, runs in geo_by_model.items():
        geo_agg[mt] = {
            "effective_rank": {
                "mean": float(np.mean([r["effective_rank"] for r in runs])),
                "std": float(np.std([r["effective_rank"] for r in runs])),
            },
            "condition_number": {
                "mean": float(np.mean([r["condition_number"] for r in runs])),
                "std": float(np.std([r["condition_number"] for r in runs])),
            },
            "participation_ratio": {
                "mean": float(np.mean([r["participation_ratio"] for r in runs])),
                "std": float(np.std([r["participation_ratio"] for r in runs])),
            },
            "n_seeds": len(runs),
            # Store cumulative variance curves (per-seed, for plotting)
            "cumulative_variance_curves": [r["cumulative_variance"] for r in runs],
        }

    geo_out = geo_dir / "results.json" if geo_dir.exists() else base_dir / "geometry_results.json"
    geo_out.parent.mkdir(parents=True, exist_ok=True)
    with open(geo_out, "w") as f:
        json.dump(geo_agg, f, indent=2)
    print(f"[Aggregate] Geometry results: {geo_out}")

    # Print geometry summary
    print("\n" + "=" * 60)
    print("Geometry Summary (mean ± std across seeds)")
    print("=" * 60)
    for mt in sorted(geo_agg.keys()):
        d = geo_agg[mt]
        print(f"\n  {mt.upper()} (n={d['n_seeds']}):")
        print(f"    Effective rank:      {d['effective_rank']['mean']:.2f} ± {d['effective_rank']['std']:.2f}")
        print(f"    Condition number:    {d['condition_number']['mean']:.1f} ± {d['condition_number']['std']:.1f}")
        print(f"    Participation ratio: {d['participation_ratio']['mean']:.2f} ± {d['participation_ratio']['std']:.2f}")

    # Aggregate MINE
    mi_files = sorted(mi_dir.glob("mine_*.json")) if mi_dir.exists() else []
    mi_by_model = {}
    for f in mi_files:
        with open(f) as fh:
            data = json.load(fh)
        mt = data["model_type"]
        if mt not in mi_by_model:
            mi_by_model[mt] = []
        mi_by_model[mt].append(data["mine"])

    mi_agg = {}
    factors = list(FACTOR_TARGETS.keys()) + ["overall"]
    for mt, runs in mi_by_model.items():
        mi_agg[mt] = {}
        for fname in factors:
            vals = [r[fname]["mi"] for r in runs if fname in r]
            if vals:
                mi_agg[mt][fname] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }

    mi_out = mi_dir / "mine_results.json" if mi_dir.exists() else base_dir / "mine_results.json"
    mi_out.parent.mkdir(parents=True, exist_ok=True)
    with open(mi_out, "w") as f:
        json.dump(mi_agg, f, indent=2)
    print(f"\n[Aggregate] MINE results: {mi_out}")

    # Print MINE summary
    print("\n" + "=" * 60)
    print("MINE Summary (mean ± std across seeds)")
    print("=" * 60)
    print(f"\n{'Factor':<12}", end="")
    for mt in sorted(mi_agg.keys()):
        print(f"  {mt:>20}", end="")
    print()
    print("-" * 55)
    for fname in factors:
        print(f"{fname:<12}", end="")
        for mt in sorted(mi_agg.keys()):
            if fname in mi_agg[mt]:
                d = mi_agg[mt][fname]
                print(f"  {d['mean']:>6.4f} ± {d['std']:.4f}   ", end="")
            else:
                print(f"  {'N/A':>15}   ", end="")
        print()

    return {"geometry": geo_agg, "mine": mi_agg}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Representation Geometry + MINE Analysis")
    parser.add_argument("--model_type", type=str, choices=["dynamite", "lstm"])
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/mechanistic")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


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

    geo_dir = Path(args.output_dir) / "geometry_analysis"
    mi_dir = Path(args.output_dir) / "mi_analysis"
    geo_dir.mkdir(parents=True, exist_ok=True)
    mi_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ReprAnalysis] Model: {args.model_type}, Seed: {args.seed}")
    print(f"[ReprAnalysis] Checkpoint: {args.ckpt}")

    init_sim(headless=args.headless)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["task"]["name"] = "randomized"
    cfg["task"]["domain_randomization"]["enabled"] = True
    cfg["task"]["num_envs"] = 32

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect data
    print(f"\n[ReprAnalysis] Collecting representations...")
    t0 = time.time()
    reprs, gt = collect_representations(
        cfg, model, device, args.num_episodes, args.model_type
    )
    print(f"  Collected {reprs.shape[0]} samples, repr_dim={reprs.shape[1]} in {time.time()-t0:.0f}s")

    # --- Geometry Analysis ---
    print(f"\n[ReprAnalysis] Computing geometry metrics...")
    geometry = compute_geometry(reprs)
    print(f"  Effective rank:      {geometry['effective_rank']:.2f}")
    print(f"  Condition number:    {geometry['condition_number']:.1f}")
    print(f"  Participation ratio: {geometry['participation_ratio']:.2f}")

    geo_result = {
        "model_type": args.model_type,
        "seed": args.seed,
        "checkpoint": str(args.ckpt),
        "geometry": geometry,
        "timestamp": datetime.now().isoformat(),
    }
    geo_file = geo_dir / f"geometry_{args.model_type}_seed{args.seed}.json"
    geo_tmp = geo_file.with_suffix(".tmp")
    json_str = json.dumps(geo_result, indent=2)
    with open(geo_tmp, "w") as f:
        f.write(json_str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(geo_tmp), str(geo_file))
    print(f"  Saved: {geo_file} ({len(json_str)} bytes)")

    # --- MINE Analysis ---
    print(f"\n[ReprAnalysis] Running MINE analysis...")
    t0 = time.time()
    mine_results = run_mine_analysis(reprs, gt, args.model_type, args.seed)
    print(f"  MINE complete in {time.time()-t0:.0f}s")

    mi_result = {
        "model_type": args.model_type,
        "seed": args.seed,
        "checkpoint": str(args.ckpt),
        "mine": mine_results,
        "timestamp": datetime.now().isoformat(),
    }
    mi_file = mi_dir / f"mine_{args.model_type}_seed{args.seed}.json"
    mi_tmp = mi_file.with_suffix(".tmp")
    json_str = json.dumps(mi_result, indent=2)
    with open(mi_tmp, "w") as f:
        f.write(json_str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(mi_tmp), str(mi_file))
    print(f"  Saved: {mi_file} ({len(json_str)} bytes)")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
