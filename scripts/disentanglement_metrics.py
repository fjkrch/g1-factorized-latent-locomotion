#!/usr/bin/env python3
"""
Disentanglement Metrics: MIG, DCI, SAP.

Computes three standard disentanglement metrics on the learned latent
representations of DynaMITE and LSTM baselines:

  - MIG  (Mutual Information Gap)     [Chen et al., 2018]
  - DCI  (Disentanglement/Completeness/Informativeness) [Eastwood & Williams, 2018]
  - SAP  (Separated Attribute Predictability) [Kumar et al., 2018]

Uses the same data collection pipeline as representation_analysis.py.

Usage:
    # Per model/seed:
    python scripts/disentanglement_metrics.py --model_type dynamite --ckpt <path> --seed 42

    # Aggregate all results:
    python scripts/disentanglement_metrics.py --aggregate
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
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression

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
# Data Collection (same as representation_analysis.py)
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
# Utility: discretize continuous factors for MI-based metrics
# ═══════════════════════════════════════════════════════════════════════════

def _discretize(y, n_bins=20):
    """Discretize a 1D continuous array into bins for MI estimation."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(y, percentiles)
    edges[-1] += 1e-8  # avoid last-bin edge case
    return np.digitize(y, edges[1:])


def _flatten_factors(gt):
    """
    Flatten multi-dim factors into list of (name, 1D-array) pairs.

    E.g. friction (N,2) -> [("friction_0", (N,)), ("friction_1", (N,))]
    """
    factors = []
    for fname, arr in gt.items():
        if arr.ndim == 1:
            factors.append((fname, arr))
        else:
            for d in range(arr.shape[1]):
                factors.append((f"{fname}_{d}", arr[:, d]))
    return factors


# ═══════════════════════════════════════════════════════════════════════════
# MIG: Mutual Information Gap  [Chen et al., 2018]
# ═══════════════════════════════════════════════════════════════════════════

def compute_mig(reprs, gt, n_bins=20, n_neighbors=5, subsample=10000):
    """
    Compute Mutual Information Gap (MIG).

    For each ground-truth factor k:
      1. Compute MI(z_j; v_k) for every latent dim j
      2. Sort MI values descending
      3. MIG_k = (MI_top1 - MI_top2) / H(v_k)

    MIG = mean over factors.

    Args:
        reprs: (N, latent_dim) latent representations
        gt: dict factor_name -> (N, factor_dim) ground truth
        n_bins: bins for discretizing factors (for entropy normalization)
        n_neighbors: KNN neighbors for MI estimation
        subsample: max samples to use (MI estimation is O(N^2)-ish)

    Returns:
        dict with 'mig' score and per-factor breakdown
    """
    factors = _flatten_factors(gt)
    if not factors:
        return {"mig": 0.0, "per_factor": {}}

    N = reprs.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        reprs_sub = reprs[idx]
        factors_sub = [(name, arr[idx]) for name, arr in factors]
    else:
        reprs_sub = reprs
        factors_sub = factors

    latent_dim = reprs_sub.shape[1]
    per_factor = {}

    for fname, fvals in factors_sub:
        # Compute MI(z_j; v_k) for each latent dim j
        mi_per_dim = mutual_info_regression(
            reprs_sub, fvals, n_neighbors=n_neighbors, random_state=42
        )
        # mi_per_dim shape: (latent_dim,)

        # Entropy of factor (discretized)
        fvals_disc = _discretize(fvals, n_bins)
        counts = np.bincount(fvals_disc)
        probs = counts[counts > 0] / counts.sum()
        H_k = -np.sum(probs * np.log(probs + 1e-10))

        # Sort MI descending
        sorted_mi = np.sort(mi_per_dim)[::-1]
        gap = sorted_mi[0] - sorted_mi[1] if len(sorted_mi) > 1 else sorted_mi[0]
        mig_k = gap / (H_k + 1e-10)

        per_factor[fname] = {
            "mig": float(mig_k),
            "top1_mi": float(sorted_mi[0]),
            "top2_mi": float(sorted_mi[1]) if len(sorted_mi) > 1 else 0.0,
            "entropy": float(H_k),
            "best_dim": int(np.argmax(mi_per_dim)),
        }

    mig_score = float(np.mean([v["mig"] for v in per_factor.values()]))
    return {"mig": mig_score, "per_factor": per_factor}


# ═══════════════════════════════════════════════════════════════════════════
# DCI: Disentanglement, Completeness, Informativeness
#      [Eastwood & Williams, 2018]
# ═══════════════════════════════════════════════════════════════════════════

def _entropy(probs):
    """Compute entropy of a probability vector."""
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log(probs))


def compute_dci(reprs, gt, subsample=10000):
    """
    Compute DCI (Disentanglement, Completeness, Informativeness).

    1. Train a Lasso regressor from each latent dim to each factor.
    2. Build importance matrix R[i,j] = |coef| of latent dim i for factor j.
    3. Disentanglement: for each latent dim i, how concentrated is its
       importance across factors? (1 - normalized entropy of row i)
    4. Completeness: for each factor j, how concentrated is the importance
       across latent dims? (1 - normalized entropy of col j)
    5. Informativeness: R² of predicting each factor from all latent dims.

    Args:
        reprs: (N, latent_dim)
        gt: dict factor_name -> (N, factor_dim)
        subsample: max samples

    Returns:
        dict with disentanglement, completeness, informativeness, details
    """
    factors = _flatten_factors(gt)
    if not factors:
        return {"disentanglement": 0.0, "completeness": 0.0, "informativeness": 0.0}

    N = reprs.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        reprs_sub = reprs[idx]
        factors_sub = [(name, arr[idx]) for name, arr in factors]
    else:
        reprs_sub = reprs
        factors_sub = factors

    scaler = StandardScaler()
    Z = scaler.fit_transform(reprs_sub)

    latent_dim = Z.shape[1]
    n_factors = len(factors_sub)

    # Build importance matrix R[i,j] using GBT feature importances
    # GBT is more robust than Lasso for non-linear relationships
    R = np.zeros((latent_dim, n_factors))
    informativeness_scores = []

    for j, (fname, fvals) in enumerate(factors_sub):
        # Standardize target
        fvals_std = (fvals - fvals.mean()) / (fvals.std() + 1e-10)

        # Use Gradient Boosting for importance (handles non-linear)
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gbr.fit(Z, fvals_std)
        R[:, j] = gbr.feature_importances_

        # Informativeness = R² on training data (proxy)
        r2 = gbr.score(Z, fvals_std)
        informativeness_scores.append(max(0.0, float(r2)))

    # Disentanglement: for each latent dim i with non-zero importance
    max_entropy = np.log(n_factors) if n_factors > 1 else 1.0
    disentanglement_per_dim = []
    for i in range(latent_dim):
        row = R[i, :]
        total = row.sum()
        if total < 1e-10:
            continue  # skip dims with no importance
        probs = row / total
        h = _entropy(probs)
        d_i = 1.0 - h / (max_entropy + 1e-10)
        disentanglement_per_dim.append((d_i, total))

    # Weighted average by total importance
    if disentanglement_per_dim:
        weights = np.array([w for _, w in disentanglement_per_dim])
        scores = np.array([d for d, _ in disentanglement_per_dim])
        disentanglement = float(np.average(scores, weights=weights))
    else:
        disentanglement = 0.0

    # Completeness: for each factor j
    max_entropy_c = np.log(latent_dim) if latent_dim > 1 else 1.0
    completeness_per_factor = []
    for j in range(n_factors):
        col = R[:, j]
        total = col.sum()
        if total < 1e-10:
            completeness_per_factor.append(0.0)
            continue
        probs = col / total
        h = _entropy(probs)
        c_j = 1.0 - h / (max_entropy_c + 1e-10)
        completeness_per_factor.append(float(c_j))

    completeness = float(np.mean(completeness_per_factor))
    informativeness = float(np.mean(informativeness_scores))

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness,
        "per_factor_completeness": {
            factors_sub[j][0]: completeness_per_factor[j]
            for j in range(n_factors)
        },
        "per_factor_informativeness": {
            factors_sub[j][0]: informativeness_scores[j]
            for j in range(n_factors)
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# SAP: Separated Attribute Predictability  [Kumar et al., 2018]
# ═══════════════════════════════════════════════════════════════════════════

def compute_sap(reprs, gt, subsample=10000):
    """
    Compute Separated Attribute Predictability (SAP).

    For each factor k:
      1. Compute R²(z_j -> v_k) for each latent dim j using linear regression
      2. Sort R² descending
      3. SAP_k = R²_top1 - R²_top2

    SAP = mean over factors.

    Args:
        reprs: (N, latent_dim)
        gt: dict factor_name -> (N, factor_dim)
        subsample: max samples

    Returns:
        dict with 'sap' score and per-factor breakdown
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    factors = _flatten_factors(gt)
    if not factors:
        return {"sap": 0.0, "per_factor": {}}

    N = reprs.shape[0]
    if N > subsample:
        idx = np.random.choice(N, subsample, replace=False)
        reprs_sub = reprs[idx]
        factors_sub = [(name, arr[idx]) for name, arr in factors]
    else:
        reprs_sub = reprs
        factors_sub = factors

    # Split into train/test
    n_train = int(0.8 * len(reprs_sub))
    Z_train, Z_test = reprs_sub[:n_train], reprs_sub[n_train:]

    latent_dim = reprs_sub.shape[1]
    per_factor = {}

    for fname, fvals in factors_sub:
        y_train, y_test = fvals[:n_train], fvals[n_train:]

        # R² for each latent dim independently
        r2_per_dim = []
        for j in range(latent_dim):
            lr = LinearRegression()
            lr.fit(Z_train[:, j:j+1], y_train)
            y_pred = lr.predict(Z_test[:, j:j+1])
            r2 = r2_score(y_test, y_pred)
            r2_per_dim.append(max(0.0, float(r2)))

        r2_sorted = sorted(r2_per_dim, reverse=True)
        gap = r2_sorted[0] - r2_sorted[1] if len(r2_sorted) > 1 else r2_sorted[0]

        per_factor[fname] = {
            "sap": float(gap),
            "top1_r2": float(r2_sorted[0]),
            "top2_r2": float(r2_sorted[1]) if len(r2_sorted) > 1 else 0.0,
            "best_dim": int(np.argmax(r2_per_dim)),
        }

    sap_score = float(np.mean([v["sap"] for v in per_factor.values()]))
    return {"sap": sap_score, "per_factor": per_factor}


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(base_dir):
    """Aggregate MIG/DCI/SAP results across seeds and models."""
    base_dir = Path(base_dir)
    dis_dir = base_dir / "disentanglement"

    files = sorted(dis_dir.glob("disentanglement_*.json")) if dis_dir.exists() else []
    if not files:
        print(f"No disentanglement result files found in {dis_dir}")
        return {}

    by_model = {}
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        mt = data["model_type"]
        if mt not in by_model:
            by_model[mt] = []
        by_model[mt].append(data)

    agg = {}
    for mt, runs in by_model.items():
        mig_vals = [r["mig"]["mig"] for r in runs]
        dci_d_vals = [r["dci"]["disentanglement"] for r in runs]
        dci_c_vals = [r["dci"]["completeness"] for r in runs]
        dci_i_vals = [r["dci"]["informativeness"] for r in runs]
        sap_vals = [r["sap"]["sap"] for r in runs]

        agg[mt] = {
            "n_seeds": len(runs),
            "mig": {"mean": float(np.mean(mig_vals)), "std": float(np.std(mig_vals))},
            "dci_disentanglement": {"mean": float(np.mean(dci_d_vals)), "std": float(np.std(dci_d_vals))},
            "dci_completeness": {"mean": float(np.mean(dci_c_vals)), "std": float(np.std(dci_c_vals))},
            "dci_informativeness": {"mean": float(np.mean(dci_i_vals)), "std": float(np.std(dci_i_vals))},
            "sap": {"mean": float(np.mean(sap_vals)), "std": float(np.std(sap_vals))},
        }

    # Save
    out_file = dis_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[Aggregate] Saved: {out_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Disentanglement Metrics Summary (mean ± std across seeds)")
    print("=" * 70)

    header = f"{'Metric':<25}"
    for mt in sorted(agg.keys()):
        header += f"  {mt:>20}"
    print(header)
    print("-" * 70)

    metrics = [
        ("MIG", "mig"),
        ("DCI-Disentanglement", "dci_disentanglement"),
        ("DCI-Completeness", "dci_completeness"),
        ("DCI-Informativeness", "dci_informativeness"),
        ("SAP", "sap"),
    ]
    for label, key in metrics:
        row = f"{label:<25}"
        for mt in sorted(agg.keys()):
            d = agg[mt][key]
            row += f"  {d['mean']:>8.4f} ± {d['std']:.4f}  "
        print(row)

    return agg


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Disentanglement Metrics: MIG, DCI, SAP")
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

    dis_dir = Path(args.output_dir) / "disentanglement"
    dis_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Disentanglement] Model: {args.model_type}, Seed: {args.seed}")
    print(f"[Disentanglement] Checkpoint: {args.ckpt}")

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
    print(f"\n[Disentanglement] Collecting representations...")
    t0 = time.time()
    reprs, gt = collect_representations(
        cfg, model, device, args.num_episodes, args.model_type
    )
    print(f"  Collected {reprs.shape[0]} samples, repr_dim={reprs.shape[1]} in {time.time()-t0:.0f}s")

    # Compute MIG
    print(f"\n[Disentanglement] Computing MIG...")
    t0 = time.time()
    mig_result = compute_mig(reprs, gt)
    print(f"  MIG = {mig_result['mig']:.4f} ({time.time()-t0:.1f}s)")
    for fname, fdata in mig_result["per_factor"].items():
        print(f"    {fname}: MIG={fdata['mig']:.4f} (top1={fdata['top1_mi']:.4f}, top2={fdata['top2_mi']:.4f}, dim={fdata['best_dim']})")

    # Compute DCI
    print(f"\n[Disentanglement] Computing DCI...")
    t0 = time.time()
    dci_result = compute_dci(reprs, gt)
    print(f"  Disentanglement = {dci_result['disentanglement']:.4f}")
    print(f"  Completeness    = {dci_result['completeness']:.4f}")
    print(f"  Informativeness = {dci_result['informativeness']:.4f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Compute SAP
    print(f"\n[Disentanglement] Computing SAP...")
    t0 = time.time()
    sap_result = compute_sap(reprs, gt)
    print(f"  SAP = {sap_result['sap']:.4f} ({time.time()-t0:.1f}s)")
    for fname, fdata in sap_result["per_factor"].items():
        print(f"    {fname}: SAP={fdata['sap']:.4f} (top1={fdata['top1_r2']:.4f}, top2={fdata['top2_r2']:.4f}, dim={fdata['best_dim']})")

    # Save results
    result = {
        "model_type": args.model_type,
        "seed": args.seed,
        "checkpoint": str(args.ckpt),
        "mig": mig_result,
        "dci": dci_result,
        "sap": sap_result,
        "timestamp": datetime.now().isoformat(),
    }

    out_file = dis_dir / f"disentanglement_{args.model_type}_seed{args.seed}.json"
    tmp_file = out_file.with_suffix(".tmp")
    json_str = json.dumps(result, indent=2)
    with open(tmp_file, "w") as f:
        f.write(json_str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_file), str(out_file))
    print(f"\n  Saved: {out_file} ({len(json_str)} bytes)")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
