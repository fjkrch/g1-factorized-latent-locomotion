"""
Latent space analysis utilities for DynaMITE.

Analyzes the learned latent dynamics representation:
- Correlation between latent factors and ground-truth dynamics parameters
- t-SNE/PCA visualization of latent space
- Factor disentanglement metrics
- Auxiliary prediction accuracy

Usage:
    python scripts/analyze_latent.py --checkpoint path/to/best.pt --output_dir figures/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def collect_latent_data(
    model: nn.Module,
    env,
    num_episodes: int = 50,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """
    Collect latent vectors and corresponding dynamics parameters.

    Args:
        model: DynaMITE model.
        env: Environment wrapper.
        num_episodes: Number of episodes to collect.
        device: Torch device.

    Returns:
        dict with:
            "latent_z": (N, latent_dim)
            "dynamics_params": {factor_name: (N, param_dim)}
    """
    from src.utils.history_buffer import HistoryBuffer

    model.eval()
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    history_len = model.history_len

    history_buf = HistoryBuffer(
        num_envs=env.num_envs,
        history_len=history_len,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    all_z = []
    all_params = {k: [] for k in ["friction", "mass", "motor", "contact", "delay"]}

    with torch.no_grad():
        reset_data = env.reset()
        obs = reset_data["obs"]
        cmd = reset_data["cmd"]
        completed = 0
        step = 0

        while completed < num_episodes:
            obs_hist, act_hist, hist_mask = history_buf.get()

            z, factors = model.get_latent(obs_hist, act_hist, cmd, hist_mask)
            all_z.append(z.cpu().numpy())

            if "dynamics_params" in reset_data:
                for k, v in reset_data["dynamics_params"].items():
                    all_params[k].append(v.cpu().numpy())

            # Take action
            output = model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                          act_hist=act_hist, hist_mask=hist_mask)
            action = output["action_mean"]

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            history_buf.insert(obs, action)
            reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
            if len(reset_ids) > 0:
                history_buf.reset_envs(reset_ids)
                completed += len(reset_ids)
            step += 1

    result = {
        "latent_z": np.concatenate(all_z, axis=0),
    }
    for k, v_list in all_params.items():
        if v_list:
            result[f"params_{k}"] = np.concatenate(v_list, axis=0)

    return result


def compute_correlations(
    latent_z: np.ndarray,
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute correlation matrix between latent dimensions and dynamics parameters.

    Returns:
        dict of param_name -> correlation matrix (latent_dim x param_dim)
    """
    correlations = {}
    for name, param_vals in params.items():
        if param_vals.ndim == 1:
            param_vals = param_vals[:, None]
        n = min(len(latent_z), len(param_vals))
        z = latent_z[:n]
        p = param_vals[:n]

        corr = np.zeros((z.shape[1], p.shape[1]))
        for i in range(z.shape[1]):
            for j in range(p.shape[1]):
                corr[i, j] = np.corrcoef(z[:, i], p[:, j])[0, 1]
        correlations[name] = corr

    return correlations


def compute_disentanglement_score(
    correlations: dict[str, np.ndarray],
    factor_assignments: dict[str, list[int]],
) -> float:
    """
    Compute a simple disentanglement score.

    For each factor, checks if its assigned latent dimensions have
    higher correlation with the corresponding params than other dims.

    Args:
        correlations: from compute_correlations
        factor_assignments: maps factor name -> list of latent dimension indices

    Returns:
        Score in [0, 1], higher = more disentangled
    """
    scores = []
    for name, dim_indices in factor_assignments.items():
        if name not in correlations:
            continue
        corr_matrix = np.abs(correlations[name])
        # Mean correlation of assigned dims
        assigned_corr = corr_matrix[dim_indices].mean()
        # Mean correlation of unassigned dims
        all_dims = list(range(corr_matrix.shape[0]))
        other_dims = [d for d in all_dims if d not in dim_indices]
        if other_dims:
            other_corr = corr_matrix[other_dims].mean()
            scores.append(assigned_corr / (assigned_corr + other_corr + 1e-8))
        else:
            scores.append(1.0)

    return float(np.mean(scores)) if scores else 0.0


def save_latent_analysis(
    analysis: dict[str, Any],
    output_dir: str | Path,
):
    """Save latent analysis results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "latent_analysis.json", "w") as f:
        json.dump(
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in analysis.items()},
            f, indent=2,
        )
