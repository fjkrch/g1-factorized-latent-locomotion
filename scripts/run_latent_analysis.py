#!/usr/bin/env python3
"""
Run latent–dynamics correlation analysis on a DynaMITE checkpoint.

Produces:
  - results/latent_analysis/latent_analysis.json   (raw correlations + scores)
  - figures/latent_correlation_heatmap.png          (5×5 factor heatmap)
  - figures/latent_tsne.png                         (t-SNE colored by friction)

Usage:
    python scripts/run_latent_analysis.py \
        --checkpoint outputs/randomized/dynamite_full/seed_42/.../checkpoints/best.pt \
        --num_episodes 100 \
        --output_dir results/latent_analysis \
        --figure_dir figures
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.analysis.latent_analysis import (
    collect_latent_data,
    compute_correlations,
    compute_disentanglement_score,
    save_latent_analysis,
)


# Factor assignments: latent dim indices → factor name
# friction: dims 0-3, mass: 4-9, motor: 10-15, contact: 16-19, delay: 20-23
FACTOR_ASSIGNMENTS = {
    "friction": list(range(0, 4)),
    "mass": list(range(4, 10)),
    "motor": list(range(10, 16)),
    "contact": list(range(16, 20)),
    "delay": list(range(20, 24)),
}

FACTOR_NAMES = ["friction", "mass", "motor", "contact", "delay"]


def plot_correlation_heatmap(correlations, factor_assignments, output_path):
    """Plot a block-structured heatmap: latent factors vs GT parameters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("  [WARN] matplotlib not available, skipping heatmap")
        return

    # Build block-mean correlation matrix (5×5)
    factor_names = FACTOR_NAMES
    block_corr = np.zeros((len(factor_names), len(factor_names)))

    for i, f_latent in enumerate(factor_names):
        dims = factor_assignments[f_latent]
        for j, f_param in enumerate(factor_names):
            if f_param in correlations:
                corr_mat = correlations[f_param]  # (latent_dim, param_dim)
                # Mean absolute correlation of this factor's latent dims with this param
                block_corr[i, j] = np.abs(corr_mat[dims]).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(block_corr, cmap="YlOrRd", vmin=0, vmax=max(0.5, block_corr.max()),
                   aspect="equal")

    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels([f"GT {n}" for n in factor_names], rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(factor_names)))
    ax.set_yticklabels([f"z_{n}" for n in factor_names], fontsize=11)
    ax.set_title("Mean |Pearson r|: Latent Factors vs GT Dynamics Parameters", fontsize=12)

    # Annotate cells
    for i in range(len(factor_names)):
        for j in range(len(factor_names)):
            color = "white" if block_corr[i, j] > 0.3 else "black"
            ax.text(j, i, f"{block_corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="|Pearson r|")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmap saved: {output_path}")


def plot_full_correlation_matrix(correlations, factor_assignments, output_path):
    """Plot the full 24×N correlation matrix with factor boundaries."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Stack all param correlations horizontally
    param_names = []
    corr_cols = []
    for name in FACTOR_NAMES:
        if name in correlations:
            c = correlations[name]  # (24, param_dim)
            for j in range(c.shape[1]):
                param_names.append(f"{name}_{j}")
                corr_cols.append(c[:, j])

    if not corr_cols:
        return

    full_corr = np.column_stack(corr_cols)  # (24, total_params)

    fig, ax = plt.subplots(figsize=(max(10, len(param_names) * 0.5), 8))
    im = ax.imshow(np.abs(full_corr), cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")

    ax.set_xlabel("Ground-truth parameter dimensions", fontsize=11)
    ax.set_ylabel("Latent dimension index", fontsize=11)
    ax.set_title("Full |Pearson r|: Each Latent Dim vs Each GT Param Dim", fontsize=12)

    # Draw factor boundaries on y-axis
    boundaries = [0, 4, 10, 16, 20, 24]
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=2)

    # Label factor groups on y-axis
    midpoints = [(boundaries[i] + boundaries[i + 1]) / 2 - 0.5 for i in range(5)]
    for mid, name in zip(midpoints, FACTOR_NAMES):
        ax.text(-1.5, mid, f"z_{name}", ha="right", va="center", fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="|Pearson r|")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Full correlation matrix saved: {output_path}")


def plot_tsne(latent_z, params, factor_name, output_path):
    """t-SNE of latent vectors colored by a single parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [WARN] sklearn or matplotlib not available, skipping t-SNE")
        return

    # Subsample for speed
    n = min(5000, len(latent_z))
    idx = np.random.choice(len(latent_z), n, replace=False)
    z_sub = latent_z[idx]
    p_sub = params[idx]

    if p_sub.ndim > 1:
        p_sub = p_sub[:, 0]  # Use first dim for coloring

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=p_sub, cmap="viridis", s=5, alpha=0.6)
    ax.set_title(f"t-SNE of DynaMITE Latent Space\n(colored by {factor_name})", fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(sc, ax=ax, label=factor_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ t-SNE saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DynaMITE Latent Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", default="results/latent_analysis")
    parser.add_argument("--figure_dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and config
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["task"]["num_envs"] = 64  # enough for stats, light on VRAM

    print(f"[Latent Analysis] Checkpoint: {ckpt_path}")
    print(f"[Latent Analysis] Model: {cfg['model']['name']}")
    print(f"[Latent Analysis] Episodes: {args.num_episodes}")

    # Init sim + env + model
    init_sim(headless=True)
    env = make_env(cfg, device=device, headless=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Check model has get_latent
    if not hasattr(model, "get_latent"):
        print("ERROR: Model does not have get_latent(). Is this a DynaMITE checkpoint?")
        sys.exit(1)

    # Collect data
    print("[Latent Analysis] Collecting latent vectors...")
    data = collect_latent_data(model, env, num_episodes=args.num_episodes, device=device)

    latent_z = data["latent_z"]
    print(f"  Collected {latent_z.shape[0]} latent vectors, dim={latent_z.shape[1]}")

    # Extract param arrays
    params = {}
    for name in FACTOR_NAMES:
        key = f"params_{name}"
        if key in data:
            params[name] = data[key]
            print(f"  {name}: {params[name].shape}")

    if not params:
        print("  WARNING: No dynamics_params returned by env. Will skip correlation analysis.")
        print("  (The env may not expose GT dynamics parameters in eval mode.)")
        # Still save latent stats
        analysis = {
            "latent_shape": list(latent_z.shape),
            "latent_mean": latent_z.mean(axis=0).tolist(),
            "latent_std": latent_z.std(axis=0).tolist(),
            "note": "No GT dynamics params available from env",
        }
        save_latent_analysis(analysis, output_dir)
        # Force-exit to avoid Isaac Lab teardown hang
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(0)

    # Compute correlations
    print("[Latent Analysis] Computing correlations...")
    correlations = compute_correlations(latent_z, params)

    # Disentanglement score
    score = compute_disentanglement_score(correlations, FACTOR_ASSIGNMENTS)
    print(f"\n  ★ Disentanglement Score: {score:.4f}")

    # Per-factor summary
    print("\n  Per-factor mean |r| (diagonal = assigned, off-diagonal = unassigned):")
    for name in FACTOR_NAMES:
        if name in correlations:
            dims = FACTOR_ASSIGNMENTS[name]
            assigned_r = np.abs(correlations[name][dims]).mean()
            other_dims = [d for d in range(latent_z.shape[1]) if d not in dims]
            unassigned_r = np.abs(correlations[name][other_dims]).mean() if other_dims else 0
            print(f"    {name:>10s}: assigned={assigned_r:.4f}  unassigned={unassigned_r:.4f}  ratio={assigned_r/(unassigned_r+1e-8):.2f}x")

    # Save JSON
    analysis = {
        "disentanglement_score": score,
        "latent_shape": list(latent_z.shape),
        "correlations": {k: v.tolist() for k, v in correlations.items()},
        "factor_assignments": FACTOR_ASSIGNMENTS,
        "num_episodes": args.num_episodes,
        "checkpoint": str(ckpt_path),
    }
    save_latent_analysis(analysis, output_dir)
    print(f"\n  ✓ Analysis saved: {output_dir / 'latent_analysis.json'}")

    # Plot heatmap
    plot_correlation_heatmap(correlations, FACTOR_ASSIGNMENTS,
                            figure_dir / "latent_correlation_heatmap.png")

    # Plot full correlation matrix
    plot_full_correlation_matrix(correlations, FACTOR_ASSIGNMENTS,
                                figure_dir / "latent_correlation_full.png")

    # Plot t-SNE colored by friction (most intuitive)
    if "friction" in params:
        plot_tsne(latent_z, params["friction"], "friction",
                  figure_dir / "latent_tsne_friction.png")

    # Plot t-SNE colored by delay
    if "delay" in params:
        plot_tsne(latent_z, params["delay"], "delay",
                  figure_dir / "latent_tsne_delay.png")

    # Force-exit to avoid Isaac Lab teardown hang (_is_closed bug)
    print("\n[Latent Analysis] Done.")
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
