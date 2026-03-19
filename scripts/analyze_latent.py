#!/usr/bin/env python3
"""
Run latent space disentanglement analysis for DynaMITE.

Usage:
    python scripts/analyze_latent.py --checkpoint path/to/best.pt \
        --task configs/task/randomized.yaml --output_dir results/latent_analysis/seed_42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="DynaMITE Latent Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="configs/task/randomized.yaml")
    parser.add_argument("--output_dir", type=str, default="results/latent_analysis")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    from src.utils.config import load_config, load_yaml
    from src.utils.seed import set_seed
    from src.envs.g1_env import init_sim, make_env
    from src.models import build_model
    from src.analysis.latent_analysis import (
        collect_latent_data,
        compute_correlations,
        compute_disentanglement_score,
        save_latent_analysis,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    print(f"[Latent] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Check if model is DynaMITE (has latent head)
    if cfg["model"]["name"] != "dynamite":
        print("[Latent] SKIP — latent analysis only applies to DynaMITE model")
        return

    # Create env
    init_sim(headless=args.headless)
    env = make_env(cfg, device=device, headless=args.headless)

    # Build model and load weights
    model = build_model(cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect latent data
    print(f"[Latent] Collecting data from {args.num_episodes} episodes...")
    data = collect_latent_data(model, env, num_episodes=args.num_episodes, device=device)

    # Extract latent_z and params from collected data
    latent_z = data["latent_z"]
    params = {}
    for k, v in data.items():
        if k.startswith("params_"):
            params[k.replace("params_", "")] = v
    print(f"[Latent] Collected {latent_z.shape[0]} samples, latent_dim={latent_z.shape[1]}")

    # Compute correlations
    print("[Latent] Computing correlations...")
    correlations = compute_correlations(latent_z, params)

    # Compute disentanglement score
    # Factor assignments: which latent dims map to which factor
    factor_cfg = cfg.get("model", {}).get("latent", {}).get("factors", {})
    factor_assignments = {}
    offset = 0
    for name, dim in factor_cfg.items():
        factor_assignments[name] = list(range(offset, offset + dim))
        offset += dim
    print(f"[Latent] Factor assignments: {factor_assignments}")

    print("[Latent] Computing disentanglement score...")
    disent_score = compute_disentanglement_score(correlations, factor_assignments)

    # Save results
    analysis = {
        "correlations": correlations,
        "disentanglement_score": disent_score,
        "factor_assignments": factor_assignments,
        "num_episodes": args.num_episodes,
        "checkpoint": str(ckpt_path),
        "latent_dim": int(latent_z.shape[1]),
        "num_samples": int(latent_z.shape[0]),
    }
    save_latent_analysis(analysis, args.output_dir)
    print(f"[Latent] Disentanglement score: {disent_score:.4f}")
    print(f"[Latent] Results saved to {args.output_dir}")

    env.close()

    # Force exit to avoid Isaac Lab teardown hang
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
