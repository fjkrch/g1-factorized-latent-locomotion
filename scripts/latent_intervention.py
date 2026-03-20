#!/usr/bin/env python3
"""
Latent Intervention Analysis (Must-do 4).

For each factor subspace in DynaMITE's latent, we:
  1. Run a baseline episode collecting per-step latent vectors
  2. "Clamp" that factor's subspace to its mean value while keeping others intact
  3. Re-run the policy with the clamped latent and measure behavioral impact

This tests whether each subspace has functional relevance: if clamping factor k
causes reward degradation when the corresponding DR parameter is varied,
that's causal evidence (beyond correlation) that the subspace encodes useful info.

Produces:
  - results/latent_intervention/intervention_results.json
  - figures/latent_intervention.png

Usage:
    python scripts/latent_intervention.py [--seeds 42 43 44] [--num_episodes 50]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.utils.history_buffer import HistoryBuffer


# Factor subspace index ranges in the 24-d latent
FACTOR_RANGES = {
    "friction": (0, 4),
    "mass": (4, 10),
    "motor": (10, 16),
    "contact": (16, 20),
    "delay": (20, 24),
}

# DR parameter to vary for each factor's test
FACTOR_DR_PARAMS = {
    "friction": {"friction_range": [[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]]},
    "mass": {"added_mass_range": [[-2.0, -2.0], [0.0, 0.0], [5.0, 5.0]]},
    "motor": {"motor_strength_range": [[0.7, 0.7], [1.0, 1.0], [1.3, 1.3]]},
    "contact": {"restitution_range": [[0.0, 0.0], [0.4, 0.4], [0.8, 0.8]]},
    "delay": {"action_delay_range": [[0, 0], [2, 2], [5, 5]]},
}


def evaluate_with_intervention(
    cfg, model, device, num_episodes, clamp_factor=None, clamp_value=None,
    existing_env=None,
):
    """Run evaluation, optionally clamping a factor subspace.

    If clamp_factor is set, after the model computes the latent z,
    we replace the clamped factor's dimensions with clamp_value before
    feeding into the policy head.
    """
    model.eval()
    if existing_env is not None:
        env = existing_env
    else:
        env = make_env(cfg, device=device)

    obs_dim = cfg["task"]["observation"]["proprioception_dim"]
    act_dim = cfg["task"]["observation"]["action_dim"]
    history_len = cfg["task"]["observation"]["history_len"]

    history_buf = HistoryBuffer(
        num_envs=cfg["task"]["num_envs"],
        history_len=history_len,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    episode_rewards = torch.zeros(env.num_envs, device=device)
    episode_lengths = torch.zeros(env.num_envs, device=device)
    all_rewards = []
    all_lengths = []
    completed = 0

    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data["cmd"]
    prev_action = torch.zeros(env.num_envs, act_dim, device=device)
    history_buf.insert(obs, prev_action)

    with torch.no_grad():
        while completed < num_episodes:
            obs_hist, act_hist, hist_mask = history_buf.get()

            # Forward through transformer + latent head
            tokens = model._build_tokens(obs_hist, act_hist, cmd)
            tokens = model.pos_enc(tokens)
            padding_mask = ~hist_mask if hist_mask is not None else None
            features = model.transformer(tokens, src_key_padding_mask=padding_mask)
            pooled = model._aggregate(features, hist_mask)

            # Get latent
            if model.factorized:
                latent_z, latent_factors = model.latent_head(pooled)
            else:
                latent_z = model.latent_head(pooled)

            # Apply clamping if requested
            if clamp_factor is not None and clamp_value is not None:
                start, end = FACTOR_RANGES[clamp_factor]
                latent_z = latent_z.clone()
                latent_z[:, start:end] = clamp_value

            # Build head input and get action
            if model.condition_on_latent and latent_z is not None:
                head_input = torch.cat([pooled, latent_z], dim=-1)
            else:
                head_input = pooled
            action_mean, _ = model.policy(head_input)
            action = action_mean

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            episode_rewards += step_data["reward"]
            episode_lengths += 1

            history_buf.insert(obs, action)
            prev_action = action
            reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
            if len(reset_ids) > 0:
                history_buf.reset_envs(reset_ids)
                history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                history_buf.lengths[reset_ids] = 1

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids:
                all_rewards.append(episode_rewards[idx].item())
                all_lengths.append(episode_lengths[idx].item())
                episode_rewards[idx] = 0
                episode_lengths[idx] = 0
                completed += 1

    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "length_mean": float(np.mean(all_lengths)),
        "n_episodes": len(all_rewards),
    }, env


def collect_baseline_latents(cfg, model, device, num_episodes, existing_env=None):
    """Collect latent z vectors from normal evaluation to compute mean per factor."""
    model.eval()
    if existing_env is not None:
        env = existing_env
    else:
        env = make_env(cfg, device=device)

    obs_dim = cfg["task"]["observation"]["proprioception_dim"]
    act_dim = cfg["task"]["observation"]["action_dim"]
    history_len = cfg["task"]["observation"]["history_len"]

    history_buf = HistoryBuffer(
        num_envs=cfg["task"]["num_envs"],
        history_len=history_len,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    all_z = []
    completed = 0
    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data["cmd"]
    prev_action = torch.zeros(env.num_envs, act_dim, device=device)
    history_buf.insert(obs, prev_action)

    with torch.no_grad():
        while completed < num_episodes:
            obs_hist, act_hist, hist_mask = history_buf.get()
            output = model(obs=obs, cmd=cmd, obs_hist=obs_hist,
                          act_hist=act_hist, hist_mask=hist_mask)
            action = output["action_mean"]
            z = output.get("latent_z", None)
            if z is not None:
                all_z.append(z.cpu())

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]
            history_buf.insert(obs, action)
            prev_action = action
            reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
            if len(reset_ids) > 0:
                history_buf.reset_envs(reset_ids)
                history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                history_buf.lengths[reset_ids] = 1
                completed += len(reset_ids)

    all_z = torch.cat(all_z, dim=0)
    return all_z, env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results/latent_intervention")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = {}

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f"  SEED {seed} ({seed_idx+1}/{len(args.seeds)})")
        print(f"{'='*60}")

        set_seed(seed)

        # Find checkpoint
        import glob
        ckpt_pattern = f"outputs/randomized/dynamite_full/seed_{seed}/*/checkpoints/best.pt"
        ckpt_matches = sorted(glob.glob(ckpt_pattern))
        if not ckpt_matches:
            print(f"  [WARN] No checkpoint for seed {seed}, skipping")
            continue
        ckpt_path = Path(ckpt_matches[-1])

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        cfg["task"]["num_envs"] = 32
        eval_cfg = load_yaml("configs/eval/default.yaml")
        cfg = {**cfg, **eval_cfg}

        init_sim(headless=True)
        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        # Step 1: Collect baseline latents to compute mean per factor
        print("  Collecting baseline latents...")
        all_z, env = collect_baseline_latents(
            cfg, model, device, args.num_episodes
        )
        factor_means = {}
        for name, (start, end) in FACTOR_RANGES.items():
            factor_means[name] = all_z[:, start:end].mean(dim=0).to(device)
            print(f"    {name}: mean_norm={factor_means[name].norm():.3f}")

        seed_results = {}

        # Step 2: For each factor, test clamping under varied DR
        for factor_name in FACTOR_RANGES:
            print(f"\n  Testing factor: {factor_name}")
            dr_param_name = list(FACTOR_DR_PARAMS[factor_name].keys())[0]
            dr_values = FACTOR_DR_PARAMS[factor_name][dr_param_name]

            factor_results = {"baseline": [], "clamped": []}

            for dr_val in dr_values:
                # Set the DR parameter
                cfg["task"]["domain_randomization"][dr_param_name] = dr_val

                # Baseline (no clamping)
                base_metrics, env = evaluate_with_intervention(
                    cfg, model, device, args.num_episodes,
                    clamp_factor=None, clamp_value=None,
                    existing_env=env,
                )
                factor_results["baseline"].append({
                    "dr_value": dr_val,
                    **base_metrics,
                })

                # Clamped
                clamp_val = factor_means[factor_name]
                clamp_metrics, env = evaluate_with_intervention(
                    cfg, model, device, args.num_episodes,
                    clamp_factor=factor_name, clamp_value=clamp_val,
                    existing_env=env,
                )
                factor_results["clamped"].append({
                    "dr_value": dr_val,
                    **clamp_metrics,
                })

                delta = clamp_metrics["reward_mean"] - base_metrics["reward_mean"]
                print(f"    {dr_param_name}={dr_val}: "
                      f"base={base_metrics['reward_mean']:.2f}  "
                      f"clamped={clamp_metrics['reward_mean']:.2f}  "
                      f"delta={delta:+.2f}")

            seed_results[factor_name] = factor_results

        all_seed_results[f"seed_{seed}"] = seed_results

        # Force exit Isaac Lab (can't create multiple apps)
        # For multi-seed, we need to restart the Python process
        # Save intermediate results
        with open(output_dir / f"intervention_seed{seed}.json", "w") as f:
            json.dump(seed_results, f, indent=2, default=str)
        print(f"\n  Saved: {output_dir / f'intervention_seed{seed}.json'}")

        # For FIRST seed only — subsequent seeds need new SimulationApp
        # break after first seed, run separately
        break

    # Save combined results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds[:1],  # seeds actually processed
        "num_episodes": args.num_episodes,
        "factors": list(FACTOR_RANGES.keys()),
        "results": all_seed_results,
    }
    with open(output_dir / "intervention_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nSaved: {output_dir / 'intervention_results.json'}")

    # Generate figure
    try:
        generate_intervention_figure(combined, output_dir)
    except Exception as e:
        print(f"[WARN] Could not generate figure: {e}")

    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


def generate_intervention_figure(results, output_dir):
    """Generate latent intervention figure showing factor-specific impact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    factors = results["factors"]
    n_factors = len(factors)

    # Use first available seed
    seed_key = list(results["results"].keys())[0]
    seed_data = results["results"][seed_key]

    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 4), sharey=False)
    if n_factors == 1:
        axes = [axes]

    colors = {"baseline": "#2ca02c", "clamped": "#d62728"}

    for idx, factor in enumerate(factors):
        ax = axes[idx]
        fdata = seed_data[factor]

        base_rewards = [r["reward_mean"] for r in fdata["baseline"]]
        clamp_rewards = [r["reward_mean"] for r in fdata["clamped"]]
        dr_vals = [str(r["dr_value"]) for r in fdata["baseline"]]

        x = np.arange(len(dr_vals))
        w = 0.35

        bars1 = ax.bar(x - w/2, base_rewards, w, label="Normal", color=colors["baseline"], alpha=0.8)
        bars2 = ax.bar(x + w/2, clamp_rewards, w, label=f"Clamp {factor}", color=colors["clamped"], alpha=0.8)

        ax.set_xlabel(f"{factor} DR value", fontsize=10)
        ax.set_ylabel("Reward" if idx == 0 else "", fontsize=10)
        ax.set_title(f"Clamp: {factor}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(dr_vals, fontsize=8, rotation=30)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == n_factors - 1:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle("Latent Factor Clamping Intervention", fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = fig_dir / "latent_intervention.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
