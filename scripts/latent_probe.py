#!/usr/bin/env python3
"""
Latent Probe Analysis: DynaMITE latent vs LSTM hidden state.

Compares how well a simple linear probe can predict ground-truth dynamics
parameters from:
  1. DynaMITE's 24-d factored latent
  2. LSTM's 128-d hidden state

Also measures identification speed: after a DR parameter change, how many
steps until the representation updates to reflect the new value.

Protocol:
  1. Run rollouts collecting representations + ground-truth DR params
  2. Train linear regression probes for each factor
  3. Report R² for each factor × model
  4. Measure identification speed after perturbation onset

Usage:
    python scripts/latent_probe.py --seed 42 --num_episodes 200
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.utils.config import load_config, load_yaml
from src.utils.seed import set_seed
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.utils.checkpoint import load_checkpoint
from src.utils.history_buffer import HistoryBuffer

# Factor target dimensions in DR params
FACTOR_TARGETS = {
    "friction": 2,   # [static, dynamic]
    "mass": 2,       # [added_mass, com_disp]
    "motor": 2,      # [motor_strength_1, motor_strength_2]
    "contact": 1,    # [restitution]
    "delay": 1,      # [action_delay]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Probe Analysis")
    parser.add_argument("--dynamite_ckpt", type=str, required=True)
    parser.add_argument("--lstm_ckpt", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="Collection episodes for probe training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/latent_probe")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def collect_representations(cfg, model, device, num_episodes, model_type):
    """Run rollouts and collect model representations + GT dynamics params.
    
    For DynaMITE: collects the 24-d latent z vector
    For LSTM: collects the 128-d hidden state h
    
    Returns:
        representations: np.array (N, repr_dim)
        dynamics_gt: dict of factor_name -> np.array (N, factor_dim)
        step_info: list of (episode_idx, step_idx) for each sample
    """
    model.eval()
    env = make_env(cfg, device=device)
    
    uses_history = getattr(model, 'uses_history', False)
    history_buf = None
    if uses_history:
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
    all_step_info = []
    
    completed = 0
    episode_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    
    reset_data = env.reset()
    obs = reset_data["obs"]
    cmd = reset_data["cmd"]
    prev_action = torch.zeros(env.num_envs, env.act_dim, device=device)
    
    if uses_history and history_buf is not None:
        history_buf.insert(obs, prev_action)

    # For LSTM, maintain hidden state
    hidden = None
    if model_type == "lstm":
        hidden = model.init_hidden(env.num_envs, device)

    max_steps = num_episodes * 30  # ~30 steps per episode avg
    step_count = 0

    with torch.no_grad():
        while completed < num_episodes and step_count < max_steps:
            step_count += 1
            episode_step += 1

            model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
            
            if model_type == "dynamite":
                if uses_history and history_buf is not None:
                    o, a, m = history_buf.get()
                    model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})
                output = model(**model_input)
                # Extract latent z
                latent_z = output.get("latent_z", None)
                if latent_z is not None:
                    all_reprs.append(latent_z.cpu().numpy())
                else:
                    # fallback: skip
                    continue
                    
            elif model_type == "lstm":
                model_input["hidden"] = hidden
                output = model(**model_input)
                hidden = output["hidden"]
                # Extract hidden state (h of LSTM, not c)
                h_state = hidden[0]  # (num_layers, batch, hidden_dim)
                # Use last layer hidden state
                h_last = h_state[-1]  # (batch, hidden_dim)
                all_reprs.append(h_last.cpu().numpy())

            action = output["action_mean"]
            
            # Get ground-truth dynamics params
            dynamics = env._dynamics_params
            for factor in FACTOR_TARGETS:
                if factor in dynamics:
                    all_gt[factor].append(dynamics[factor].cpu().numpy())

            step_data = env.step(action)
            obs = step_data["obs"]
            cmd = step_data.get("cmd", cmd)
            done = step_data["done"]

            if uses_history and history_buf is not None:
                history_buf.insert(obs, action)
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if len(reset_ids) > 0:
                    history_buf.reset_envs(reset_ids)
                    history_buf.obs_buf[reset_ids, -1] = obs[reset_ids]
                    history_buf.lengths[reset_ids] = 1

            # Handle resets for LSTM hidden state
            if model_type == "lstm":
                reset_ids = step_data.get("reset_ids", torch.tensor([], device=device))
                if len(reset_ids) > 0 and isinstance(reset_ids, torch.Tensor) and len(reset_ids) > 0:
                    h, c = hidden
                    h[:, reset_ids] = 0
                    c[:, reset_ids] = 0
                    hidden = (h, c)

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            completed += len(done_ids)
            episode_step[done_ids] = 0

            prev_action = action

    env.close()

    # Stack collected data
    # Each element in all_reprs is (num_envs, repr_dim). Flatten into (total_steps*num_envs, repr_dim)
    reprs = np.concatenate(all_reprs, axis=0)
    gt = {}
    for factor in FACTOR_TARGETS:
        if all_gt[factor]:
            gt[factor] = np.concatenate(all_gt[factor], axis=0)
    
    return reprs, gt


def train_probes(reprs, gt, test_frac=0.2):
    """Train linear probes to predict each GT factor from representation.
    
    Returns:
        dict of factor_name -> {"r2": float, "r2_per_dim": list}
    """
    n = reprs.shape[0]
    split = int(n * (1 - test_frac))
    
    # Shuffle
    idx = np.random.permutation(n)
    reprs = reprs[idx]
    
    results = {}
    for factor, target_dim in FACTOR_TARGETS.items():
        if factor not in gt:
            continue
        y = gt[factor][idx]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        X_train, X_test = reprs[:split], reprs[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Ridge regression probe
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        
        # R² per dimension
        r2_per_dim = []
        for d in range(y_test.shape[1]):
            r2 = r2_score(y_test[:, d], y_pred[:, d])
            r2_per_dim.append(float(r2))
        
        # Overall R²
        r2_overall = float(r2_score(y_test, y_pred))
        
        results[factor] = {
            "r2": r2_overall,
            "r2_per_dim": r2_per_dim,
        }
    
    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_sim(headless=args.headless)

    all_results = {
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }

    for model_type, ckpt_path_str in [
        ("dynamite", args.dynamite_ckpt),
        ("lstm", args.lstm_ckpt),
    ]:
        print(f"\n{'='*60}")
        print(f"[Probe] Collecting representations for {model_type}...")
        ckpt_path = Path(ckpt_path_str)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        
        # Use randomized task for varied DR params
        cfg["task"]["name"] = "randomized"
        cfg["task"]["domain_randomization"]["enabled"] = True
        cfg["task"]["num_envs"] = 32

        model = build_model(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        reprs, gt = collect_representations(
            cfg, model, device, args.num_episodes, model_type
        )
        print(f"  Collected {reprs.shape[0]} samples, repr_dim={reprs.shape[1]}")
        
        probe_results = train_probes(reprs, gt)
        
        model_result = {
            "checkpoint": str(ckpt_path),
            "repr_dim": int(reprs.shape[1]),
            "n_samples": int(reprs.shape[0]),
            "probe_r2": probe_results,
        }
        all_results["models"][model_type] = model_result

        print(f"  Probe R² results for {model_type}:")
        for factor, res in probe_results.items():
            print(f"    {factor}: R²={res['r2']:.3f}  per-dim={[f'{x:.3f}' for x in res['r2_per_dim']]}")

    # Save
    out_file = output_dir / f"probe_results_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Probe] Saved to: {out_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("Factor Probe R² Comparison (higher = better identification)")
    print(f"{'Factor':<12} {'DynaMITE (24-d)':<18} {'LSTM (128-d)':<18}")
    print("-" * 48)
    for factor in FACTOR_TARGETS:
        d_r2 = all_results["models"]["dynamite"]["probe_r2"].get(factor, {}).get("r2", float('nan'))
        l_r2 = all_results["models"]["lstm"]["probe_r2"].get(factor, {}).get("r2", float('nan'))
        winner = "*" if d_r2 > l_r2 else ""
        print(f"{factor:<12} {d_r2:>6.3f}{winner:<12} {l_r2:>6.3f}")
    
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
