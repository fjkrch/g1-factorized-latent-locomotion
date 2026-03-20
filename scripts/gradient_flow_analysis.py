#!/usr/bin/env python3
"""
Analysis 1: Gradient Flow Analysis for DynaMITE.

Instruments PPO training to log per-component gradient norms and cosine
similarities between the PPO gradient and each auxiliary loss gradient.

One process per seed (SimulationApp constraint).

Usage:
    python scripts/gradient_flow_analysis.py --seed 42
    python scripts/gradient_flow_analysis.py --seed 43
    python scripts/gradient_flow_analysis.py --seed 44
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.checkpoint import save_checkpoint
from src.utils.metrics_io import write_step_header, append_step_row
from src.utils.run_naming import make_run_id
from src.utils.manifest import create_manifest, save_manifest, update_manifest
from src.models import build_model
from src.envs.g1_env import init_sim, make_env
from src.algos.ppo import PPOTrainer, RolloutBuffer
from src.utils.history_buffer import HistoryBuffer

FACTOR_NAMES = ["friction", "mass", "motor", "contact", "delay"]


def compute_gradient_stats(model, loss_dict):
    """
    Compute gradient norms and cosine similarities for each loss component.

    Args:
        model: DynaMITE model
        loss_dict: {"ppo": ppo_loss, "aux_friction": loss, ...}

    Returns:
        norms: dict of loss_name -> gradient L2 norm
        cosines: dict of aux_factor_name -> cosine sim with PPO gradient
    """
    params = [p for p in model.parameters() if p.requires_grad]

    grad_vectors = {}
    for name, loss_val in loss_dict.items():
        if loss_val is None:
            continue
        if isinstance(loss_val, torch.Tensor) and loss_val.abs().item() < 1e-12:
            continue
        try:
            grads = torch.autograd.grad(
                loss_val, params, retain_graph=True, allow_unused=True
            )
            vec = torch.cat([
                g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(grads, params)
            ])
            grad_vectors[name] = vec
        except RuntimeError:
            continue

    norms = {name: float(vec.norm().item()) for name, vec in grad_vectors.items()}

    cosines = {}
    if "ppo" in grad_vectors:
        ppo_vec = grad_vectors["ppo"]
        ppo_norm = ppo_vec.norm()
        if ppo_norm > 1e-10:
            for name, vec in grad_vectors.items():
                if name.startswith("aux_"):
                    vec_norm = vec.norm()
                    if vec_norm > 1e-10:
                        cos = F.cosine_similarity(
                            ppo_vec.unsqueeze(0), vec.unsqueeze(0)
                        ).item()
                    else:
                        cos = 0.0
                    cosines[name] = float(cos)

    return norms, cosines


class InstrumentedPPOTrainer(PPOTrainer):
    """PPOTrainer with gradient flow instrumentation."""

    def __init__(self, *args, gradient_log_interval=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_log = []
        self.gradient_log_interval = gradient_log_interval
        self._update_count = 0

    def _ppo_update(self):
        """PPO update with gradient flow logging on first minibatch every N iterations."""
        self._update_count += 1
        should_instrument = (self._update_count % self.gradient_log_interval == 0)

        buf = self.rollout_buffer
        batch_size = buf.num_steps * buf.num_envs
        minibatch_size = batch_size // self.num_minibatches

        # Flatten rollout data
        obs_flat = buf.obs.reshape(-1, buf.obs.shape[-1])
        cmd_flat = buf.cmd.reshape(batch_size, buf.cmd.shape[-1])
        actions_flat = buf.actions.reshape(-1, buf.actions.shape[-1])
        old_log_probs = buf.log_probs.reshape(-1)
        advantages = buf.advantages.reshape(-1)
        returns = buf.returns.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_hist_flat, act_hist_flat, hist_mask_flat = None, None, None
        if self.uses_history:
            obs_hist_flat = buf.obs_hist.reshape(-1, *buf.obs_hist.shape[2:])
            act_hist_flat = buf.act_hist.reshape(-1, *buf.act_hist.shape[2:])
            hist_mask_flat = buf.hist_mask.reshape(-1, buf.hist_mask.shape[-1])

        dynamics_flat = None
        if self.uses_aux and buf.dynamics_targets[0] is not None:
            dynamics_flat = {}
            for key in buf.dynamics_targets[0].keys():
                parts = [buf.dynamics_targets[t][key] for t in range(buf.num_steps)]
                dynamics_flat[key] = torch.stack(parts).reshape(-1, parts[0].shape[-1])

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_aux_loss = 0.0
        total_approx_kl = 0.0
        n_updates = 0
        first_mb_instrumented = False

        for epoch in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_flat[mb_idx]
                mb_cmd = cmd_flat[mb_idx]
                mb_actions = actions_flat[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                model_input = {"obs": mb_obs, "cmd": mb_cmd, "prev_action": mb_actions}
                if self.uses_history:
                    model_input.update({
                        "obs_hist": obs_hist_flat[mb_idx],
                        "act_hist": act_hist_flat[mb_idx],
                        "hist_mask": hist_mask_flat[mb_idx],
                    })
                if dynamics_flat is not None:
                    model_input["dynamics_targets"] = {
                        k: v[mb_idx] for k, v in dynamics_flat.items()
                    }

                output = self.model(**model_input)

                dist = Normal(output["action_mean"], output["action_log_std"].exp())
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(output["value"], mb_returns)

                aux_loss = output.get("aux_loss", torch.tensor(0.0, device=self.device))
                aux_per_factor = output.get("aux_per_factor", {})

                # --- Gradient instrumentation (first minibatch only) ---
                if should_instrument and not first_mb_instrumented:
                    ppo_loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )

                    loss_dict = {"ppo": ppo_loss}
                    for fname, floss in aux_per_factor.items():
                        loss_dict[f"aux_{fname}"] = self.aux_loss_weight * floss

                    norms, cosines = compute_gradient_stats(self.model, loss_dict)

                    # Total gradient norm
                    total_loss_for_grad = ppo_loss + self.aux_loss_weight * aux_loss
                    try:
                        params = [p for p in self.model.parameters() if p.requires_grad]
                        total_grads = torch.autograd.grad(
                            total_loss_for_grad, params,
                            retain_graph=True, allow_unused=True
                        )
                        total_vec = torch.cat([
                            g.flatten() if g is not None
                            else torch.zeros(p.numel(), device=p.device)
                            for g, p in zip(total_grads, params)
                        ])
                        norms["total"] = float(total_vec.norm().item())
                    except RuntimeError:
                        norms["total"] = 0.0

                    self.gradient_log.append({
                        "iteration": self._update_count,
                        "global_step": self.global_step,
                        "norms": norms,
                        "cosines": cosines,
                    })
                    first_mb_instrumented = True

                # --- Normal PPO update ---
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.aux_loss_weight * aux_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_aux_loss += (
                    aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                )
                total_approx_kl += approx_kl.item()
                n_updates += 1

            if (
                self.lr_schedule == "adaptive"
                and total_approx_kl / n_updates > self.target_kl * 1.5
            ):
                break

        if self.lr_schedule == "adaptive":
            kl = total_approx_kl / n_updates
            if kl > self.target_kl * 2:
                self.lr = max(1e-5, self.lr / 1.5)
            elif kl < self.target_kl / 2:
                self.lr = min(1e-2, self.lr * 1.5)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.lr

        return {
            "loss/policy": total_policy_loss / n_updates,
            "loss/value": total_value_loss / n_updates,
            "loss/entropy": total_entropy / n_updates,
            "loss/aux": total_aux_loss / n_updates,
            "loss/approx_kl": total_approx_kl / n_updates,
            "train/lr": self.lr,
        }


def main():
    parser = argparse.ArgumentParser(description="Gradient Flow Analysis")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output_dir", type=str,
        default="results/mechanistic/gradient_analysis"
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gradient_log_interval", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_config(
        base_path="configs/base.yaml",
        task_path="configs/task/randomized.yaml",
        model_path="configs/model/dynamite.yaml",
        train_path="configs/train/default.yaml",
    )
    cfg["seed"] = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_sim(headless=args.headless)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg, device=device, headless=args.headless)
    model = build_model(cfg)
    model.to(device)

    metrics_csv_path = str(run_dir / "metrics.csv")
    write_step_header(metrics_csv_path)

    logger = Logger(str(run_dir))
    logger.log_config(cfg)

    trainer = InstrumentedPPOTrainer(
        cfg, model, env, logger, str(run_dir),
        variant="gradient_analysis",
        metrics_csv_path=metrics_csv_path,
        gradient_log_interval=args.gradient_log_interval,
    )

    print(f"[GradientFlow] Training DynaMITE seed={args.seed}")
    print(f"[GradientFlow] Log interval: every {args.gradient_log_interval} iterations")
    start = time.time()
    trainer.train()
    wall = time.time() - start
    print(f"[GradientFlow] Training complete: {wall:.0f}s")

    # Save gradient log
    out_file = output_dir / f"seed_{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(trainer.gradient_log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"[GradientFlow] Saved: {out_file} ({len(trainer.gradient_log)} entries)")

    logger.close()
    env.close()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
