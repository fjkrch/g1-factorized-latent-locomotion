"""
PPO (Proximal Policy Optimization) implementation.

Designed to work with all model architectures (MLP, LSTM, Transformer, DynaMITE).
Handles both single-step and history-conditioned policies.

Uses Isaac Lab's PPO implementation when available, falling back to this
custom implementation for flexibility with auxiliary losses.

Usage:
    from src.algos.ppo import PPOTrainer
    trainer = PPOTrainer(cfg, model, env, logger)
    trainer.train()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.utils.history_buffer import HistoryBuffer
from src.utils.checkpoint import save_checkpoint, find_latest_checkpoint
from src.utils.manifest import create_manifest, save_manifest, update_manifest
from src.utils.metrics import MetricsTracker
from src.utils.metrics_io import append_step_row, STEP_CSV_COLUMNS


class RolloutBuffer:
    """
    Buffer for storing rollout data (observations, actions, rewards, etc)
    for PPO update computation.

    Stores `num_steps` transitions for `num_envs` environments.
    """

    def __init__(self, num_envs: int, num_steps: int, obs_dim: int, act_dim: int,
                 cmd_dim: int, history_len: int, device: str = "cuda",
                 uses_history: bool = False):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.uses_history = uses_history

        # Core buffers
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.cmd = torch.zeros(num_steps, num_envs, cmd_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        # History buffers (for transformer/dynamite)
        if uses_history:
            self.obs_hist = torch.zeros(num_steps, num_envs, history_len, obs_dim, device=device)
            self.act_hist = torch.zeros(num_steps, num_envs, history_len, act_dim, device=device)
            self.hist_mask = torch.zeros(num_steps, num_envs, history_len, dtype=torch.bool, device=device)

        # Dynamics targets (for auxiliary loss)
        self.dynamics_targets: list[dict[str, torch.Tensor] | None] = [None] * num_steps

        # Computed during finalization
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.ptr = 0

    def insert(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        obs_hist: torch.Tensor | None = None,
        act_hist: torch.Tensor | None = None,
        hist_mask: torch.Tensor | None = None,
        dynamics_targets: dict[str, torch.Tensor] | None = None,
    ):
        """Insert one step of data."""
        self.obs[self.ptr] = obs
        self.cmd[self.ptr] = cmd
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done.float()
        self.values[self.ptr] = value

        if self.uses_history and obs_hist is not None:
            self.obs_hist[self.ptr] = obs_hist
            self.act_hist[self.ptr] = act_hist
            self.hist_mask[self.ptr] = hist_mask

        if dynamics_targets is not None:
            self.dynamics_targets[self.ptr] = {
                k: v.clone() for k, v in dynamics_targets.items()
            }

        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        """Compute GAE advantages and discounted returns."""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def reset(self):
        """Reset pointer for next rollout."""
        self.ptr = 0


class PPOTrainer:
    """
    PPO trainer compatible with all model architectures.

    Handles:
    - Rollout collection (with or without history)
    - PPO policy/value updates
    - Auxiliary loss computation (for DynaMITE)
    - Logging, checkpointing, evaluation
    """

    def __init__(self, cfg: dict, model: nn.Module, env, logger, run_dir: str,
                 variant: str = "full", metrics_csv_path: str | None = None):
        self.cfg = cfg
        self.model = model
        self.env = env
        self.logger = logger
        self.run_dir = run_dir
        self.variant = variant
        self.metrics_csv_path = metrics_csv_path or str(Path(run_dir) / "metrics.csv")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        train_cfg = cfg["train"]
        self.num_steps = train_cfg["num_steps"]
        self.num_minibatches = train_cfg["num_minibatches"]
        self.num_epochs = train_cfg["num_epochs"]
        self.lr = train_cfg["learning_rate"]
        self.clip_range = train_cfg["clip_range"]
        self.gamma = train_cfg["gamma"]
        self.lam = train_cfg["lam"]
        self.entropy_coef = train_cfg["entropy_coef"]
        self.value_coef = train_cfg["value_coef"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.total_timesteps = train_cfg["total_timesteps"]
        self.log_interval = train_cfg["log_interval"]
        self.save_interval = train_cfg["save_interval"]
        self.eval_interval = train_cfg["eval_interval"]
        self.target_kl = train_cfg.get("target_kl", 0.01)
        self.lr_schedule = train_cfg.get("lr_schedule", "fixed")

        # Aux loss
        uses_aux = (
            hasattr(model, 'aux_enabled') and model.aux_enabled
            if hasattr(model, 'aux_enabled') else False
        )
        self.uses_aux = uses_aux
        self.aux_loss_weight = cfg["model"].get("auxiliary", {}).get("loss_weight", 0.0) if uses_aux else 0.0

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, eps=1e-5)

        # History buffer
        self.uses_history = getattr(model, 'uses_history', False)
        task_cfg = cfg["task"]["observation"]
        self.history_buffer = None
        if self.uses_history:
            self.history_buffer = HistoryBuffer(
                num_envs=cfg["task"]["num_envs"],
                history_len=task_cfg["history_len"],
                obs_dim=task_cfg["proprioception_dim"],
                act_dim=task_cfg["action_dim"],
                device=self.device,
            )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_envs=cfg["task"]["num_envs"],
            num_steps=self.num_steps,
            obs_dim=task_cfg["proprioception_dim"],
            act_dim=task_cfg["action_dim"],
            cmd_dim=task_cfg["command_dim"],
            history_len=task_cfg["history_len"],
            device=self.device,
            uses_history=self.uses_history,
        )

        # LSTM hidden state
        self.lstm_hidden = None
        if model.model_type == "lstm":
            self.lstm_hidden = model.init_hidden(cfg["task"]["num_envs"], self.device)

        # Metrics
        self.metrics = MetricsTracker()
        self.best_eval_reward = -float("inf")
        self.global_step = 0

        # Per-episode reward tracking (across envs)
        self._episode_rewards = torch.zeros(cfg["task"]["num_envs"], device=self.device)
        self._episode_lengths = torch.zeros(cfg["task"]["num_envs"], device=self.device)
        self._recent_episode_rewards: list[float] = []
        self._recent_episode_lengths: list[float] = []

        # Timing
        self._train_start_time: float = 0.0
        self._last_log_step: int = 0
        self._last_log_time: float = 0.0

        # Manifest
        self.manifest = create_manifest(cfg, run_dir)
        save_manifest(self.manifest, run_dir)

    def train(self):
        """Main training loop."""
        self.model.to(self.device)
        self.model.train()

        num_iterations = self.total_timesteps // (self.num_steps * self.cfg["task"]["num_envs"])
        print(f"[Train] Starting training: {num_iterations} iterations, "
              f"{self.total_timesteps:,} total timesteps")

        self._train_start_time = time.time()
        self._last_log_time = self._train_start_time
        self._last_log_step = self.global_step

        # Initial reset
        reset_data = self.env.reset()
        obs = reset_data["obs"]
        cmd = reset_data["cmd"]
        prev_action = torch.zeros(self.cfg["task"]["num_envs"], self.env.act_dim, device=self.device)

        for iteration in range(1, num_iterations + 1):
            # ─── Collect Rollout ───
            self.model.eval()
            with torch.no_grad():
                for step in range(self.num_steps):
                    # Get history
                    obs_hist, act_hist, hist_mask = None, None, None
                    dynamics_targets = None

                    if self.uses_history:
                        obs_hist, act_hist, hist_mask = self.history_buffer.get()

                    # Forward pass
                    model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
                    if self.uses_history:
                        model_input.update({
                            "obs_hist": obs_hist,
                            "act_hist": act_hist,
                            "hist_mask": hist_mask,
                        })
                    if self.model.model_type == "lstm":
                        model_input["hidden"] = self.lstm_hidden

                    output = self.model(**model_input)

                    # Sample action
                    action_mean = output["action_mean"]
                    action_log_std = output["action_log_std"]
                    dist = Normal(action_mean, action_log_std.exp())
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    value = output["value"]

                    if self.model.model_type == "lstm":
                        self.lstm_hidden = output["hidden"]

                    # Step environment
                    step_data = self.env.step(action)
                    next_obs = step_data["obs"]
                    reward = step_data["reward"]
                    done = step_data["done"]
                    dynamics_targets = step_data.get("dynamics_params", None)

                    # Track per-episode rewards
                    self._episode_rewards += reward
                    self._episode_lengths += 1
                    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    for idx in done_ids:
                        self._recent_episode_rewards.append(self._episode_rewards[idx].item())
                        self._recent_episode_lengths.append(self._episode_lengths[idx].item())
                        self._episode_rewards[idx] = 0
                        self._episode_lengths[idx] = 0

                    # Update history buffer
                    if self.uses_history:
                        self.history_buffer.insert(obs, action)
                        reset_ids = step_data.get("reset_ids", torch.tensor([], device=self.device))
                        if len(reset_ids) > 0:
                            self.history_buffer.reset_envs(reset_ids)

                    # Reset LSTM hidden for done envs
                    if self.model.model_type == "lstm" and done.any():
                        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                        h, c = self.lstm_hidden
                        h[:, done_ids] = 0
                        c[:, done_ids] = 0
                        self.lstm_hidden = (h, c)

                    # Store in rollout buffer
                    self.rollout_buffer.insert(
                        obs=obs, cmd=cmd, action=action,
                        log_prob=log_prob, reward=reward, done=done, value=value,
                        obs_hist=obs_hist, act_hist=act_hist, hist_mask=hist_mask,
                        dynamics_targets=dynamics_targets,
                    )

                    obs = next_obs
                    cmd = step_data.get("cmd", cmd)
                    prev_action = action
                    self.global_step += self.cfg["task"]["num_envs"]

                # Compute last value for GAE
                model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
                if self.uses_history:
                    o, a, m = self.history_buffer.get()
                    model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})
                if self.model.model_type == "lstm":
                    model_input["hidden"] = self.lstm_hidden
                last_output = self.model(**model_input)
                last_value = last_output["value"]

            self.rollout_buffer.compute_returns_and_advantages(last_value, self.gamma, self.lam)

            # ─── PPO Update ───
            self.model.train()
            update_stats = self._ppo_update()

            self.rollout_buffer.reset()

            # ─── Logging ───
            if iteration % self.log_interval == 0:
                now = time.time()
                wall_time_s = now - self._train_start_time
                steps_since_last = self.global_step - self._last_log_step
                time_since_last = now - self._last_log_time
                fps = steps_since_last / max(time_since_last, 1e-6)

                # GPU memory
                gpu_mem_mb = 0
                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)

                # Episode-level reward (from tracked completions)
                if self._recent_episode_rewards:
                    import numpy as _np
                    reward_mean = float(_np.mean(self._recent_episode_rewards))
                    reward_std = float(_np.std(self._recent_episode_rewards))
                    ep_len_mean = float(_np.mean(self._recent_episode_lengths))
                else:
                    reward_mean = self.rollout_buffer.rewards.sum(dim=0).mean().item()
                    reward_std = 0.0
                    ep_len_mean = 0.0

                # Write standardized step-level CSV row
                step_row = {
                    "iteration": iteration,
                    "global_step": self.global_step,
                    "wall_time_s": round(wall_time_s, 1),
                    "reward_mean": round(reward_mean, 4),
                    "reward_std": round(reward_std, 4),
                    "episode_length_mean": round(ep_len_mean, 1),
                    "policy_loss": round(update_stats["loss/policy"], 6),
                    "value_loss": round(update_stats["loss/value"], 6),
                    "entropy": round(update_stats["loss/entropy"], 6),
                    "approx_kl": round(update_stats["loss/approx_kl"], 6),
                    "aux_loss": round(update_stats["loss/aux"], 6),
                    "learning_rate": update_stats["train/lr"],
                    "fps": round(fps, 0),
                    "gpu_mem_mb": gpu_mem_mb,
                }
                append_step_row(self.metrics_csv_path, step_row)

                # Also log to TensorBoard + console via Logger
                log_data = {
                    "iteration": iteration,
                    "global_step": self.global_step,
                    "reward/mean": reward_mean,
                    "reward/std": reward_std,
                    "perf/fps": fps,
                    "perf/gpu_mem_mb": gpu_mem_mb,
                    **update_stats,
                }
                self.logger.log_dict(log_data, step=self.global_step)

                # Reset trackers
                self._recent_episode_rewards.clear()
                self._recent_episode_lengths.clear()
                self._last_log_step = self.global_step
                self._last_log_time = now

            # ─── Checkpointing ───
            if iteration % self.save_interval == 0:
                save_checkpoint(
                    self.run_dir, self.model, self.optimizer,
                    self.global_step, self.cfg,
                    stats={"iteration": iteration},
                )

            # ─── Evaluation ───
            if iteration % self.eval_interval == 0:
                eval_reward = self._evaluate()
                is_best = eval_reward > self.best_eval_reward
                if is_best:
                    self.best_eval_reward = eval_reward
                    save_checkpoint(
                        self.run_dir, self.model, self.optimizer,
                        self.global_step, self.cfg, is_best=True,
                    )
                self.logger.log_dict({"eval/reward_mean": eval_reward}, step=self.global_step)

        # Final save
        save_checkpoint(
            self.run_dir, self.model, self.optimizer,
            self.global_step, self.cfg,
            stats={"status": "completed"},
        )
        wall_time = time.time() - self._train_start_time
        update_manifest(self.run_dir, {
            "status": "completed",
            "training": {
                "final_step": self.global_step,
                "total_iterations": num_iterations,
                "wall_time_s": round(wall_time, 1),
                "best_eval_reward": self.best_eval_reward,
            },
        })
        print(f"[Train] Training completed. {self.global_step:,} timesteps in {wall_time:.0f}s.")

    def _ppo_update(self) -> dict[str, float]:
        """Perform PPO update using collected rollout data."""
        buf = self.rollout_buffer
        batch_size = buf.num_steps * buf.num_envs
        minibatch_size = batch_size // self.num_minibatches

        # Flatten rollout data
        obs_flat = buf.obs.reshape(-1, buf.obs.shape[-1])
        cmd_flat = buf.cmd.reshape(-1, buf.cmd.shape[-1])
        actions_flat = buf.actions.reshape(-1, buf.actions.shape[-1])
        old_log_probs = buf.log_probs.reshape(-1)
        advantages = buf.advantages.reshape(-1)
        returns = buf.returns.reshape(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # History data
        obs_hist_flat, act_hist_flat, hist_mask_flat = None, None, None
        if self.uses_history:
            obs_hist_flat = buf.obs_hist.reshape(-1, *buf.obs_hist.shape[2:])
            act_hist_flat = buf.act_hist.reshape(-1, *buf.act_hist.shape[2:])
            hist_mask_flat = buf.hist_mask.reshape(-1, buf.hist_mask.shape[-1])

        # Dynamics targets
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

        for epoch in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                # Get minibatch data
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

                # Action distribution
                dist = Normal(output["action_mean"], output["action_log_std"].exp())
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(output["value"], mb_returns)

                # Auxiliary loss (DynaMITE only)
                aux_loss = output.get("aux_loss", torch.tensor(0.0, device=self.device))

                # Total loss
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

                # Track
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                total_approx_kl += approx_kl.item()
                n_updates += 1

            # Early stopping on KL
            if self.lr_schedule == "adaptive" and total_approx_kl / n_updates > self.target_kl * 1.5:
                break

        # LR schedule
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

    def _evaluate(self, num_episodes: int | None = None) -> float:
        """Run evaluation episodes and return mean reward."""
        if num_episodes is None:
            num_episodes = self.cfg["eval"]["num_episodes"]

        self.model.eval()
        total_rewards = []

        with torch.no_grad():
            reset_data = self.env.reset()
            obs = reset_data["obs"]
            cmd = reset_data["cmd"]
            prev_action = torch.zeros(self.env.num_envs, self.env.act_dim, device=self.device)

            episode_rewards = torch.zeros(self.env.num_envs, device=self.device)
            completed = 0

            if self.uses_history:
                self.history_buffer.reset_all()

            while completed < num_episodes:
                model_input = {"obs": obs, "cmd": cmd, "prev_action": prev_action}
                if self.uses_history:
                    o, a, m = self.history_buffer.get()
                    model_input.update({"obs_hist": o, "act_hist": a, "hist_mask": m})

                output = self.model(**model_input)
                action = output["action_mean"]  # deterministic

                step_data = self.env.step(action)
                obs = step_data["obs"]
                cmd = step_data.get("cmd", cmd)
                done = step_data["done"]

                episode_rewards += step_data["reward"]

                if self.uses_history:
                    self.history_buffer.insert(obs, action)
                    reset_ids = step_data.get("reset_ids", torch.tensor([], device=self.device))
                    if len(reset_ids) > 0:
                        self.history_buffer.reset_envs(reset_ids)

                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                for idx in done_ids:
                    total_rewards.append(episode_rewards[idx].item())
                    episode_rewards[idx] = 0
                    completed += 1

                prev_action = action

        self.model.train()
        mean_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        return mean_reward
