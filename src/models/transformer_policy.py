"""
PPO + Vanilla Transformer baseline policy.

Processes observation-action history via a small transformer encoder.
NO explicit latent dynamics inference — the transformer output is
directly fed to policy/value heads.

Architecture:
    For each timestep t in history:
        token_t = [ObsEmbed(obs_t), ActEmbed(act_t), CmdEmbed(cmd_t)]  (concatenated)
    tokens -> Linear(d_model) -> PositionalEncoding -> TransformerEncoder
    aggregated_output -> PolicyHead -> action_mean
    aggregated_output -> ValueHead -> value

Approximate parameter count: ~400k
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.components import (
    ActEmbedding,
    CmdEmbedding,
    LearnedPE,
    MLP,
    ObsEmbedding,
    PolicyHead,
    SinusoidalPE,
    ValueHead,
    count_parameters,
)


class TransformerPolicy(nn.Module):
    """
    Vanilla Transformer policy for PPO.

    Processes a fixed-length window of recent observation-action pairs.
    Uses standard transformer encoder, NO latent dynamics inference.
    The aggregated transformer output (mean-pooled) goes directly to heads.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        obs_dim = cfg["task"]["observation"]["proprioception_dim"]
        cmd_dim = cfg["task"]["observation"]["command_dim"]
        act_dim = cfg["task"]["observation"]["action_dim"]

        model_cfg = cfg["model"]
        self.history_len = model_cfg["history_len"]
        d_model = model_cfg["transformer"]["d_model"]
        nhead = model_cfg["transformer"]["nhead"]
        num_layers = model_cfg["transformer"]["num_layers"]
        dim_ff = model_cfg["transformer"]["dim_feedforward"]
        dropout = model_cfg["transformer"]["dropout"]
        pe_type = model_cfg["transformer"].get("positional_encoding", "sinusoidal")

        obs_embed_dim = model_cfg["obs_embed"]["hidden_dim"]
        act_embed_dim = model_cfg["act_embed"]["hidden_dim"]
        cmd_embed_dim = model_cfg["cmd_embed"]["hidden_dim"]

        # Embeddings
        self.obs_embed = ObsEmbedding(obs_dim, obs_embed_dim)
        self.act_embed = ActEmbedding(act_dim, act_embed_dim)
        self.cmd_embed = CmdEmbedding(cmd_dim, cmd_embed_dim)

        # Project concatenated embeddings to d_model
        token_dim = obs_embed_dim + act_embed_dim + cmd_embed_dim
        self.token_proj = nn.Linear(token_dim, d_model)

        # Positional encoding
        if pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(d_model, max_len=self.history_len + 4)
        else:
            self.pos_enc = LearnedPE(d_model, max_len=self.history_len + 4)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy head
        policy_cfg = model_cfg["policy_head"]
        self.policy = PolicyHead(
            input_dim=d_model,
            hidden_dims=policy_cfg["hidden_dims"],
            action_dim=act_dim,
            activation=policy_cfg["activation"],
        )

        # Value head
        value_cfg = model_cfg["value_head"]
        self.value = ValueHead(
            input_dim=d_model,
            hidden_dims=value_cfg["hidden_dims"],
            activation=value_cfg["activation"],
        )

        self.model_type = "transformer"
        self.uses_history = True

    def _build_tokens(
        self,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build token sequence from history.

        Args:
            obs_hist: (batch, history_len, obs_dim)
            act_hist: (batch, history_len, act_dim)
            cmd: (batch, cmd_dim) — broadcast to all timesteps

        Returns:
            tokens: (batch, history_len, d_model)
        """
        batch_size = obs_hist.shape[0]
        seq_len = obs_hist.shape[1]

        obs_emb = self.obs_embed(obs_hist.reshape(-1, obs_hist.shape[-1]))
        obs_emb = obs_emb.reshape(batch_size, seq_len, -1)

        act_emb = self.act_embed(act_hist.reshape(-1, act_hist.shape[-1]))
        act_emb = act_emb.reshape(batch_size, seq_len, -1)

        cmd_expanded = cmd.unsqueeze(1).expand(-1, seq_len, -1)
        cmd_emb = self.cmd_embed(cmd_expanded.reshape(-1, cmd.shape[-1]))
        cmd_emb = cmd_emb.reshape(batch_size, seq_len, -1)

        tokens = torch.cat([obs_emb, act_emb, cmd_emb], dim=-1)
        tokens = self.token_proj(tokens)
        return tokens

    def forward(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        hist_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) current observation (unused, folded into history)
            cmd: (batch, cmd_dim) velocity command
            obs_hist: (batch, history_len, obs_dim)
            act_hist: (batch, history_len, act_dim)
            hist_mask: (batch, history_len) True=valid, False=padding

        Returns:
            dict with: action_mean, action_log_std, value
        """
        tokens = self._build_tokens(obs_hist, act_hist, cmd)
        tokens = self.pos_enc(tokens)

        # Transformer expects src_key_padding_mask where True = ignored
        padding_mask = None
        if hist_mask is not None:
            padding_mask = ~hist_mask  # invert: True=padding

        features = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Aggregate: mean pool over valid positions
        if hist_mask is not None:
            mask_expanded = hist_mask.unsqueeze(-1).float()
            pooled = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        action_mean, action_log_std = self.policy(pooled)
        value = self.value(pooled)

        return {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
        }

    def get_param_count(self) -> int:
        return count_parameters(self)
