"""
DynaMITE: Dynamic Mismatch Inference via Transformer Encoder.

PROPOSED METHOD — the core contribution.

Key differences from vanilla Transformer policy:
1. Explicit latent dynamics inference head
2. Factorized latent space: z = [z_friction, z_mass, z_motor, z_contact, z_delay]
3. Auxiliary dynamics identification losses (supervised from domain randomization GT)
4. Latent z is concatenated to policy/value head inputs
5. Interpretable: latent factors can be correlated with actual dynamics parameters

Architecture:
    History tokens -> TransformerEncoder -> aggregated features
    features -> FactorizedLatentHead -> z, {z_k}
    [aggregated_features, z] -> PolicyHead -> action_mean
    [aggregated_features, z] -> ValueHead -> value
    {z_k} -> AuxiliaryHead -> predicted dynamics params (training only)

Approximate parameter count: ~450k (still feasible on RTX 4060)
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
from src.models.latent_heads import (
    AuxiliaryIdentificationHead,
    FactorizedLatentHead,
    LatentHead,
)


class DynaMITEPolicy(nn.Module):
    """
    DynaMITE policy for PPO.

    A lightweight transformer that explicitly infers a factorized latent
    dynamics context from short observation-action history, then conditions
    the policy and value heads on this latent.

    The latent space is structured to capture different aspects of dynamics
    mismatch (friction, mass, motor strength, contact, delay), and auxiliary
    identification losses encourage meaningful latent representations.
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

        latent_cfg = model_cfg["latent"]
        aux_cfg = model_cfg["auxiliary"]
        self.use_latent = latent_cfg["total_dim"] > 0
        self.factorized = latent_cfg.get("factorized", False)
        self.aggregation = latent_cfg.get("aggregation", "mean")
        self.aux_enabled = aux_cfg.get("enabled", False)
        self.aux_loss_weight = aux_cfg.get("loss_weight", 0.5)

        # ─── Embeddings ───
        self.obs_embed = ObsEmbedding(obs_dim, obs_embed_dim)
        self.act_embed = ActEmbedding(act_dim, act_embed_dim)
        self.cmd_embed = CmdEmbedding(cmd_dim, cmd_embed_dim)

        token_dim = obs_embed_dim + act_embed_dim + cmd_embed_dim
        self.token_proj = nn.Linear(token_dim, d_model)

        # ─── Positional Encoding ───
        if pe_type == "sinusoidal":
            self.pos_enc = SinusoidalPE(d_model, max_len=self.history_len + 4)
        else:
            self.pos_enc = LearnedPE(d_model, max_len=self.history_len + 4)

        # ─── Transformer Encoder ───
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ─── Attention Aggregation (optional) ───
        if self.aggregation == "attention":
            self.agg_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.agg_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

        # ─── Latent Dynamics Head ───
        latent_dim = latent_cfg["total_dim"]
        if self.use_latent:
            if self.factorized:
                factors = latent_cfg["factors"]  # dict: name -> dim
                self.latent_head = FactorizedLatentHead(
                    input_dim=d_model,
                    factors=factors,
                    hidden_dim=64,
                    activation=latent_cfg.get("bottleneck_activation", "tanh"),
                )
            else:
                self.latent_head = LatentHead(
                    input_dim=d_model,
                    latent_dim=latent_dim,
                    hidden_dim=64,
                    activation=latent_cfg.get("bottleneck_activation", "tanh"),
                )

        # ─── Auxiliary Identification Head ───
        if self.aux_enabled and self.use_latent:
            if self.factorized:
                factor_configs = {}
                for name, ldim in latent_cfg["factors"].items():
                    tdim = aux_cfg["targets"].get(name, 1)
                    factor_configs[name] = {"latent_dim": ldim, "target_dim": tdim}
            else:
                # Single latent: one combined head
                combined_target_dim = sum(aux_cfg["targets"].values())
                factor_configs = {"combined": {"latent_dim": latent_dim, "target_dim": combined_target_dim}}
            self.aux_head = AuxiliaryIdentificationHead(
                factor_configs=factor_configs,
                hidden_dim=aux_cfg.get("head_hidden_dim", 32),
            )

        # ─── Policy and Value Heads ───
        policy_cfg = model_cfg["policy_head"]
        value_cfg = model_cfg["value_head"]
        condition_on_latent = policy_cfg.get("condition_on_latent", True) and self.use_latent
        self.condition_on_latent = condition_on_latent

        head_input_dim = d_model + (latent_dim if condition_on_latent else 0)

        self.policy = PolicyHead(
            input_dim=head_input_dim,
            hidden_dims=policy_cfg["hidden_dims"],
            action_dim=act_dim,
            activation=policy_cfg["activation"],
        )

        self.value = ValueHead(
            input_dim=head_input_dim,
            hidden_dims=value_cfg["hidden_dims"],
            activation=value_cfg["activation"],
        )

        self.model_type = "dynamite"
        self.uses_history = True
        self._d_model = d_model

    def _build_tokens(
        self,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Build token sequence from observation-action history."""
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

    def _aggregate(self, features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """
        Aggregate transformer output across sequence dimension.

        Args:
            features: (batch, seq_len, d_model)
            mask: (batch, seq_len) True=valid
        """
        if self.aggregation == "last":
            if mask is not None:
                # Get last valid position
                lengths = mask.sum(dim=1).long() - 1
                batch_idx = torch.arange(features.shape[0], device=features.device)
                return features[batch_idx, lengths]
            return features[:, -1]

        elif self.aggregation == "attention":
            query = self.agg_query.expand(features.shape[0], -1, -1)
            key_padding = ~mask if mask is not None else None
            out, _ = self.agg_attn(query, features, features, key_padding_mask=key_padding)
            return out.squeeze(1)

        else:  # mean
            if mask is not None:
                mask_exp = mask.unsqueeze(-1).float()
                return (features * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
            return features.mean(dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        cmd: torch.Tensor,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        hist_mask: torch.Tensor | None = None,
        dynamics_targets: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) current observation
            cmd: (batch, cmd_dim) velocity command
            obs_hist: (batch, history_len, obs_dim) observation history
            act_hist: (batch, history_len, act_dim) action history
            hist_mask: (batch, history_len) True=valid
            dynamics_targets: dict of factor_name -> (batch, target_dim) GT params
                              Only needed during training for auxiliary loss.

        Returns:
            dict with: action_mean, action_log_std, value, latent_z,
                       latent_factors, aux_loss, aux_per_factor
        """
        # Build and encode tokens
        tokens = self._build_tokens(obs_hist, act_hist, cmd)
        tokens = self.pos_enc(tokens)

        padding_mask = ~hist_mask if hist_mask is not None else None
        features = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Aggregate across sequence
        pooled = self._aggregate(features, hist_mask)  # (batch, d_model)

        # Latent dynamics inference
        latent_z = None
        latent_factors = None
        aux_loss = torch.tensor(0.0, device=obs.device)
        aux_per_factor = {}

        if self.use_latent:
            if self.factorized:
                latent_z, latent_factors = self.latent_head(pooled)
            else:
                latent_z = self.latent_head(pooled)
                latent_factors = {"combined": latent_z}

            # Auxiliary identification loss
            if self.aux_enabled and dynamics_targets is not None and self.training:
                aux_loss, aux_per_factor = self.aux_head(latent_factors, dynamics_targets)

        # Build head inputs
        if self.condition_on_latent and latent_z is not None:
            head_input = torch.cat([pooled, latent_z], dim=-1)
        else:
            head_input = pooled

        action_mean, action_log_std = self.policy(head_input)
        value = self.value(head_input)

        return {
            "action_mean": action_mean,
            "action_log_std": action_log_std,
            "value": value,
            "latent_z": latent_z,
            "latent_factors": latent_factors,
            "aux_loss": aux_loss,
            "aux_per_factor": aux_per_factor,
        }

    def get_latent(
        self,
        obs_hist: torch.Tensor,
        act_hist: torch.Tensor,
        cmd: torch.Tensor,
        hist_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Extract latent dynamics vector (for analysis/visualization).

        Returns:
            latent_z: (batch, latent_dim)
            latent_factors: dict or None
        """
        tokens = self._build_tokens(obs_hist, act_hist, cmd)
        tokens = self.pos_enc(tokens)
        padding_mask = ~hist_mask if hist_mask is not None else None
        features = self.transformer(tokens, src_key_padding_mask=padding_mask)
        pooled = self._aggregate(features, hist_mask)

        if self.factorized:
            return self.latent_head(pooled)
        else:
            z = self.latent_head(pooled)
            return z, None

    def get_param_count(self) -> int:
        return count_parameters(self)
