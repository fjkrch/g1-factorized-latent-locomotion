"""
Latent dynamics inference heads for DynaMITE.

LatentHead: Projects transformer output -> single latent vector z.
FactorizedLatentHead: Projects transformer output -> factorized latent z = [z_1, ..., z_K].
AuxiliaryHead: Predicts ground-truth dynamics parameters from latent z (for auxiliary loss).

These are the KEY NOVEL COMPONENTS that distinguish DynaMITE from a vanilla transformer policy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import MLP


class LatentHead(nn.Module):
    """
    Single (non-factorized) latent dynamics inference head.

    Takes aggregated transformer output and projects to a latent vector z.
    z is intended to capture the hidden dynamics state (friction, mass, etc).

    Architecture:
        transformer_output (d_model) -> Linear -> activation -> Linear -> z (latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        activation: str = "tanh",
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.activation = _get_bottleneck_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model) aggregated transformer output.
        Returns:
            z: (batch, latent_dim) latent dynamics vector.
        """
        z = self.projection(x)
        if self.activation is not None:
            z = self.activation(z)
        return z


class FactorizedLatentHead(nn.Module):
    """
    Factorized latent dynamics inference head.

    Projects transformer output to multiple sub-latent vectors,
    each intended to capture a different dynamics factor:
      z = [z_friction, z_mass, z_motor, z_contact, z_delay]

    Each factor has its own small projection head to encourage
    disentanglement.

    Architecture per factor:
        transformer_output (d_model) -> Linear -> ELU -> Linear -> z_k (factor_dim)
    """

    def __init__(
        self,
        input_dim: int,
        factors: dict[str, int],     # e.g. {"friction": 4, "mass": 6, ...}
        hidden_dim: int = 64,
        activation: str = "tanh",
    ):
        super().__init__()
        self.factor_names = list(factors.keys())
        self.factor_dims = factors

        self.heads = nn.ModuleDict()
        for name, dim in factors.items():
            self.heads[name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, dim),
            )

        self.activation = _get_bottleneck_activation(activation)
        self.total_dim = sum(factors.values())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, d_model) aggregated transformer output.
        Returns:
            z: (batch, total_latent_dim) concatenated latent vector.
            z_factors: dict mapping factor name -> (batch, factor_dim).
        """
        z_factors = {}
        z_parts = []
        for name in self.factor_names:
            z_k = self.heads[name](x)
            if self.activation is not None:
                z_k = self.activation(z_k)
            z_factors[name] = z_k
            z_parts.append(z_k)

        z = torch.cat(z_parts, dim=-1)
        return z, z_factors


class AuxiliaryIdentificationHead(nn.Module):
    """
    Auxiliary head for dynamics parameter identification.

    Predicts ground-truth dynamics parameters from the inferred latent.
    This provides a supervised signal that encourages the latent to
    actually encode useful dynamics information.

    During training with domain randomization, ground-truth dynamics
    parameters are available and used as targets.

    Can operate on:
    - The full concatenated latent z (single latent mode)
    - Individual factor sub-latents (factorized mode)
    """

    def __init__(
        self,
        factor_configs: dict[str, dict[str, int]],
        # e.g. {"friction": {"latent_dim": 4, "target_dim": 2}, ...}
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.factor_names = list(factor_configs.keys())
        self.heads = nn.ModuleDict()

        for name, dims in factor_configs.items():
            latent_dim = dims["latent_dim"]
            target_dim = dims["target_dim"]
            self.heads[name] = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, target_dim),
            )

    def forward(
        self,
        z_factors: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute auxiliary identification losses.

        Args:
            z_factors: dict of factor name -> (batch, factor_dim) latent vectors.
            targets: dict of factor name -> (batch, target_dim) ground-truth params.

        Returns:
            total_loss: scalar, sum of per-factor MSE losses.
            per_factor_losses: dict of factor name -> scalar loss.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        per_factor_losses = {}

        for name in self.factor_names:
            if name not in z_factors or name not in targets:
                continue
            pred = self.heads[name](z_factors[name])
            target = targets[name]
            loss = F.mse_loss(pred, target)
            per_factor_losses[name] = loss
            total_loss = total_loss + loss

        return total_loss, per_factor_losses

    def predict(self, z_factors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict dynamics parameters from latent (for analysis/visualization)."""
        predictions = {}
        for name in self.factor_names:
            if name in z_factors:
                predictions[name] = self.heads[name](z_factors[name])
        return predictions


def _get_bottleneck_activation(name: str | None):
    """Get bottleneck activation function."""
    if name is None or name == "none":
        return None
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softsign":
        return nn.Softsign()
    else:
        raise ValueError(f"Unknown bottleneck activation: {name}")
