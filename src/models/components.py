"""
Shared model components used across all architectures.

- ObsEmbedding: MLP to embed proprioceptive observations
- ActEmbedding: MLP to embed previous actions
- CmdEmbedding: MLP to embed velocity commands
- PolicyHead: MLP outputting action mean (and optionally log_std)
- ValueHead: MLP outputting scalar value
- SinusoidalPE: Sinusoidal positional encoding
- RunningNormalizer: Online mean/var normalization
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP block with configurable layers and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str = "elu",
        output_activation: str | None = None,
        init_type: str = "orthogonal",
        init_gain: float = 1.414,
    ):
        super().__init__()
        act_fn = _get_activation(activation)
        out_act_fn = _get_activation(output_activation) if output_activation else None

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if out_act_fn is not None:
            layers.append(out_act_fn())

        self.net = nn.Sequential(*layers)
        self._init_weights(init_type, init_gain)

    def _init_weights(self, init_type: str, gain: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObsEmbedding(nn.Module):
    """Embed proprioceptive observations to a fixed-dim vector."""

    def __init__(self, obs_dim: int, embed_dim: int, activation: str = "elu"):
        super().__init__()
        self.mlp = MLP(obs_dim, [embed_dim], embed_dim, activation=activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class ActEmbedding(nn.Module):
    """Embed previous actions to a fixed-dim vector."""

    def __init__(self, act_dim: int, embed_dim: int, activation: str = "elu"):
        super().__init__()
        self.mlp = MLP(act_dim, [embed_dim], embed_dim, activation=activation)

    def forward(self, act: torch.Tensor) -> torch.Tensor:
        return self.mlp(act)


class CmdEmbedding(nn.Module):
    """Embed velocity commands to a fixed-dim vector."""

    def __init__(self, cmd_dim: int, embed_dim: int, activation: str = "elu"):
        super().__init__()
        self.mlp = MLP(cmd_dim, [embed_dim], embed_dim, activation=activation)

    def forward(self, cmd: torch.Tensor) -> torch.Tensor:
        return self.mlp(cmd)


class PolicyHead(nn.Module):
    """
    Policy head: outputs action mean.
    Log_std is a learned parameter (not state-dependent).
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], action_dim: int, activation: str = "elu"):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, action_dim, activation=activation)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_mean, action_log_std)."""
        mean = self.mlp(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class ValueHead(nn.Module):
    """Value head: outputs scalar state value."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], activation: str = "elu"):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dims, 1, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class LearnedPE(nn.Module):
    """Learned positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class RunningNormalizer(nn.Module):
    """
    Online running mean/variance normalizer.
    Used to normalize observations and value targets.
    """

    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-5):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0.0))
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update running statistics with a batch of data."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input."""
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


def _get_activation(name: str | None):
    """Get activation class by name."""
    if name is None:
        return None
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
