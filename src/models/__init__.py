"""
Model registry and factory.

Usage:
    from src.models import build_model
    model = build_model(cfg)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.mlp_policy import MLPPolicy
from src.models.lstm_policy import LSTMPolicy
from src.models.transformer_policy import TransformerPolicy
from src.models.dynamite_policy import DynaMITEPolicy
from src.models.components import count_parameters

MODEL_REGISTRY = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
    "transformer": TransformerPolicy,
    "dynamite": DynaMITEPolicy,
}


def build_model(cfg: dict) -> nn.Module:
    """
    Build a model from config.

    Args:
        cfg: Full merged config dict. model.type determines the class.

    Returns:
        Instantiated nn.Module.
    """
    model_type = cfg["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[model_type](cfg)
    print(f"[Model] Built {model_type} with {count_parameters(model):,} parameters")
    return model
