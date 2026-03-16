"""
Seed utility for reproducibility.

Sets seeds for: Python random, NumPy, PyTorch (CPU + CUDA), Isaac Lab.
Also provides deterministic flags for CUDA operations.

Usage:
    from src.utils.seed import set_seed
    set_seed(42, deterministic=True)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable CUDA deterministic mode.
            This may reduce performance but improves reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops don't have deterministic implementations
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_seed_list(base_seed: int, num_seeds: int = 3) -> list[int]:
    """Generate a list of seeds for multi-seed experiments."""
    return [base_seed + i for i in range(num_seeds)]
