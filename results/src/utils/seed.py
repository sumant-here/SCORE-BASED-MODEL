"""Reproducibility and deterministic random seed utilities."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seed for Python, NumPy, PyTorch, and CUDA.

    Args:
        seed: Integer seed value.
        deterministic: If True, configure PyTorch CUDA backend for determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def get_rng_states() -> dict:
    """Capture current RNG states across all random libraries for checkpointing."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore RNG states from captured dictionary."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])
