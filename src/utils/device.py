"""Device management and memory monitoring utilities."""

import os
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the appropriate torch.device based on preference and hardware availability.

    Args:
        preference: "auto", "cuda", "mps", or "cpu".

    Returns:
        torch.device instance.
    """
    pref = preference.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage in MB if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        return {
            "device": device_name,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "max_allocated_mb": round(max_allocated, 2),
        }
    return {"device": "cpu", "allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
