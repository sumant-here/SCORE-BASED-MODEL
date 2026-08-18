"""Model architectures package."""

import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.ddpm import DDPMUNet
from src.models.ddpmpp import DDPMPlusPlus
from src.models.ncsnpp import NCSNPlusPlus
from src.models.unet import BaseUNet


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs) -> nn.Module:
    """Factory to instantiate score models.

    Args:
        name: One of 'ddpm', 'ddpmpp', 'ddpm++', 'ncsnpp', 'ncsn++'.
        **kwargs: Architecture parameters (base_channels, num_res_blocks, etc.).

    Returns:
        nn.Module instance.
    """
    clean_name = name.lower().replace("+", "pp").replace("-", "").replace("_", "")

    in_channels = kwargs.get("in_channels", 3)
    out_channels = kwargs.get("out_channels", 3)
    base_channels = kwargs.get("base_channels", kwargs.get("width", 64))
    channel_multipliers = tuple(kwargs.get("channel_multipliers", kwargs.get("ch_mult", (1, 2, 2, 2))))
    num_res_blocks = kwargs.get("num_res_blocks", kwargs.get("depth", 2))
    attention_resolutions = tuple(kwargs.get("attention_resolutions", (16,)))
    dropout = kwargs.get("dropout", 0.1)

    if clean_name == "ddpm":
        return DDPMUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
    elif clean_name == "ddpmpp":
        return DDPMPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )
    elif clean_name == "ncsnpp":
        fourier_scale = kwargs.get("fourier_scale", 16.0)
        return NCSNPlusPlus(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            fourier_scale=fourier_scale,
        )
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose from 'ddpm', 'ddpmpp', 'ncsnpp'.")


__all__ = [
    "DDPMUNet",
    "DDPMPlusPlus",
    "NCSNPlusPlus",
    "BaseUNet",
    "get_model",
    "count_parameters",
]
