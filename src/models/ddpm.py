"""DDPM: Denoising Diffusion Probabilistic Model Architecture (Ho et al., 2020)."""

import torch
import torch.nn as nn
from typing import Tuple
from src.models.unet import BaseUNet


class DDPMUNet(nn.Module):
    """Classic DDPM U-Net architecture.
    Designed for noise prediction epsilon_theta(x_t, t) with sinusoidal embeddings.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks

        self.unet = BaseUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting noise epsilon."""
        return self.unet(x, timesteps)

    def get_score(self, x: torch.Tensor, timesteps: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute score s_theta(x, t) = -epsilon_theta(x, t) / std."""
        eps = self.forward(x, timesteps)
        std_expanded = std.view(-1, *([1] * (x.ndim - 1)))
        return -eps / (std_expanded + 1e-8)
