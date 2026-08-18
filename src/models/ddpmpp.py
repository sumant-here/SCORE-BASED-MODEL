"""DDPM++: Enhanced Score-Based Architecture (Song et al., 2020)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from src.models.embeddings import SinusoidalPositionalEmbedding, TimestepMLP
from src.models.layers import BigGANResnetBlock, AttentionBlock, Downsample, Upsample, get_norm


class DDPMPlusPlus(nn.Module):
    """DDPM++ score network with BigGAN residual blocks and skip scaling (1/sqrt(2)).

    Key characteristics:
    - BigGAN-style residual blocks with skip scaling.
    - Progressive multi-resolution skip connections.
    - High-capacity attention blocks at multiple resolutions.
    - Continuous time/noise conditioning.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        time_emb_dim = base_channels * 4
        self.time_emb_dim = time_emb_dim
        self.time_embed = SinusoidalPositionalEmbedding(base_channels)
        self.time_mlp = TimestepMLP(base_channels, time_emb_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_ch = base_channels
        ch_in_list = [current_ch]
        current_res = 32

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = BigGANResnetBlock(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
                attn = (
                    AttentionBlock(out_ch)
                    if current_res in attention_resolutions
                    else nn.Identity()
                )
                self.down_blocks.append(nn.ModuleDict({"res": block, "attn": attn}))
                current_ch = out_ch
                ch_in_list.append(current_ch)

            if level != len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(current_ch))
                ch_in_list.append(current_ch)
                current_res //= 2

        # Middle blocks
        self.mid_block1 = BigGANResnetBlock(
            in_channels=current_ch,
            out_channels=current_ch,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.mid_attn = AttentionBlock(current_ch)
        self.mid_block2 = BigGANResnetBlock(
            in_channels=current_ch,
            out_channels=current_ch,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = ch_in_list.pop()
                block = BigGANResnetBlock(
                    in_channels=current_ch + skip_ch,
                    out_channels=out_ch,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
                attn = (
                    AttentionBlock(out_ch)
                    if current_res in attention_resolutions
                    else nn.Identity()
                )
                self.up_blocks.append(nn.ModuleDict({"res": block, "attn": attn}))
                current_ch = out_ch

            if level != 0:
                self.up_blocks.append(Upsample(current_ch))
                current_res *= 2

        self.norm_out = get_norm(current_ch)
        self.conv_out = nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass for DDPM++ predicting noise or score."""
        t_emb = self.time_mlp(self.time_embed(timesteps))

        h = self.conv_in(x)
        skips = [h]

        for block in self.down_blocks:
            if isinstance(block, Downsample):
                h = block(h)
                skips.append(h)
            else:
                h = block["res"](h, t_emb)
                h = block["attn"](h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block["res"](h, t_emb)
                h = block["attn"](h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)

    def get_score(self, x: torch.Tensor, timesteps: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Compute score s_theta(x, t) = -epsilon_theta(x, t) / std."""
        eps = self.forward(x, timesteps)
        std_expanded = std.view(-1, *([1] * (x.ndim - 1)))
        return -eps / (std_expanded + 1e-8)
