"""Neural network building blocks: ResNet blocks, Attention, Down/Up-sampling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with safe group count for smaller channel widths."""
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=1e-5)


class Downsample(nn.Module):
    """Spatial downsampling by a factor of 2 via strided 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling by a factor of 2 via nearest interpolation followed by 3x3 conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Standard ResNet block with timestep conditioning projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = get_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = get_norm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject time conditioning
        t_emb = self.time_proj(time_emb)[:, :, None, None]
        h = h + t_emb
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class BigGANResnetBlock(nn.Module):
    """BigGAN-style ResNet block with 1/sqrt(2) skip scaling used in DDPM++."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = get_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = get_norm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        t_emb = self.time_proj(time_emb)[:, :, None, None]
        h = h + t_emb
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        # BigGAN 1/sqrt(2) residual scale
        return (h + self.skip(x)) / math.sqrt(2.0)


class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention block over spatial dimensions."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = get_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, head_dim, HW)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out
