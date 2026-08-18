"""Timestep and continuous noise level embeddings for score-based generative models."""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding as used in DDPM (Ho et al., 2020)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 1D Tensor of shape (B,) containing timestep values.

        Returns:
            Embeddings of shape (B, dim).
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier feature projection for continuous noise level conditioning (Song et al., 2020)."""

    def __init__(self, embedding_size: int = 256, scale: float = 16.0):
        super().__init__()
        # Random Fourier Features (fixed weights)
        self.register_buffer("W", torch.randn(embedding_size // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 1D Tensor of continuous time/noise levels of shape (B,).

        Returns:
            Gaussian Fourier embeddings of shape (B, embedding_size).
        """
        t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class TimestepMLP(nn.Module):
    """Two-layer MLP to project positional/fourier embeddings to model dimension."""

    def __init__(self, embedding_dim: int, hidden_dim: int, act_fn: str = "silu"):
        super().__init__()
        act = nn.SiLU() if act_fn.lower() == "silu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(emb)
