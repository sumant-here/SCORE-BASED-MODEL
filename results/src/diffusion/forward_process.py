"""Forward diffusion perturbation processes."""

from typing import Tuple
import torch
from src.sde.base import BaseSDE


def forward_diffuse_sde(
    sde: BaseSDE, x_0: torch.Tensor, t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perturb clean image x_0 to noisy state x_t using continuous SDE marginals.

    Args:
        sde: BaseSDE instance (VPSDE, VESDE, SubVPSDE).
        x_0: Clean image tensor (B, C, H, W).
        t: Continuous time tensor (B,).

    Returns:
        x_t: Noisy image tensor (B, C, H, W).
        noise: Standard Gaussian noise tensor z ~ N(0, I).
        std: Marginal standard deviation at time t.
    """
    mean, std = sde.marginal_prob(x_0, t)
    noise = torch.randn_like(x_0)
    std_expanded = std if std.ndim == x_0.ndim else std.view(-1, *([1] * (x_0.ndim - 1)))
    x_t = mean + std_expanded * noise
    return x_t, noise, std


def forward_diffuse_discrete(
    x_0: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor, sqrt_one_minus_alphas_cumprod: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perturb clean image x_0 using discrete DDPM alpha schedules.

    Args:
        x_0: Clean image (B, C, H, W).
        t: Integer timesteps (B,).
        sqrt_alphas_cumprod: (N,) tensor.
        sqrt_one_minus_alphas_cumprod: (N,) tensor.

    Returns:
        x_t: Noisy image.
        noise: Standard normal noise z.
    """
    noise = torch.randn_like(x_0)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    return x_t, noise
