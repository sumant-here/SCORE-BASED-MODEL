"""Loss functions for score-based generative modeling and diffusion models."""

import torch
import torch.nn as nn
from src.sde.base import BaseSDE
from src.diffusion.forward_process import forward_diffuse_sde


def get_loss_fn(sde: BaseSDE, eps: float = 1e-5):
    """Factory creating continuous-time denoising score matching loss function.

    Args:
        sde: BaseSDE instance (VPSDE, VESDE, SubVPSDE).
        eps: Minimum timestep to prevent singularity at t=0.

    Returns:
        loss_fn(model, x_0) -> scalar loss.
    """
    def loss_fn(model: nn.Module, x_0: torch.Tensor) -> torch.Tensor:
        """Compute Denoising Score Matching loss for batch x_0.

        Args:
            model: Score network / noise predictor.
            x_0: Clean image batch (B, C, H, W).

        Returns:
            Scalar training loss.
        """
        batch_size = x_0.shape[0]
        # Uniform sampling of continuous time t ~ U(eps, T)
        t = torch.rand(batch_size, device=x_0.device) * (sde.T - eps) + eps

        # Sample noisy image x_t and standard noise z ~ N(0, I)
        x_t, noise, std = forward_diffuse_sde(sde, x_0, t)

        # Forward pass through model
        output = model(x_t, t)

        # Depending on model/SDE parameterization:
        # Standard noise matching objective: || model(x_t, t) - noise ||^2
        # This is equivalent to weighted score matching with lambda(t) = std(t)^2
        loss = torch.mean(torch.sum((output - noise) ** 2, dim=(1, 2, 3)))
        return loss

    return loss_fn
