"""Variance Exploding Stochastic Differential Equation (VE-SDE)."""

from typing import Tuple
import torch
import numpy as np
from src.sde.base import BaseSDE


class VESDE(BaseSDE):
    """Variance Exploding SDE:
    dx = sqrt(d[sigma^2(t)]/dt) * dW_t

    where sigma(t) = sigma_min * (sigma_max / sigma_min)^t.
    """

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        N: int = 1000,
        T: float = 1.0,
    ):
        super().__init__(N=N, T=T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def sde_type(self) -> str:
        return "VE"

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Geometric noise schedule: sigma(t) = sigma_min * (sigma_max / sigma_min)^t."""
        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** t)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift f(x, t) = 0 and diffusion g(t) = sigma(t) * sqrt(2 * log(sigma_max / sigma_min))."""
        drift = torch.zeros_like(x)
        sigma_t = self._sigma(t)
        diffusion = sigma_t * np.sqrt(2.0 * (np.log(self.sigma_max) - np.log(self.sigma_min)))
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute transition mean and std:
        mean = x
        std = sigma(t)
        """
        mean = x
        std = self._sigma(t)
        std_expanded = std.view(-1, *([1] * (x.ndim - 1)))
        return mean, std_expanded

    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from Gaussian prior N(0, sigma_max^2 * I)."""
        return torch.randn(shape, device=device) * self.sigma_max

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability under prior N(0, sigma_max^2 * I)."""
        shape = z.shape
        N_dims = int(np.prod(shape[1:]))
        logps = (
            -N_dims / 2.0 * np.log(2.0 * np.pi * self.sigma_max ** 2)
            - 0.5 * torch.sum(z ** 2, dim=list(range(1, z.ndim))) / (self.sigma_max ** 2)
        )
        return logps
