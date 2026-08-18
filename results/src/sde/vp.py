"""Variance Preserving Stochastic Differential Equation (VP-SDE)."""

from typing import Tuple
import torch
import numpy as np
from src.sde.base import BaseSDE


class VPSDE(BaseSDE):
    """Variance Preserving SDE:
    dx = -1/2 * beta(t) * x * dt + sqrt(beta(t)) * dW_t

    where beta(t) = beta_min + t * (beta_max - beta_min).
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000, T: float = 1.0):
        super().__init__(N=N, T=T)
        self.beta_min = beta_min
        self.beta_max = beta_max

    @property
    def sde_type(self) -> str:
        return "VP"

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule: beta(t) = beta_min + t * (beta_max - beta_min)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift f(x, t) = -0.5 * beta(t) * x and diffusion g(t) = sqrt(beta(t))."""
        beta_t = self._beta(t)
        # Broadcast beta_t across spatial/channel dimensions
        beta_t_expanded = beta_t.view(-1, *([1] * (x.ndim - 1)))
        drift = -0.5 * beta_t_expanded * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute transition mean and std:
        integral_beta = 1/2 * (beta_max - beta_min) * t^2 + beta_min * t
        mean = x * exp(-1/2 * integral_beta)
        std = sqrt(1 - exp(-integral_beta))
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff_expanded = log_mean_coeff.view(-1, *([1] * (x.ndim - 1)))
        mean = torch.exp(log_mean_coeff_expanded) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff_expanded))
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from standard Gaussian prior N(0, I)."""
        return torch.randn(shape, device=device)

    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Log probability under standard normal prior N(0, I)."""
        shape = z.shape
        N_dims = int(np.prod(shape[1:]))
        logps = -N_dims / 2.0 * np.log(2.0 * np.pi) - 0.5 * torch.sum(z ** 2, dim=list(range(1, z.ndim)))
        return logps
