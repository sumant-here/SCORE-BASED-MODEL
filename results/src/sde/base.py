"""Abstract Base Class for Continuous-Time Stochastic Differential Equations (SDEs)."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import numpy as np


class BaseSDE(ABC):
    """Abstract base class for all SDE formulations (VP, VE, Sub-VP)."""

    def __init__(self, N: int = 1000, T: float = 1.0):
        """
        Args:
            N: Number of discretization timesteps.
            T: End time of forward diffusion process (usually 1.0).
        """
        self.N = N
        self.T = T

    @property
    @abstractmethod
    def sde_type(self) -> str:
        """Returns name of SDE ('VP', 'VE', or 'Sub-VP')."""
        pass

    @abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift f(x, t) and diffusion coefficient g(t).

        Args:
            x: Current state tensor of shape (B, C, H, W).
            t: Continuous time tensor of shape (B,).

        Returns:
            f: Drift term of shape (B, C, H, W).
            g: Diffusion coefficient of shape (B,).
        """
        pass

    @abstractmethod
    def marginal_prob(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and standard deviation of transition kernel p_{0t}(x(t) | x(0)).

        Args:
            x: Clean data tensor x(0) of shape (B, C, H, W).
            t: Time tensor of shape (B,).

        Returns:
            mean: Expected value of x(t) given x(0).
            std: Standard deviation of x(t) given x(0).
        """
        pass

    @abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution p_T(x).

        Args:
            shape: Tensor shape (e.g. (B, C, H, W)).
            device: torch.device.

        Returns:
            x_T: Sample from prior distribution.
        """
        pass

    @abstractmethod
    def prior_logp(self, z: torch.Tensor) -> torch.Tensor:
        """Compute analytical log-probability of z under prior p_T.

        Args:
            z: Tensor of shape (B, C, H, W).

        Returns:
            logp: Tensor of shape (B,).
        """
        pass

    def discretize(
        self, x: torch.Tensor, t: torch.Tensor, step_size: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler discretization of the forward SDE:
        x_{i+1} = x_i + f(x_i, t) * dt + g(t) * sqrt(dt) * z

        Args:
            x: State tensor (B, C, H, W).
            t: Time tensor (B,).
            step_size: Time step dt.

        Returns:
            f_step: Drift increment f(x, t) * dt.
            g_step: Diffusion scaling g(t) * sqrt(dt).
        """
        f, g = self.sde(x, t)
        f_step = f * step_size
        g_step = g * np.sqrt(step_size)
        return f_step, g_step
