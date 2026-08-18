"""SDE formulations package."""

from typing import Union
from src.sde.base import BaseSDE
from src.sde.vp import VPSDE
from src.sde.ve import VESDE
from src.sde.subvp import SubVPSDE


def get_sde(name: str, **kwargs) -> BaseSDE:
    """Instantiate SDE object from formulation name.

    Args:
        name: One of 'vp', 've', 'subvp', 'sub_vp' (case-insensitive).
        **kwargs: Arguments passed to SDE constructor (e.g. beta_min, beta_max, sigma_min, sigma_max, N).

    Returns:
        Instance of BaseSDE.
    """
    clean_name = name.lower().replace("-", "").replace("_", "")
    if clean_name == "vp":
        # Extract relevant args
        beta_min = kwargs.get("beta_min", 0.1)
        beta_max = kwargs.get("beta_max", 20.0)
        N = kwargs.get("num_timesteps", kwargs.get("N", 1000))
        return VPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    elif clean_name == "ve":
        sigma_min = kwargs.get("sigma_min", 0.01)
        sigma_max = kwargs.get("sigma_max", 50.0)
        N = kwargs.get("num_timesteps", kwargs.get("N", 1000))
        return VESDE(sigma_min=sigma_min, sigma_max=sigma_max, N=N)
    elif clean_name == "subvp":
        beta_min = kwargs.get("beta_min", 0.1)
        beta_max = kwargs.get("beta_max", 20.0)
        N = kwargs.get("num_timesteps", kwargs.get("N", 1000))
        return SubVPSDE(beta_min=beta_min, beta_max=beta_max, N=N)
    else:
        raise ValueError(f"Unknown SDE formulation '{name}'. Choose from 'vp', 've', 'subvp'.")


__all__ = ["BaseSDE", "VPSDE", "VESDE", "SubVPSDE", "get_sde"]
