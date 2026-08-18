"""Numerical samplers for score-based generative models and reverse-time SDEs."""

import math
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
from src.sde.base import BaseSDE


def euler_maruyama_sampler(
    model: nn.Module,
    sde: BaseSDE,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: Optional[int] = None,
    eps: float = 1e-3,
    denoise: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """Euler-Maruyama numerical integrator for reverse-time SDE:
    dx = [f(x, t) - g(t)^2 * score(x, t)] dt + g(t) dW_bar

    Args:
        model: Score model / noise predictor.
        sde: BaseSDE instance.
        shape: Shape of generated tensor (B, C, H, W).
        device: torch.device.
        num_steps: Discretization steps (default sde.N).
        eps: Small epsilon preventing division by zero at t=0.
        denoise: Whether to remove final marginal noise at t=eps.
        show_progress: Show tqdm progress bar.

    Returns:
        Generated images tensor (B, C, H, W) normalized in [-1, 1].
    """
    N = num_steps if num_steps is not None else sde.N
    dt = (sde.T - eps) / N
    time_steps = torch.linspace(sde.T, eps, N, device=device)

    # Initialize from prior distribution p_T(x)
    x = sde.prior_sampling(shape, device)

    iterator = tqdm(range(N), desc="Euler-Maruyama Sampling", disable=not show_progress)
    with torch.no_grad():
        for i in iterator:
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            # Drift and diffusion from forward SDE
            f, g = sde.sde(x, vec_t)
            _, std = sde.marginal_prob(x, vec_t)

            # Compute score function
            if hasattr(model, "get_score"):
                score = model.get_score(x, vec_t, std)
            else:
                eps_pred = model(x, vec_t)
                std_exp = std.view(-1, *([1] * (x.ndim - 1)))
                score = -eps_pred / (std_exp + 1e-8)

            # Reverse SDE drift
            g_sq = g ** 2
            g_sq_exp = g_sq.view(-1, *([1] * (x.ndim - 1)))
            reverse_drift = f - g_sq_exp * score

            # Update state: x_{t - dt} = x_t - reverse_drift * dt + g * sqrt(dt) * z
            z = torch.randn_like(x) if i < N - 1 else torch.zeros_like(x)
            g_exp = g.view(-1, *([1] * (x.ndim - 1)))
            x = x - reverse_drift * dt + g_exp * math.sqrt(dt) * z

        if denoise:
            # Tweedie denoising step at t = eps
            vec_eps = torch.ones(shape[0], device=device) * eps
            _, std = sde.marginal_prob(x, vec_eps)
            if hasattr(model, "get_score"):
                score = model.get_score(x, vec_eps, std)
            else:
                eps_pred = model(x, vec_eps)
                std_exp = std.view(-1, *([1] * (x.ndim - 1)))
                score = -eps_pred / (std_exp + 1e-8)
            std_exp = std.view(-1, *([1] * (x.ndim - 1)))
            x = x + (std_exp ** 2) * score

    return x


def pc_sampler(
    model: nn.Module,
    sde: BaseSDE,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: Optional[int] = None,
    snr: float = 0.16,
    n_cur_steps: int = 1,
    eps: float = 1e-3,
    show_progress: bool = True,
) -> torch.Tensor:
    """Predictor-Corrector (PC) Sampler combining Reverse SDE Predictor and Langevin Corrector.

    Args:
        model: Score model.
        sde: BaseSDE instance.
        shape: Output shape (B, C, H, W).
        device: torch.device.
        num_steps: Discretization steps.
        snr: Signal-to-noise ratio for Langevin step size.
        n_cur_steps: Number of corrector iterations per predictor step.
        eps: Smallest timestep.
        show_progress: Progress bar flag.

    Returns:
        Generated images tensor (B, C, H, W).
    """
    N = num_steps if num_steps is not None else sde.N
    dt = (sde.T - eps) / N
    time_steps = torch.linspace(sde.T, eps, N, device=device)

    x = sde.prior_sampling(shape, device)

    iterator = tqdm(range(N), desc="Predictor-Corrector Sampling", disable=not show_progress)
    with torch.no_grad():
        for i in iterator:
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            # --- Corrector Step (Langevin Dynamics) ---
            for _ in range(n_cur_steps):
                _, std = sde.marginal_prob(x, vec_t)
                if hasattr(model, "get_score"):
                    score = model.get_score(x, vec_t, std)
                else:
                    eps_pred = model(x, vec_t)
                    std_exp = std.view(-1, *([1] * (x.ndim - 1)))
                    score = -eps_pred / (std_exp + 1e-8)

                z = torch.randn_like(x)
                # Compute adaptive step size based on SNR
                grad_norm = torch.norm(score.reshape(shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(z.reshape(shape[0], -1), dim=-1).mean()
                step_size = (2 * (snr * noise_norm / (grad_norm + 1e-8)) ** 2).item()
                step_size = min(step_size, 0.01)

                x = x + step_size * score + math.sqrt(2 * step_size) * z

            # --- Predictor Step (Reverse Euler-Maruyama) ---
            f, g = sde.sde(x, vec_t)
            _, std = sde.marginal_prob(x, vec_t)
            if hasattr(model, "get_score"):
                score = model.get_score(x, vec_t, std)
            else:
                eps_pred = model(x, vec_t)
                std_exp = std.view(-1, *([1] * (x.ndim - 1)))
                score = -eps_pred / (std_exp + 1e-8)

            g_sq_exp = (g ** 2).view(-1, *([1] * (x.ndim - 1)))
            reverse_drift = f - g_sq_exp * score

            z = torch.randn_like(x) if i < N - 1 else torch.zeros_like(x)
            g_exp = g.view(-1, *([1] * (x.ndim - 1)))
            x = x - reverse_drift * dt + g_exp * math.sqrt(dt) * z

    return x


def ode_sampler(
    model: nn.Module,
    sde: BaseSDE,
    shape: Tuple[int, ...],
    device: torch.device,
    num_steps: Optional[int] = None,
    eps: float = 1e-3,
    show_progress: bool = True,
) -> torch.Tensor:
    """Probability Flow ODE deterministic sampler:
    dx = [f(x, t) - 0.5 * g(t)^2 * score(x, t)] dt
    """
    N = num_steps if num_steps is not None else sde.N
    dt = (sde.T - eps) / N
    time_steps = torch.linspace(sde.T, eps, N, device=device)

    x = sde.prior_sampling(shape, device)

    iterator = tqdm(range(N), desc="Probability Flow ODE Sampling", disable=not show_progress)
    with torch.no_grad():
        for i in iterator:
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            f, g = sde.sde(x, vec_t)
            _, std = sde.marginal_prob(x, vec_t)

            if hasattr(model, "get_score"):
                score = model.get_score(x, vec_t, std)
            else:
                eps_pred = model(x, vec_t)
                std_exp = std.view(-1, *([1] * (x.ndim - 1)))
                score = -eps_pred / (std_exp + 1e-8)

            g_sq_exp = (g ** 2).view(-1, *([1] * (x.ndim - 1)))
            ode_drift = f - 0.5 * g_sq_exp * score

            # Deterministic Euler step
            x = x - ode_drift * dt

    return x


def get_sampler(name: str) -> Callable:
    """Get sampler function by name."""
    clean = name.lower().replace("-", "_").replace(" ", "_")
    if clean in ["euler", "euler_maruyama", "sde"]:
        return euler_maruyama_sampler
    elif clean in ["pc", "predictor_corrector"]:
        return pc_sampler
    elif clean in ["ode", "probability_flow"]:
        return ode_sampler
    else:
        return euler_maruyama_sampler
