"""Sample image generation and comparison grids."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from PIL import Image

from src.data.transforms import unnormalize_to_zero_one
from src.sde.base import BaseSDE
from src.diffusion.samplers import get_sampler


def generate_sample_grid(
    model: nn.Module,
    sde: BaseSDE,
    device: torch.device,
    num_samples: int = 16,
    num_steps: int = 100,
    sampler_name: str = "euler",
    save_path: Optional[Union[str, Path]] = None,
    nrow: int = 4,
) -> torch.Tensor:
    """Generate image grid from model and SDE."""
    model.eval()
    sampler_fn = get_sampler(sampler_name)
    shape = (num_samples, 3, 32, 32)

    with torch.no_grad():
        samples = sampler_fn(
            model=model,
            sde=sde,
            shape=shape,
            device=device,
            num_steps=num_steps,
            show_progress=False,
        )

    samples_01 = unnormalize_to_zero_one(samples)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(samples_01, path, nrow=nrow, normalize=False)

    return samples_01


def generate_fixed_seed_comparison(
    models_dict: dict,
    sde: BaseSDE,
    device: torch.device,
    seed: int = 42,
    num_samples: int = 8,
    num_steps: int = 100,
    save_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """Generate side-by-side comparison across models starting from identical initial noise x_T."""
    torch.manual_seed(seed)
    initial_noise = sde.prior_sampling((num_samples, 3, 32, 32), device=device)

    sampler_fn = get_sampler("euler")
    all_model_samples = []

    for name, model in models_dict.items():
        model.eval()
        # Custom reverse loop starting from initial_noise
        x = initial_noise.clone()
        dt = (sde.T - 1e-3) / num_steps
        time_steps = torch.linspace(sde.T, 1e-3, num_steps, device=device)

        with torch.no_grad():
            for i in range(num_steps):
                t = time_steps[i]
                vec_t = torch.ones(num_samples, device=device) * t
                f, g = sde.sde(x, vec_t)
                _, std = sde.marginal_prob(x, vec_t)
                score = model.get_score(x, vec_t, std) if hasattr(model, "get_score") else -model(x, vec_t) / (std.view(-1, 1, 1, 1) + 1e-8)
                reverse_drift = f - (g ** 2).view(-1, 1, 1, 1) * score
                z = torch.randn_like(x) if i < num_steps - 1 else torch.zeros_like(x)
                x = x - reverse_drift * dt + g.view(-1, 1, 1, 1) * (dt ** 0.5) * z

            # Denoise Tweedie
            vec_eps = torch.ones(num_samples, device=device) * 1e-3
            _, std = sde.marginal_prob(x, vec_eps)
            score = model.get_score(x, vec_eps, std) if hasattr(model, "get_score") else -model(x, vec_eps) / (std.view(-1, 1, 1, 1) + 1e-8)
            x = x + (std.view(-1, 1, 1, 1) ** 2) * score

        all_model_samples.append(unnormalize_to_zero_one(x))

    # Concat along row: (num_models * num_samples, 3, 32, 32)
    comparison_tensor = torch.cat(all_model_samples, dim=0)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(comparison_tensor, path, nrow=num_samples, normalize=False)

    return comparison_tensor
