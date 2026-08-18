"""Unit tests for Euler-Maruyama, Predictor-Corrector, and ODE samplers."""

import torch
import pytest
from src.models import get_model
from src.sde import get_sde
from src.diffusion.samplers import euler_maruyama_sampler, pc_sampler, ode_sampler, get_sampler
from src.utils.seed import set_seed


def test_samplers_output_shapes():
    model = get_model("ddpm", base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
    sde = get_sde("vp")
    device = torch.device("cpu")
    shape = (2, 3, 32, 32)

    # 1. Euler-Maruyama
    samples_euler = euler_maruyama_sampler(model, sde, shape, device, num_steps=5, show_progress=False)
    assert samples_euler.shape == shape

    # 2. Predictor-Corrector
    samples_pc = pc_sampler(model, sde, shape, device, num_steps=5, n_cur_steps=1, show_progress=False)
    assert samples_pc.shape == shape

    # 3. Probability Flow ODE
    samples_ode = ode_sampler(model, sde, shape, device, num_steps=5, show_progress=False)
    assert samples_ode.shape == shape


def test_sampler_reproducibility():
    model = get_model("ddpm", base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
    sde = get_sde("vp")
    device = torch.device("cpu")
    shape = (2, 3, 32, 32)

    set_seed(123)
    s1 = euler_maruyama_sampler(model, sde, shape, device, num_steps=5, show_progress=False)

    set_seed(123)
    s2 = euler_maruyama_sampler(model, sde, shape, device, num_steps=5, show_progress=False)

    assert torch.allclose(s1, s2, atol=1e-5)
