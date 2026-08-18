"""Unit tests for VP, VE, and Sub-VP SDE mathematical formulations."""

import torch
import pytest
from src.sde import get_sde, VPSDE, VESDE, SubVPSDE


@pytest.mark.parametrize("sde_name,expected_cls", [
    ("vp", VPSDE),
    ("ve", VESDE),
    ("subvp", SubVPSDE),
])
def test_sde_factory_instantiation(sde_name, expected_cls):
    sde = get_sde(sde_name)
    assert isinstance(sde, expected_cls)
    assert sde.T == 1.0


@pytest.mark.parametrize("sde_name", ["vp", "ve", "subvp"])
def test_sde_drift_and_diffusion_shapes(sde_name):
    sde = get_sde(sde_name)
    x = torch.randn(4, 3, 32, 32)
    t = torch.tensor([0.1, 0.4, 0.7, 0.9])

    drift, diffusion = sde.sde(x, t)
    assert drift.shape == (4, 3, 32, 32)
    assert diffusion.shape == (4,)


@pytest.mark.parametrize("sde_name", ["vp", "ve", "subvp"])
def test_sde_marginal_prob_and_prior_sampling(sde_name):
    sde = get_sde(sde_name)
    x = torch.randn(4, 3, 32, 32)
    t = torch.tensor([0.2, 0.5, 0.8, 1.0])

    mean, std = sde.marginal_prob(x, t)
    assert mean.shape == (4, 3, 32, 32)
    assert std.shape == (4, 1, 1, 1) or std.shape == (4,)

    prior_sample = sde.prior_sampling((4, 3, 32, 32), device=torch.device("cpu"))
    assert prior_sample.shape == (4, 3, 32, 32)

    logp = sde.prior_logp(prior_sample)
    assert logp.shape == (4,)
    assert not torch.isnan(logp).any()


def test_sde_discretization():
    sde = get_sde("vp")
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.5, 0.5])
    step_size = 0.01

    f_step, g_step = sde.discretize(x, t, step_size)
    assert f_step.shape == (2, 3, 32, 32)
    assert g_step.shape == (2,)
