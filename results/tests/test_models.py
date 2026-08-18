"""Unit tests for DDPM, DDPM++, and NCSN++ score model architectures."""

import torch
import pytest
from src.models import get_model, count_parameters
from src.models.embeddings import SinusoidalPositionalEmbedding, GaussianFourierProjection
from src.models.layers import ResnetBlock, BigGANResnetBlock, AttentionBlock


def test_embeddings_shapes():
    t = torch.tensor([0.1, 0.5, 0.9])
    sin_emb = SinusoidalPositionalEmbedding(dim=64)
    out_sin = sin_emb(t)
    assert out_sin.shape == (3, 64)

    fourier_emb = GaussianFourierProjection(embedding_size=64, scale=16.0)
    out_four = fourier_emb(t)
    assert out_four.shape == (3, 64)


def test_layers_forward():
    x = torch.randn(2, 32, 16, 16)
    t_emb = torch.randn(2, 128)

    res = ResnetBlock(in_channels=32, out_channels=32, time_emb_dim=128)
    assert res(x, t_emb).shape == (2, 32, 16, 16)

    biggan = BigGANResnetBlock(in_channels=32, out_channels=64, time_emb_dim=128)
    assert biggan(x, t_emb).shape == (2, 64, 16, 16)

    attn = AttentionBlock(channels=32, num_heads=4)
    assert attn(x).shape == (2, 32, 16, 16)


@pytest.mark.parametrize("model_name", ["ddpm", "ddpmpp", "ncsnpp"])
def test_models_forward_and_score(model_name):
    model = get_model(
        model_name,
        base_channels=16,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0.2, 0.8])
    std = torch.tensor([0.3, 0.7])

    out = model(x, t)
    assert out.shape == (2, 3, 32, 32)

    score = model.get_score(x, t, std)
    assert score.shape == (2, 3, 32, 32)
    assert not torch.isnan(score).any()

    param_count = count_parameters(model)
    assert param_count > 0
