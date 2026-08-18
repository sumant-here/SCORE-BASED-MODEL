"""Fréchet Inception Distance (FID) computation for generative models (Heusel et al., 2017)."""

import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Union
from src.evaluation.inception_score import InceptionFeatureExtractor


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate the Fréchet distance between two multivariate Gaussians:
    d^2 = ||mu_1 - mu_2||_2^2 + Tr(sigma_1 + sigma_2 - 2 * sqrt(sigma_1 * sigma_2))

    Args:
        mu1, mu2: Mean vectors.
        sigma1, sigma2: Covariance matrices.
        eps: Small epsilon for numerical stability.

    Returns:
        FID scalar distance (lower is better).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if isinstance(covmean, tuple):
        covmean = covmean[0]

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]

    # Numerical imaginary artifact removal
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # keep real part if imaginary component is small
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)


def compute_statistics_from_tensors(
    images: torch.Tensor,
    feature_extractor: InceptionFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of feature activations from image tensor."""
    if images.min() < 0:
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)

    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    act_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            feats, _ = feature_extractor(batch)
            act_list.append(feats.cpu().numpy())

    activations = np.concatenate(act_list, axis=0)
    mu = np.mean(activations, axis=0)
    # Check if number of samples is smaller than feature dim
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    feature_extractor: Optional[InceptionFeatureExtractor] = None,
) -> float:
    """Calculate Fréchet Inception Distance between real and generated images."""
    if device is None:
        device = real_images.device

    if feature_extractor is None:
        feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()

    mu_real, sigma_real = compute_statistics_from_tensors(real_images, feature_extractor, device, batch_size)
    mu_gen, sigma_gen = compute_statistics_from_tensors(generated_images, feature_extractor, device, batch_size)

    return calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
