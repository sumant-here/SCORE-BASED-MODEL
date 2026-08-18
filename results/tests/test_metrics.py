"""Unit tests for FID, Inception Score, and metrics CSV persistence."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.fid import calculate_frechet_distance
from src.evaluation.metrics import ExperimentMetrics, save_metrics_to_csv, load_metrics_from_csv


def test_frechet_distance_identical_distributions():
    """FID between identical distributions should be zero."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.eye(3)
    fid = calculate_frechet_distance(mu, sigma, mu, sigma)
    assert abs(fid) < 1e-4


def test_frechet_distance_shifted_distributions():
    """FID between shifted Gaussians should equal squared Euclidean distance."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([3.0, 4.0])
    sigma = np.eye(2)
    fid = calculate_frechet_distance(mu1, sigma, mu2, sigma)
    assert abs(fid - 25.0) < 1e-3


def test_metrics_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_results.csv"
        metrics = ExperimentMetrics(
            experiment_id="test_exp_1",
            model="ddpm",
            sde="vp",
            width=32,
            depth=2,
            learning_rate=1e-4,
            steps=1000,
            parameters=150000,
            training_time=12.5,
            sampling_time=1.2,
            fid=35.4,
            inception_score=4.2,
            inception_score_std=0.3,
            num_samples=64,
            seed=42,
        )

        save_metrics_to_csv(metrics, csv_file)
        assert csv_file.exists()

        df = load_metrics_from_csv(csv_file)
        assert len(df) == 1
        assert df.iloc[0]["experiment_id"] == "test_exp_1"
        assert df.iloc[0]["fid"] == 35.4
