"""Unit tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "gpu_memory" in data


def test_api_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    model_names = [m["name"] for m in data]
    assert "ddpm" in model_names
    assert "ddpmpp" in model_names
    assert "ncsnpp" in model_names


def test_api_experiments_endpoint():
    response = client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "total_experiments" in data
    assert "experiments" in data


def test_api_generate_endpoint():
    payload = {
        "model": "ddpm",
        "sde": "vp",
        "num_samples": 2,
        "steps": 5,
        "sampler": "euler",
        "seed": 42,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["images"]) == 2
    assert data["images"][0]["data_url"].startswith("data:image/png;base64,")


def test_api_generate_invalid_model():
    payload = {
        "model": "nonexistent_model",
        "sde": "vp",
        "num_samples": 2,
        "steps": 5,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
