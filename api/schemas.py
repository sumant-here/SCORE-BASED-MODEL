"""Pydantic schemas and data validation models for FastAPI inference API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Image generation request payload."""

    model: str = Field(default="ddpm", description="Architecture: 'ddpm', 'ddpmpp', 'ncsnpp'")
    sde: str = Field(default="vp", description="SDE formulation: 'vp', 've', 'subvp'")
    num_samples: int = Field(default=16, ge=1, le=64, description="Number of images to generate (1-64)")
    steps: int = Field(default=100, ge=1, le=1000, description="Number of reverse integration steps")
    sampler: str = Field(default="euler", description="Sampler type: 'euler', 'pc', 'ode'")
    seed: Optional[int] = Field(default=42, description="Random seed for deterministic generation")
    checkpoint_path: Optional[str] = Field(default=None, description="Optional path to model checkpoint .pt")


class GeneratedImageItem(BaseModel):
    """Base64 or URL representation of generated image."""

    index: int
    data_url: str


class GenerationResponse(BaseModel):
    """Image generation response."""

    success: bool
    model: str
    sde: str
    num_samples: int
    sampler: str
    sampling_time_seconds: float
    seed: Optional[int]
    images: List[GeneratedImageItem]


class ModelInfo(BaseModel):
    """Model architecture metadata."""

    name: str
    supported_sdes: List[str]
    description: str
    parameters: Optional[int] = None


class HealthResponse(BaseModel):
    """System health and runtime status."""

    status: str
    version: str
    device: str
    cuda_available: bool
    gpu_memory: Dict[str, Any]
    loaded_models: List[str]


class ExperimentItem(BaseModel):
    """Summary record for executed experiment."""

    experiment_id: str
    model: str
    sde: str
    width: int
    depth: int
    fid: float
    inception_score: float
    parameters: int
    training_time: float


class ExperimentsResponse(BaseModel):
    """List of all local ablation experiment results."""

    total_experiments: int
    experiments: List[Dict[str, Any]]
