"""FastAPI inference and benchmark service for Score-Based Generative Models."""

from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import torch

from src.utils.device import get_device, get_gpu_memory_info
from src.evaluation.metrics import load_metrics_from_csv
from src.experiments.registry import SUPPORTED_MODELS, SUPPORTED_SDES
from api.schemas import (
    GenerationRequest,
    GenerationResponse,
    GeneratedImageItem,
    ModelInfo,
    HealthResponse,
    ExperimentsResponse,
)
from api.inference import get_inference_service

app = FastAPI(
    title="Score-Based Generative Models API",
    description="Inference and Benchmark API for DDPM, DDPM++, and NCSN++ across VP, VE, and Sub-VP SDEs on CIFAR-10",
    version="1.0.0",
)

# CORS middleware for Streamlit / external frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to OpenAPI docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """System health check and GPU telemetry."""
    device = get_device("auto")
    service = get_inference_service()
    loaded = list(service.cache.keys())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        gpu_memory=get_gpu_memory_info(),
        loaded_models=loaded,
    )


@app.get("/models", response_model=List[ModelInfo], tags=["Metadata"])
async def list_models():
    """List supported score-based architectures and SDE formulations."""
    return [
        ModelInfo(
            name="ddpm",
            supported_sdes=["vp", "ve", "subvp"],
            description="Classic Denoising Diffusion Probabilistic Model U-Net with Sinusoidal Positional Embeddings (Ho et al., 2020).",
        ),
        ModelInfo(
            name="ddpmpp",
            supported_sdes=["vp", "ve", "subvp"],
            description="DDPM++ Architecture with BigGAN-style Residual Blocks, 1/sqrt(2) skip scaling, and Progressive Multi-Resolution Features (Song et al., 2020).",
        ),
        ModelInfo(
            name="ncsnpp",
            supported_sdes=["vp", "ve", "subvp"],
            description="NCSN++ Multi-Scale Score Network with Gaussian Random Fourier Feature noise conditioning and explicit score parameterization (Song et al., 2020).",
        ),
    ]


@app.get("/experiments", response_model=ExperimentsResponse, tags=["Ablation"])
async def list_experiments():
    """Get all completed ablation experiments from local results database."""
    csv_path = Path("results/metrics/results.csv")
    df = load_metrics_from_csv(csv_path)
    if df.empty:
        return ExperimentsResponse(total_experiments=0, experiments=[])
    records = df.to_dict(orient="records")
    return ExperimentsResponse(total_experiments=len(records), experiments=records)


@app.get("/metrics", tags=["Ablation"])
async def get_metrics():
    """Get aggregated metrics and summary comparisons."""
    csv_path = Path("results/metrics/results.csv")
    df = load_metrics_from_csv(csv_path)
    if df.empty:
        return {"message": "No experiments recorded yet.", "summary": {}}

    summary = {
        "best_fid_experiment": df.sort_values(by="fid").iloc[0].to_dict() if "fid" in df.columns else {},
        "best_is_experiment": df.sort_values(by="inception_score", ascending=False).iloc[0].to_dict() if "inception_score" in df.columns else {},
        "total_experiments_run": len(df),
    }
    return {"summary": summary, "records": df.to_dict(orient="records")}


@app.post("/generate", response_model=GenerationResponse, tags=["Inference"])
async def generate_images(req: GenerationRequest):
    """Generate CIFAR-10 images using selected model, SDE, sampler, and checkpoint."""
    model_name = req.model.lower().replace("+", "pp")
    sde_name = req.sde.lower().replace("-", "")

    if model_name not in ["ddpm", "ddpmpp", "ncsnpp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model '{req.model}'. Choose from: {SUPPORTED_MODELS}",
        )
    if sde_name not in ["vp", "ve", "subvp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported SDE '{req.sde}'. Choose from: {SUPPORTED_SDES}",
        )

    service = get_inference_service()
    try:
        data_urls, elapsed_sec = service.generate(
            model_name=model_name,
            sde_name=sde_name,
            num_samples=req.num_samples,
            steps=req.steps,
            sampler_name=req.sampler,
            seed=req.seed,
            checkpoint_path=req.checkpoint_path,
        )

        image_items = [GeneratedImageItem(index=i, data_url=url) for i, url in enumerate(data_urls)]

        return GenerationResponse(
            success=True,
            model=req.model,
            sde=req.sde,
            num_samples=req.num_samples,
            sampler=req.sampler,
            sampling_time_seconds=round(elapsed_sec, 3),
            seed=req.seed,
            images=image_items,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )
