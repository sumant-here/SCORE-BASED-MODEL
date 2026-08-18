# Multi-stage production Dockerfile for Score-Based Generative Models
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# System dependencies for OpenCV, PIL, and scientific tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy project code
COPY . .

# Expose ports: FastAPI (8000), Streamlit (8501), MLflow (5000)
EXPOSE 8000 8501 5000

# Default command launches FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
