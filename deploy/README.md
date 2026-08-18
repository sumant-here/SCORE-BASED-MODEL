# Deployment Guide: Score-Based Generative Models

This repository provides production-ready deployment configurations across cloud and container environments.

## 1. Local Containerized Deployment (Docker Compose)

Start FastAPI (port 8000), Streamlit UI (port 8501), and MLflow (port 5000) simultaneously:

```bash
# Build and launch all services in background
docker compose up -d

# View logs
docker compose logs -f

# Teardown
docker compose down
```

### Endpoints
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI Docs**: `http://localhost:8000/docs`
- **MLflow Tracking**: `http://localhost:5000`

---

## 2. GPU Accelerated Docker Deployment

To enable NVIDIA CUDA GPU acceleration inside Docker, ensure `nvidia-container-toolkit` is installed on the host:

Add the GPU reservation block to `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then run:
```bash
docker compose up --build
```

---

## 3. Hugging Face Spaces Deployment (Streamlit)

1. Create a new Space on Hugging Face with **Streamlit** SDK.
2. Push repository code to the Hugging Face repository.
3. Configure `app/streamlit_app.py` as the entrypoint in `README.md` frontmatter:
```yaml
title: Score Based Generative Models Ablation
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
```

---

## 4. Cloud VM Deployment (AWS EC2 / GCP Compute Engine)

```bash
# Clone repository
git clone https://github.com/your-username/score-based-generative-models.git
cd score-based-generative-models

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run inference service with Systemd or Gunicorn/Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```
