.PHONY: help install download-data train sample evaluate ablation plots test api app docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  make install        Install Python dependencies"
	@echo "  make download-data  Download CIFAR-10 dataset"
	@echo "  make train          Train model (default: configs/dev/ddpm_vp.yaml)"
	@echo "  make sample         Sample images from trained checkpoint"
	@echo "  make evaluate       Evaluate model FID and IS"
	@echo "  make ablation       Run automated ablation experiments"
	@echo "  make plots          Generate publication-ready ablation plots"
	@echo "  make test           Run all pytest unit tests"
	@echo "  make api            Run FastAPI inference service"
	@echo "  make app            Run Streamlit dashboard UI"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services with docker-compose"

install:
	pip install -r requirements.txt

download-data:
	python scripts/download_data.py

train:
	python scripts/train.py --config configs/dev/ddpm_vp.yaml

sample:
	python scripts/sample.py --config configs/dev/ddpm_vp.yaml --checkpoint checkpoints/ddpm_vp_latest.pt --num_samples 16

evaluate:
	python scripts/evaluate.py --config configs/dev/ddpm_vp.yaml --checkpoint checkpoints/ddpm_vp_latest.pt --num_samples 32

ablation:
	python scripts/run_ablation.py --config configs/experiments/dev_ablation.yaml

plots:
	python scripts/generate_plots.py

test:
	pytest tests/ -v

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d
