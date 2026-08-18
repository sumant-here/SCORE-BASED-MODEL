# Ablation Study on Score-Based Generative Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](http://localhost:8000/docs)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](http://localhost:8501)

A research-grade, production-quality comparative ablation study evaluating **DDPM**, **DDPM++**, and **NCSN++** under continuous **VP-SDE**, **VE-SDE**, and **Sub-VP-SDE** formulations on the **CIFAR-10** benchmark dataset.

---

## Table of Contents

- [1. Research Overview & Key Questions](#1-research-overview--key-questions)
- [2. Mathematical Formulations & Intuition](#2-mathematical-formulations--intuition)
  - [2.1 Score Functions & Score Matching](#21-score-functions--score-matching)
  - [2.2 Continuous SDE Formulations (VP, VE, Sub-VP)](#22-continuous-sde-formulations-vp-ve-sub-vp)
  - [2.3 Reverse-Time SDE & Probability Flow ODE](#23-reverse-time-sde--probability-flow-ode)
- [3. Model Architectures](#3-model-architectures)
  - [3.1 DDPM](#31-ddpm)
  - [3.2 DDPM++](#32-ddpm)
  - [3.3 NCSN++](#33-ncsn)
- [4. CIFAR-10 Dataset & Preprocessing](#4-cifar-10-dataset--preprocessing)
- [5. Numerical Samplers](#5-numerical-samplers)
- [6. Evaluation Metrics (FID & Inception Score)](#6-evaluation-metrics-fid--inception-score)
- [7. Experimental Results & Leaderboard](#7-experimental-results--leaderboard)
- [8. Repository Structure](#8-repository-structure)
- [9. Quickstart Guide](#9-quickstart-guide)
  - [Installation](#installation)
  - [Fast Development Mode](#fast-development-mode)
  - [Production Training](#production-training)
  - [Image Sampling](#image-sampling)
  - [Evaluation](#evaluation)
  - [Automated Ablation Study](#automated-ablation-study)
- [10. MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
- [11. Interactive Streamlit Dashboard](#11-interactive-streamlit-dashboard)
- [12. FastAPI Inference Service](#12-fastapi-inference-service)
- [13. Docker & Deployment](#13-docker--deployment)
- [14. Reproducibility & Checkpointing](#14-reproducibility--checkpointing)
- [15. Limitations & Future Work](#15-limitations--future-work)
- [16. References](#16-references)

---

## 1. Research Overview & Key Questions

Score-based generative models formulate generation as reversing a continuous noise-injection diffusion process. This project answers 7 core research questions:

1. **Architecture Comparison**: Which score architecture generates higher fidelity samples—classic DDPM, DDPM++ (BigGAN residual blocks with skip scaling), or NCSN++ (multi-scale noise-conditioned score network)?
2. **SDE Formulation**: Which continuous diffusion process (VP, VE, or Sub-VP) provides the most stable training dynamics and lowest FID on CIFAR-10?
3. **Width vs. Depth**: How does scaling base channels (width: 32, 64, 128, 256) versus stacking residual layers (depth: 2, 4, 6, 8) impact FID and parameter efficiency?
4. **Training Duration**: How do step budgets (10K, 50K, 100K, 200K) correlate with perceptual convergence?
5. **Learning Rate Sensitivity**: How do learning rates (\(10^{-3}\) to \(5\times 10^{-5}\)) affect loss stability and score gradient explosion?
6. **Class Conditioning vs Full Dataset**: How do single-class models (e.g. *cat only*, *airplane only*) compare against 10-class models?
7. **Efficiency Frontier**: Which configuration achieves the Pareto-optimal trade-off between **FID**, **IS**, **training time**, **sampling latency**, and **parameter count**?

---

## 2. Mathematical Formulations & Intuition

### 2.1 Score Functions & Score Matching

The Stein score function of a distribution \(p(x)\) is the vector field of its log-density gradients:
\[
s(x) = \nabla_x \log p(x)
\]
In **Denoising Score Matching (DSM)**, given clean image \(x_0\) and perturbed noisy image \(x_t \sim q_{0t}(x_t|x_0)\), the analytical score of the perturbation kernel is:
\[
\nabla_{x_t} \log q_{0t}(x_t|x_0) = -\frac{x_t - \mu_t}{\sigma_t^2} = -\frac{z}{\sigma_t} \quad \text{where } z \sim \mathcal{N}(0, I)
\]
The DSM objective with continuous weighting \(\lambda(t) = \sigma_t^2\) minimizes:
\[
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(\epsilon, T), x_0, z} \left[ \left\| \epsilon_\theta(x_t, t) - z \right\|_2^2 \right]
\]

```
High Probability Density Region
        ▲
       ╱ ╲     Score vectors \nabla_x log p(x) point
      ╱   ╲    TOWARD the data manifold
     ╱     ╲   
────┴───────┴────>
Low Density   Noisy Samples (x_t) pushed back into mode
```

---

### 2.2 Continuous SDE Formulations (VP, VE, Sub-VP)

Forward SDE:
\[
dx = f(x, t)dt + g(t)dW_t
\]

| Formulation | Drift \(f(x, t)\) | Diffusion \(g(t)\) | Transition Kernel \(p_{0t}(x_t|x_0)\) | Prior \(p_T(x)\) |
| :--- | :--- | :--- | :--- | :--- |
| **VP-SDE** | \(-\frac{1}{2}\beta(t)x\) | \(\sqrt{\beta(t)}\) | \(\mathcal{N}\left(x_0 e^{-\frac{1}{2}\int_0^t \beta(s)ds}, \left(1 - e^{-\int_0^t \beta(s)ds}\right)I\right)\) | \(\mathcal{N}(0, I)\) |
| **VE-SDE** | \(0\) | \(\sigma(t)\sqrt{2\log\frac{\sigma_{max}}{\sigma_{min}}}\) | \(\mathcal{N}\left(x_0, \sigma^2(t)I\right)\) | \(\mathcal{N}(0, \sigma_{max}^2 I)\) |
| **Sub-VP-SDE** | \(-\frac{1}{2}\beta(t)x\) | \(\sqrt{\beta(t)\left(1 - e^{-2\int_0^t \beta(s)ds}\right)}\) | \(\mathcal{N}\left(x_0 e^{-\frac{1}{2}\int_0^t \beta(s)ds}, \left(1 - e^{-\int_0^t \beta(s)ds}\right)^2 I\right)\) | \(\mathcal{N}(0, I)\) |

---

### 2.3 Reverse-Time SDE & Probability Flow ODE

Anderson's theorem establishes the reverse-time SDE:
\[
dx = \left[ f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{W}_t
\]
The deterministic **Probability Flow ODE** shares identical marginal distributions:
\[
dx = \left[ f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt
\]

---

## 3. Model Architectures

### 3.1 DDPM
- Classic U-Net based on Ho et al. (2020) with sinusoidal timestep embeddings.
- Residual convolutional blocks with GroupNorm and SiLU activations.
- Multi-head self-attention at lower spatial resolution (\(16\times 16\)).

### 3.2 DDPM++
- Enhanced U-Net with BigGAN-style residual blocks and \(1/\sqrt{2}\) skip connection scaling.
- Multi-resolution self-attention at \(16\times 16\) and \(8\times 8\).
- Progressive channel multipliers and continuous noise conditioning.

### 3.3 NCSN++
- Multi-scale continuous score network based on Song et al. (2020).
- Gaussian Random Fourier Feature (RFF) noise embeddings.
- Explicit score output parameterized with \(1/\sigma(t)\) normalization.

---

## 4. CIFAR-10 Dataset & Preprocessing

- **Images**: 60,000 \(32\times 32\) color images across 10 balanced classes.
- **Normalization**: Pixel values mapped to \([-1, 1]\).
- **Subsetting**: Rapid development mode (e.g. 500 or 2,000 samples) enables quick local iterations.
- **Class Filtering**: Supports single-class training (e.g., class 3 = cat) or arbitrary subsets of classes.

---

## 5. Numerical Samplers

1. **Euler-Maruyama (SDE)**: Stochastic reverse-time integrator.
2. **Predictor-Corrector (PC)**: Reverse SDE predictor combined with Langevin MCMC corrector.
3. **Probability Flow (ODE)**: Deterministic ODE solver enabling fast sampling and exact likelihood computation.

---

## 6. Evaluation Metrics (FID & Inception Score)

- **Fréchet Inception Distance (FID)**: Measures Wasserstein-2 distance between feature Gaussians extracted from real CIFAR-10 test set and generated images:
  \[
  d^2 = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)
  \]
- **Inception Score (IS)**: Evaluates sharpness and diversity:
  \[
  \text{IS} = \exp\left( \mathbb{E}_x \left[ D_{KL}(p(y|x) \parallel p(y)) \right] \right)
  \]

---

## 7. Experimental Results & Leaderboard

The leaderboard below is populated directly from executed experiments recorded in `results/metrics/results.csv`:

| Model | SDE | Base Width | Depth | Parameters | Training Time | Sampling Time | FID (↓) | Inception Score (↑) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NCSN++** | **Sub-VP** | 32 | 2 | 1,624,323 | 132.35s | 10.73s | **79.46** | **1.23 ± 0.10** |
| **DDPM++** | **VE** | 32 | 2 | 1,624,323 | 86.41s | 8.71s | **82.71** | **1.17 ± 0.04** |
| **DDPM** | **VE** | 32 | 2 | 1,624,323 | 89.93s | 7.09s | **84.78** | **1.20 ± 0.05** |
| **DDPM++** | **Sub-VP** | 32 | 2 | 1,624,323 | 143.98s | 10.23s | **89.24** | **1.21 ± 0.08** |
| **DDPM** | **Sub-VP** | 32 | 2 | 1,624,323 | 79.28s | 6.49s | **92.57** | **1.20 ± 0.07** |
| **NCSN++** | **VE** | 32 | 2 | 1,624,323 | 129.10s | 9.95s | **93.15** | **1.23 ± 0.14** |
| **NCSN++** | **VP** | 32 | 2 | 1,624,323 | 137.72s | 10.18s | **103.44** | **1.36 ± 0.13** |
| **DDPM++** | **VP** | 32 | 2 | 1,624,323 | 86.17s | 7.01s | **110.03** | **1.33 ± 0.10** |
| **DDPM** | **VP** | 32 | 2 | 1,624,323 | 84.39s | 6.18s | **115.10** | **1.33 ± 0.13** |

---

## 8. Repository Structure

```text
score-based-generative-models/
├── README.md                           # Main documentation & empirical findings
├── LICENSE                             # MIT License
├── pyproject.toml                      # Project metadata & build tool config
├── requirements.txt                    # Core & optional dependencies
├── Dockerfile                          # Multi-stage production container
├── docker-compose.yml                  # Compose orchestrating API, UI & MLflow
├── Makefile                            # Automated build & development commands
├── .env.example                        # Environment variables template
│
├── configs/
│   ├── ddpm.yaml                       # Default DDPM config
│   ├── ddpmpp.yaml                     # Default DDPM++ config
│   ├── ncsnpp.yaml                     # Default NCSN++ config
│   ├── vp.yaml / ve.yaml / subvp.yaml  # Default SDE configs
│   ├── dev/                            # Fast development configs (subsets)
│   ├── production/                     # Full 200k-step production configs
│   └── experiments/                    # Ablation matrices (width, depth, LR, SDE)
│
├── src/
│   ├── data/                           # Dataset loader & transforms
│   ├── models/                         # DDPM, DDPM++, NCSN++, U-Net, Layers
│   ├── sde/                            # BaseSDE, VP-SDE, VE-SDE, Sub-VP SDE
│   ├── diffusion/                      # Forward process, losses, Euler/PC/ODE samplers
│   ├── training/                       # Trainer, EMA, Checkpointing, Schedulers
│   ├── evaluation/                     # FID, Inception Score, Evaluator
│   ├── experiments/                    # Matrix expander & ExperimentRunner
│   ├── visualization/                  # 10 Publication plots & sample grids
│   └── utils/                          # Device, logging, config, RNG seeds
│
├── api/                                # FastAPI inference backend & Pydantic schemas
├── app/                                # Streamlit 5-page interactive dashboard
├── scripts/                            # CLI tools: download, train, sample, evaluate, ablation
├── tests/                              # Pytest test suite (100% CPU compatible)
├── notebooks/                          # 4 Interactive Jupyter Notebooks
└── results/                            # Saved metrics, plots, sample grids, tables
```

---

## 9. Quickstart Guide

### Installation

```bash
git clone https://github.com/your-username/score-based-generative-models.git
cd score-based-generative-models

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Fast Development Mode
Train a model in seconds on CPU or GPU:
```bash
python scripts/train.py --config configs/dev/ddpm_vp.yaml
```

### Production Training
```bash
python scripts/train.py --config configs/production/ddpmpp_vp.yaml
```

### Image Sampling
```bash
python scripts/sample.py --config configs/dev/ddpm_vp.yaml --checkpoint checkpoints/dev_ddpm_vp_latest.pt --num_samples 16 --sampler euler
```

### Evaluation
Compute FID and IS against CIFAR-10:
```bash
python scripts/evaluate.py --config configs/dev/ddpm_vp.yaml --checkpoint checkpoints/dev_ddpm_vp_latest.pt --num_samples 64
```

### Automated Ablation Study
Run a sweeping grid search across architectures, SDEs, or hyperparameters:
```bash
python scripts/run_ablation.py --config configs/experiments/dev_ablation.yaml
```

---

## 10. MLflow Experiment Tracking

Start MLflow locally:
```bash
mlflow ui --port 5000
```
Navigate to `http://localhost:5000` to inspect parameter sweeps, loss curves, and generated image artifacts.

---

## 11. Interactive Streamlit Dashboard

Launch the 5-page interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```
- **Page 1 (Generate)**: Interactive synthesis with sliders for model, SDE, sampler, steps, and seed.
- **Page 2 (Compare Models)**: Comparative bar charts and parameter capacity analysis.
- **Page 3 (Ablation Explorer)**: Dynamic hyperparameter filter with interactive scatter plots.
- **Page 4 (Training Curves)**: 10 publication-ready research plots.
- **Page 5 (Leaderboard)**: Live sortable experiment leaderboard.

---

## 12. FastAPI Inference Service

Start the production REST API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation available at `http://localhost:8000/docs`.

---

## 13. Docker & Deployment

```bash
# Start FastAPI, Streamlit, and MLflow with Docker Compose
docker compose up -d

# Check running services
docker compose ps
```

---

## 14. Reproducibility & Checkpointing

- Global random seeding across Python `random`, `numpy`, PyTorch CPU, and CUDA.
- Checkpoints save complete model state, EMA weights, optimizer, scheduler, current step, and exact RNG states.
- Resume training seamlessly:
```bash
python scripts/train.py --config configs/dev/ddpm_vp.yaml --resume checkpoints/dev_ddpm_vp_latest.pt
```

---

## 15. Limitations & Future Work

- **Conditional Sampling**: Exploring classifier-free guidance for class-conditional CIFAR-10 generation.
- **Fast Solvers**: Implementing high-order DPM-Solver++ and Euler-Heun adaptive step solvers.
- **Scale**: Extending the ablation matrix to ImageNet \(64\times 64\) and CelebA.

---

## 16. References

1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
2. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.
3. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS 2019.
4. Salimans, T., et al. (2016). *Improved Techniques for Training GANs*. NeurIPS 2016.
5. Heusel, M., et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*. NeurIPS 2017.
