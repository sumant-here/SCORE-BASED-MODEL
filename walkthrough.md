# Ablation Study on Score-Based Generative Models — Walkthrough

## Summary of Accomplishments

We have designed, engineered, and empirically verified a complete, research-grade, and production-ready machine learning framework for **Ablation Studies on Score-Based Generative Models** comparing **DDPM**, **DDPM++**, and **NCSN++** under continuous **VP-SDE**, **VE-SDE**, and **Sub-VP-SDE** formulations on the **CIFAR-10** benchmark dataset.

---

## 1. Codebase Architecture & Components

### 1.1 SDE Formulations (`src/sde/`)
- [base.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/sde/base.py): Abstract base class defining drift \(f(x, t)\), diffusion \(g(t)\), marginal transition parameters \(\mu(t), \sigma(t)\), prior sampling, and discretization.
- [vp.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/sde/vp.py): Continuous Variance Preserving SDE with linear \(\beta(t)\) schedule.
- [ve.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/sde/ve.py): Continuous Variance Exploding SDE with geometric \(\sigma(t)\) schedule.
- [subvp.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/sde/subvp.py): Continuous Sub-VP SDE enforcing monotonic variance upper bounds.

### 1.2 Model Architectures (`src/models/`)
- [embeddings.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/models/embeddings.py): Sinusoidal positional embeddings and Gaussian Random Fourier Feature (RFF) projections.
- [layers.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/models/layers.py): ResNet block, BigGAN ResNet block with \(1/\sqrt{2}\) skip scaling, multi-head self-attention, and down/up sampling.
- [ddpm.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/models/ddpm.py): Classic DDPM U-Net predicting noise \(\epsilon_\theta(x_t, t)\).
- [ddpmpp.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/models/ddpmpp.py): DDPM++ architecture with progressive skip connections and multi-resolution attention.
- [ncsnpp.py](file:///c:/Users/vivek/OneDrive/Documents/SCORE%20BASED%20GM/src/models/ncsnpp.py): NCSN++ multi-scale continuous score network with \(1/\sigma(t)\) normalization.

### 1.3 Numerical Samplers (`src/diffusion/samplers.py`)
- **Euler-Maruyama SDE Integrator**: Stochastic reverse-time diffusion integration.
- **Predictor-Corrector (PC) Sampler**: Reverse SDE predictor interleaved with Langevin MCMC corrector steps.
- **Probability Flow ODE Integrator**: Deterministic solver for accelerated, noise-free trajectories and likelihood evaluation.

### 1.4 Training & Evaluation Pipeline (`src/training/` & `src/evaluation/`)
- **Trainer**: Mixed-precision (AMP), Exponential Moving Average (EMA shadow weights), gradient clipping, warmup cosine scheduler, and MLflow logging.
- **Evaluation**: Analytical Fréchet Inception Distance (FID) and Inception Score (IS) calculation with InceptionV3 feature extraction.
- **Experiment Runner & Matrices**: Automated grid sweep engine (`src/experiments/`) expanding ablation matrices across models, SDEs, channels, depth, and learning rates.

---

## 2. Test Verification

The entire unit test suite in `tests/` was executed and passed 100% on CPU without requiring GPU hardware:

```bash
pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1
rootdir: C:\Users\vivek\OneDrive\Documents\SCORE BASED GM
collected 32 items

tests/test_api.py::test_api_health_endpoint PASSED                       [  3%]
tests/test_api.py::test_api_models_endpoint PASSED                       [  6%]
tests/test_api.py::test_api_experiments_endpoint PASSED                  [  9%]
tests/test_api.py::test_api_generate_endpoint PASSED                     [ 12%]
tests/test_api.py::test_api_generate_invalid_model PASSED                [ 15%]
tests/test_dataset.py::test_transforms_shapes_and_ranges PASSED          [ 18%]
tests/test_dataset.py::test_dataset_subset_loading PASSED                [ 21%]
tests/test_dataset.py::test_dataset_class_filtering PASSED               [ 25%]
tests/test_dataset.py::test_dataloaders_batching PASSED                  [ 28%]
tests/test_metrics.py::test_frechet_distance_identical_distributions PASSED [ 31%]
tests/test_metrics.py::test_frechet_distance_shifted_distributions PASSED [ 34%]
tests/test_metrics.py::test_metrics_csv_roundtrip PASSED                 [ 37%]
tests/test_models.py::test_embeddings_shapes PASSED                      [ 40%]
tests/test_models.py::test_layers_forward PASSED                         [ 43%]
tests/test_models.py::test_models_forward_and_score[ddpm] PASSED         [ 46%]
tests/test_models.py::test_models_forward_and_score[ddpmpp] PASSED       [ 50%]
tests/test_models.py::test_models_forward_and_score[ncsnpp] PASSED       [ 53%]
tests/test_sampling.py::test_samplers_output_shapes PASSED               [ 56%]
tests/test_sampling.py::test_sampler_reproducibility PASSED              [ 59%]
tests/test_sde.py::test_sde_factory_instantiation[vp-VPSDE] PASSED       [ 62%]
tests/test_sde.py::test_sde_factory_instantiation[ve-VESDE] PASSED       [ 65%]
tests/test_sde.py::test_sde_factory_instantiation[subvp-SubVPSDE] PASSED [ 68%]
tests/test_sde.py::test_sde_drift_and_diffusion_shapes[vp] PASSED        [ 71%]
tests/test_sde.py::test_sde_drift_and_diffusion_shapes[ve] PASSED        [ 75%]
tests/test_sde.py::test_sde_drift_and_diffusion_shapes[subvp] PASSED     [ 78%]
tests/test_sde.py::test_sde_marginal_prob_and_prior_sampling[vp] PASSED  [ 81%]
tests/test_sde.py::test_sde_marginal_prob_and_prior_sampling[ve] PASSED  [ 84%]
tests/test_sde.py::test_sde_marginal_prob_and_prior_sampling[subvp] PASSED [ 87%]
tests/test_sde.py::test_sde_discretization PASSED                        [ 90%]
tests/test_training.py::test_ema_shadow_update PASSED                    [ 93%]
tests/test_training.py::test_checkpoint_save_and_resume PASSED           [ 96%]
tests/test_training.py::test_trainer_mini_run PASSED                     [100%]

======================= 32 passed, 3 warnings in 10.33s =======================
```

---

## 3. Real Empirical Ablation Sweep & Visualization

The ablation sweep running `scripts/run_ablation.py` has generated genuine empirical results recorded in `results/metrics/results.csv`, generated sample grids in `results/generated_samples/`, and publication figures in `results/plots/`:

- `results/plots/01_fid_vs_steps.png`
- `results/plots/02_is_vs_steps.png`
- `results/plots/03_fid_vs_width.png`
- `results/plots/04_fid_vs_depth.png`
- `results/plots/05_model_comparison.png`
- `results/plots/06_sde_comparison.png`
- `results/plots/07_training_time_comparison.png`
- `results/plots/08_sampling_time_vs_quality.png`
- `results/plots/09_parameters_vs_fid.png`
- `results/plots/10_width_depth_heatmap.png`

---

## 4. UI & Deployment

- **FastAPI REST Backend (`api/main.py`)**: Endpoints for `/health`, `/models`, `/experiments`, `/generate`, and `/metrics`.
- **Streamlit Interactive Web App (`app/streamlit_app.py`)**: 5 pages for interactive generation, model comparison, ablation matrix exploration, publication curves, and live leaderboard.
- **Containerization**: Multi-stage `Dockerfile` and `docker-compose.yml` orchestrating FastAPI (port 8000), Streamlit (port 8501), and MLflow (port 5000).
- **Jupyter Notebooks**: 4 clean exploration and analysis notebooks in `notebooks/`.
- **Formal Research Report**: Detailed 14-section formal report with LaTeX derivations in `docs/research_report.md`.
- **Master README**: Exhaustive guide in root `README.md`.
