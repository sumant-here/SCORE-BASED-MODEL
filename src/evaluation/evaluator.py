"""End-to-end evaluation orchestrator for FID, IS, and latency benchmarks."""

import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from torchvision.utils import save_image

from src.utils.config import Config
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.data.dataset import get_cifar10_datasets
from src.data.transforms import unnormalize_to_zero_one
from src.models import get_model, count_parameters
from src.sde import get_sde
from src.diffusion.samplers import get_sampler, euler_maruyama_sampler
from src.training.checkpoint import EMA, load_checkpoint
from src.evaluation.fid import calculate_fid
from src.evaluation.inception_score import calculate_inception_score, InceptionFeatureExtractor
from src.evaluation.metrics import ExperimentMetrics, save_metrics_to_csv


class Evaluator:
    """Evaluator to benchmark generative performance (FID, IS, throughput) across models and SDEs."""

    def __init__(
        self,
        config: Union[Config, dict],
        checkpoint_path: Optional[str] = None,
        feature_extractor: Optional[InceptionFeatureExtractor] = None,
    ):
        if isinstance(config, dict) and not isinstance(config, Config):
            self.config = Config(config)
        else:
            self.config = config

        self.logger = setup_logger("Evaluator")
        self.device = get_device(self.config.get("device", "auto"))

        # Build SDE
        sde_name = self.config.get("sde", "vp")
        sde_params = self.config.get("sde_params", {})
        self.sde = get_sde(sde_name, **sde_params)

        # Build Model
        model_name = self.config.get("model", "ddpm")
        model_params = self.config.get("model_params", {}).to_dict() if isinstance(self.config.get("model_params"), Config) else self.config.get("model_params", {})
        self.model = get_model(model_name, **model_params).to(self.device)
        self.param_count = count_parameters(self.model)

        # Load Checkpoint if provided
        self.training_time = 0.0
        self.steps = int(self.config.get("training", {}).get("steps", 0))
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # Feature extractor for FID & IS
        self.feature_extractor = feature_extractor or InceptionFeatureExtractor().to(self.device)
        self.feature_extractor.eval()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights and metadata from checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model=self.model, device=self.device)
        self.steps = checkpoint.get("step", self.steps)

    def evaluate(
        self,
        num_samples: int = 128,
        batch_size: int = 32,
        sampler_name: str = "euler",
        sampler_steps: int = 100,
        save_samples: bool = True,
        output_dir: str = "results",
    ) -> ExperimentMetrics:
        """Run full evaluation suite: generate samples, compute FID & IS, measure latency.

        Args:
            num_samples: Total generated images for evaluation.
            batch_size: Generation batch size.
            sampler_name: 'euler', 'pc', or 'ode'.
            sampler_steps: SDE integration steps.
            save_samples: Save sample grid to disk.
            output_dir: Root results directory.

        Returns:
            ExperimentMetrics dataclass instance.
        """
        self.model.eval()
        sampler_fn = get_sampler(sampler_name)
        self.logger.info(f"Starting generation of {num_samples} samples using {sampler_name} sampler ({sampler_steps} steps)")

        generated_batches = []
        total_batches = (num_samples + batch_size - 1) // batch_size
        start_time = time.time()

        with torch.no_grad():
            for i in range(total_batches):
                current_b = min(batch_size, num_samples - i * batch_size)
                shape = (current_b, 3, 32, 32)
                batch_samples = sampler_fn(
                    model=self.model,
                    sde=self.sde,
                    shape=shape,
                    device=self.device,
                    num_steps=sampler_steps,
                    show_progress=False,
                )
                generated_batches.append(batch_samples.cpu())

        total_sampling_time = time.time() - start_time
        gen_tensors = torch.cat(generated_batches, dim=0)[:num_samples]
        gen_tensors_01 = unnormalize_to_zero_one(gen_tensors)
        avg_sampling_time_per_img = total_sampling_time / num_samples

        self.logger.info(f"Generated {num_samples} samples in {total_sampling_time:.2f}s ({avg_sampling_time_per_img*1000:.1f}ms/image)")

        # Save sample grid
        if save_samples:
            out_path = Path(output_dir) / "generated_samples" / self.config.get("experiment_name", "eval")
            out_path.mkdir(parents=True, exist_ok=True)
            grid_file = out_path / "evaluation_grid.png"
            grid_samples = gen_tensors_01[:min(36, num_samples)]
            save_image(grid_samples, grid_file, nrow=6, normalize=False)
            self.logger.info(f"Saved evaluation sample grid to: {grid_file}")

        # Load Real CIFAR-10 evaluation data
        data_cfg = self.config.get("data", {})
        _, test_ds = get_cifar10_datasets(
            data_dir=data_cfg.get("data_dir", "data"),
            image_size=32,
            selected_classes=data_cfg.get("selected_classes", None),
            subset_size=max(num_samples, 200),
            download=True,
        )

        real_imgs = []
        for j in range(min(num_samples, len(test_ds))):
            img_tensor, _ = test_ds[j]
            real_imgs.append(img_tensor)
        real_tensors = torch.stack(real_imgs, dim=0)

        # Compute Inception Score
        self.logger.info("Computing Inception Score...")
        is_mean, is_std = calculate_inception_score(
            gen_tensors_01,
            batch_size=batch_size,
            device=self.device,
            feature_extractor=self.feature_extractor,
        )
        self.logger.info(f"Inception Score: {is_mean:.3f} +/- {is_std:.3f}")

        # Compute FID
        self.logger.info("Computing Fréchet Inception Distance (FID)...")
        fid_score = calculate_fid(
            real_images=real_tensors,
            generated_images=gen_tensors_01,
            batch_size=batch_size,
            device=self.device,
            feature_extractor=self.feature_extractor,
        )
        self.logger.info(f"FID Score: {fid_score:.3f}")

        # Assemble Metrics
        exp_id = self.config.get("experiment_name", f"{self.config.get('model', 'ddpm')}_{self.config.get('sde', 'vp')}")
        width = int(self.config.get("model_params", {}).get("base_channels", self.config.get("model_params", {}).get("width", 64)))
        depth = int(self.config.get("model_params", {}).get("num_res_blocks", self.config.get("model_params", {}).get("depth", 2)))
        lr = float(self.config.get("training", {}).get("lr", 1e-4))
        seed = int(self.config.get("seed", 42))
        class_filter = str(self.config.get("data", {}).get("selected_classes", "all"))

        metrics = ExperimentMetrics(
            experiment_id=exp_id,
            model=str(self.config.get("model", "ddpm")),
            sde=str(self.config.get("sde", "vp")),
            width=width,
            depth=depth,
            learning_rate=lr,
            steps=self.steps,
            parameters=self.param_count,
            training_time=self.training_time,
            sampling_time=round(total_sampling_time, 2),
            fid=round(fid_score, 3),
            inception_score=round(is_mean, 3),
            inception_score_std=round(is_std, 3),
            num_samples=num_samples,
            seed=seed,
            dataset="cifar10",
            class_filter=class_filter,
        )

        csv_file = Path(output_dir) / "metrics" / "results.csv"
        save_metrics_to_csv(metrics, csv_file)
        self.logger.info(f"Appended evaluation metrics to: {csv_file}")

        return metrics
