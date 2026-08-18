"""Generic Training Pipeline for Score-Based Generative Models."""

import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.config import Config
from src.utils.seed import set_seed
from src.utils.device import get_device, get_gpu_memory_info
from src.utils.logging import setup_logger
from src.data.dataset import get_dataloaders
from src.data.transforms import unnormalize_to_zero_one
from src.models import get_model, count_parameters
from src.sde import get_sde
from src.diffusion.losses import get_loss_fn
from src.diffusion.samplers import euler_maruyama_sampler
from src.training.checkpoint import EMA, save_checkpoint, load_checkpoint
from src.training.scheduler import get_lr_scheduler

# MLflow safe import
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Trainer:
    """End-to-End Trainer supporting DDPM, DDPM++, NCSN++ with VP, VE, Sub-VP SDEs."""

    def __init__(self, config: Union[Config, dict], resume_path: Optional[str] = None):
        if isinstance(config, dict) and not isinstance(config, Config):
            self.config = Config(config)
        else:
            self.config = config

        self.logger = setup_logger("Trainer")
        self.device = get_device(self.config.get("device", "auto"))
        self.logger.info(f"Initialized training on device: {self.device}")

        # Set reproducibility seed
        set_seed(self.config.get("seed", 42))

        # Output paths
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.results_dir = Path(self.config.get("results_dir", "results"))
        self.samples_dir = self.results_dir / "generated_samples" / self.config.get("experiment_name", "experiment")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Build SDE
        sde_name = self.config.get("sde", "vp")
        sde_params = self.config.get("sde_params", {})
        self.sde = get_sde(sde_name, **sde_params)
        self.logger.info(f"Loaded SDE: {self.sde.sde_type}")

        # Build Model
        model_name = self.config.get("model", "ddpm")
        model_params = self.config.get("model_params", {}).to_dict() if isinstance(self.config.get("model_params"), Config) else self.config.get("model_params", {})
        self.model = get_model(model_name, **model_params).to(self.device)
        self.param_count = count_parameters(self.model)
        self.logger.info(f"Loaded model '{model_name}' with {self.param_count:,} trainable parameters")

        # EMA Tracking
        use_ema = self.config.get("training", {}).get("use_ema", True)
        ema_decay = self.config.get("training", {}).get("ema_decay", 0.9999)
        self.ema = EMA(self.model, decay=ema_decay) if use_ema else None

        # Optimizer
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("lr", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 0.0))
        opt_name = train_cfg.get("optimizer", "adam").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

        # Training Steps and Schedulers
        self.total_steps = int(train_cfg.get("steps", 100000))
        warmup_steps = int(train_cfg.get("warmup_steps", min(500, self.total_steps // 10)))
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            scheduler_type=train_cfg.get("lr_scheduler", "warmup_cosine"),
            warmup_steps=warmup_steps,
            total_steps=self.total_steps,
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )

        # Mixed precision
        use_amp = train_cfg.get("amp", True) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        # Loss function
        self.loss_fn = get_loss_fn(self.sde)

        # Dataloaders
        data_cfg = self.config.get("data", {})
        self.batch_size = int(data_cfg.get("batch_size", 64))
        self.train_loader, self.test_loader = get_dataloaders(
            data_dir=data_cfg.get("data_dir", "data"),
            batch_size=self.batch_size,
            image_size=int(data_cfg.get("image_size", 32)),
            selected_classes=data_cfg.get("selected_classes", None),
            subset_size=data_cfg.get("subset_size", None),
            num_workers=int(data_cfg.get("num_workers", 0)),
            seed=int(self.config.get("seed", 42)),
            download=True,
        )

        # Resume if specified
        self.start_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.mlflow_run_id = None

        if resume_path:
            self.resume(resume_path)

        # MLflow setup
        self.use_mlflow = self.config.get("mlflow", {}).get("enabled", False) and MLFLOW_AVAILABLE
        if self.use_mlflow:
            try:
                uri = self.config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
                exp_name = self.config.get("mlflow", {}).get("experiment_name", "score_based_models")
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(exp_name)
                if not mlflow.active_run():
                    active_run = mlflow.start_run(run_name=self.config.get("experiment_name", "run"))
                    self.mlflow_run_id = active_run.info.run_id
                    mlflow.log_params(self._flatten_dict(self.config.to_dict()))
            except Exception as e:
                self.logger.warning(f"Failed to connect to MLflow server: {e}. Continuing without MLflow.")
                self.use_mlflow = False

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v) if isinstance(v, (list, tuple)) else v))
        return dict(items)

    def resume(self, resume_path: str) -> None:
        """Resume training state from checkpoint."""
        self.logger.info(f"Resuming checkpoint from: {resume_path}")
        checkpoint = load_checkpoint(
            resume_path,
            model=self.model,
            ema=self.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.start_step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.mlflow_run_id = checkpoint.get("mlflow_run_id", None)
        self.logger.info(f"Resumed at step {self.start_step}, epoch {self.epoch}, best_loss {self.best_loss:.4f}")

    def train(self) -> Dict[str, Any]:
        """Execute full training loop."""
        self.model.train()
        train_cfg = self.config.get("training", {})
        log_every = int(train_cfg.get("log_every", 100))
        save_every = int(train_cfg.get("save_every", 1000))
        sample_every = int(train_cfg.get("sample_every", 1000))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))

        step = self.start_step
        running_loss = 0.0
        start_time = time.time()
        self.logger.info(f"Starting training loop: steps {step} -> {self.total_steps}")

        data_iter = iter(self.train_loader)
        pbar = tqdm(total=self.total_steps, initial=step, desc="Training")

        while step < self.total_steps:
            try:
                batch, _ = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_loader)
                batch, _ = next(data_iter)

            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.loss_fn(self.model, batch)

            self.scaler.scale(loss).backward()

            if grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            step += 1
            loss_val = loss.item()
            running_loss += loss_val
            pbar.update(1)

            # Logging
            if step % log_every == 0 or step == self.total_steps:
                avg_loss = running_loss / log_every
                running_loss = 0.0
                curr_lr = self.scheduler.get_last_lr()[0]
                mem = get_gpu_memory_info()
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{curr_lr:.2e}"})

                if self.use_mlflow:
                    try:
                        mlflow.log_metric("loss", avg_loss, step=step)
                        mlflow.log_metric("lr", curr_lr, step=step)
                        if mem["device"] != "cpu":
                            mlflow.log_metric("gpu_mem_mb", mem["allocated_mb"], step=step)
                    except Exception:
                        pass

            # Sample Generation for visualization
            if step % sample_every == 0 or step == self.total_steps:
                self.generate_and_save_samples(step, num_samples=16)

            # Checkpoint saving
            if step % save_every == 0 or step == self.total_steps:
                exp_name = self.config.get("experiment_name", "model")
                latest_path = self.checkpoint_dir / f"{exp_name}_latest.pt"
                save_checkpoint(
                    latest_path,
                    model=self.model,
                    ema=self.ema,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step,
                    epoch=self.epoch,
                    best_loss=self.best_loss,
                    config=self.config.to_dict(),
                    mlflow_run_id=self.mlflow_run_id,
                )

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    best_path = self.checkpoint_dir / f"{exp_name}_best.pt"
                    save_checkpoint(
                        best_path,
                        model=self.model,
                        ema=self.ema,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=step,
                        epoch=self.epoch,
                        best_loss=self.best_loss,
                        config=self.config.to_dict(),
                        mlflow_run_id=self.mlflow_run_id,
                    )

        pbar.close()
        total_training_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.2f}m)")

        if self.use_mlflow:
            try:
                mlflow.log_metric("total_training_time_sec", total_training_time)
                mlflow.end_run()
            except Exception:
                pass

        return {
            "total_steps": step,
            "best_loss": self.best_loss,
            "training_time": total_training_time,
            "parameters": self.param_count,
        }

    def generate_and_save_samples(self, step: int, num_samples: int = 16) -> Path:
        """Sample a grid of images and save to disk."""
        self.model.eval()
        backup = self.ema.apply_shadow(self.model) if self.ema is not None else None

        sample_shape = (num_samples, 3, 32, 32)
        samples = euler_maruyama_sampler(
            model=self.model,
            sde=self.sde,
            shape=sample_shape,
            device=self.device,
            num_steps=100,  # Fast sampling during training progress checks
            show_progress=False,
        )

        samples_01 = unnormalize_to_zero_one(samples)
        grid_path = self.samples_dir / f"sample_step_{step:06d}.png"
        save_image(samples_01, grid_path, nrow=4, normalize=False)

        if self.ema is not None and backup is not None:
            self.ema.restore(self.model, backup)

        self.model.train()
        return grid_path
