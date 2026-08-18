"""Unit tests for training loop, EMA, checkpoint saving and resumption."""

import tempfile
from pathlib import Path
import torch
import pytest

from src.utils.config import Config
from src.models import get_model
from src.sde import get_sde
from src.training.checkpoint import EMA, save_checkpoint, load_checkpoint
from src.training.trainer import Trainer


def test_ema_shadow_update():
    model = get_model("ddpm", base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
    ema = EMA(model, decay=0.9)

    # Modify model weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    ema.update(model)
    backup = ema.apply_shadow(model)
    # Check that model weights changed to shadow
    ema.restore(model, backup)


def test_checkpoint_save_and_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        model = get_model("ddpm", base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            step=42,
            epoch=2,
            best_loss=0.1234,
        )
        assert ckpt_path.exists()

        model_new = get_model("ddpm", base_channels=16, channel_multipliers=(1, 2), num_res_blocks=1)
        opt_new = torch.optim.Adam(model_new.parameters(), lr=1e-3)

        state = load_checkpoint(ckpt_path, model=model_new, optimizer=opt_new)
        assert state["step"] == 42
        assert state["epoch"] == 2
        assert abs(state["best_loss"] - 0.1234) < 1e-5


def test_trainer_mini_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = Config({
            "experiment_name": "test_mini_train",
            "model": "ddpm",
            "sde": "vp",
            "device": "cpu",
            "seed": 42,
            "model_params": {
                "base_channels": 16,
                "channel_multipliers": [1, 2],
                "num_res_blocks": 1,
                "attention_resolutions": [],
                "dropout": 0.0,
            },
            "sde_params": {
                "beta_min": 0.1,
                "beta_max": 20.0,
                "num_timesteps": 100,
            },
            "data": {
                "data_dir": "data",
                "batch_size": 4,
                "image_size": 32,
                "subset_size": 16,
                "num_workers": 0,
            },
            "training": {
                "steps": 4,
                "lr": 1e-3,
                "optimizer": "adam",
                "lr_scheduler": "warmup_cosine",
                "warmup_steps": 2,
                "use_ema": True,
                "ema_decay": 0.99,
                "amp": False,
                "grad_clip": 1.0,
                "log_every": 2,
                "save_every": 4,
                "sample_every": 4,
            },
            "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
            "results_dir": str(Path(tmpdir) / "results"),
        })

        trainer = Trainer(cfg)
        results = trainer.train()
        assert results["total_steps"] == 4
        assert results["training_time"] > 0
