"""Experiment registry for models, SDEs, and ablation parameters."""

from typing import Dict, Any

SUPPORTED_MODELS = ["ddpm", "ddpmpp", "ncsnpp"]
SUPPORTED_SDES = ["vp", "ve", "subvp"]
SUPPORTED_SAMPLERS = ["euler", "pc", "ode"]

DEFAULT_DEV_CONFIG: Dict[str, Any] = {
    "experiment_name": "dev_run",
    "model": "ddpm",
    "sde": "vp",
    "device": "auto",
    "seed": 42,
    "model_params": {
        "base_channels": 32,
        "channel_multipliers": [1, 2, 2],
        "num_res_blocks": 2,
        "attention_resolutions": [16],
        "dropout": 0.1,
    },
    "sde_params": {
        "beta_min": 0.1,
        "beta_max": 20.0,
        "num_timesteps": 1000,
    },
    "data": {
        "data_dir": "data",
        "batch_size": 32,
        "image_size": 32,
        "subset_size": 500,
        "num_workers": 0,
        "selected_classes": None,
    },
    "training": {
        "steps": 200,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "lr_scheduler": "warmup_cosine",
        "warmup_steps": 20,
        "use_ema": True,
        "ema_decay": 0.999,
        "amp": True,
        "grad_clip": 1.0,
        "log_every": 20,
        "save_every": 100,
        "sample_every": 100,
    },
    "mlflow": {
        "enabled": False,
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "dev_experiments",
    },
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",
}
