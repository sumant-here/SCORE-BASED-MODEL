"""Robust checkpoint serialization and resumption utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from src.utils.seed import get_rng_states, set_rng_states


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.register(model)

    def register(self, model: nn.Module) -> None:
        """Initialize shadow weights with current model parameters."""
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with exponential decay."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(param.device)
                self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Apply shadow weights to model, returning original weights for restoration."""
        backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().detach() for k, v in state_dict.items()}


def save_checkpoint(
    save_path: Union[str, Path],
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    best_loss: float = float("inf"),
    config: Optional[dict] = None,
    mlflow_run_id: Optional[str] = None,
) -> None:
    """Save training state checkpoint."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_states": get_rng_states(),
        "config": config,
        "mlflow_run_id": mlflow_run_id,
    }
    torch.save(state, path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load training state checkpoint for resumption or evaluation."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    try:
        checkpoint = torch.load(path, map_location=device if device is not None else "cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device if device is not None else "cpu")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "rng_states" in checkpoint and checkpoint["rng_states"] is not None:
        try:
            set_rng_states(checkpoint["rng_states"])
        except Exception:
            pass  # Non-fatal if CUDA count differs

    return checkpoint
