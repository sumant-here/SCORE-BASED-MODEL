"""Learning rate schedulers with warmup and cosine decay."""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """Cosine Annealing learning rate schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 500,
        total_steps: int = 100000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        if step < self.warmup_steps:
            # Linear warmup
            factor = step / float(max(1, self.warmup_steps))
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "warmup_cosine",
    warmup_steps: int = 500,
    total_steps: int = 100000,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """Factory to get learning rate scheduler."""
    stype = scheduler_type.lower()
    if stype in ["warmup_cosine", "cosine", "cosine_annealing"]:
        return WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    elif stype == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        return WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
