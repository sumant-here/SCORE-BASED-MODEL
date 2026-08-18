"""Data transformations and normalization for diffusion models."""

import torch
import torchvision.transforms as T
from typing import Tuple


def get_cifar10_transforms(
    image_size: int = 32,
    random_flip: bool = True,
) -> Tuple[T.Compose, T.Compose]:
    """Get train and evaluation transforms for CIFAR-10.
    Maps images from [0, 1] to [-1, 1].

    Args:
        image_size: Target image dimension (default 32 for CIFAR-10).
        random_flip: Whether to apply RandomHorizontalFlip for train set.

    Returns:
        (train_transforms, eval_transforms)
    """
    train_list = []
    if image_size != 32:
        train_list.append(T.Resize((image_size, image_size)))
    if random_flip:
        train_list.append(T.RandomHorizontalFlip())
    train_list.extend([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Maps [0, 1] to [-1, 1]
    ])

    eval_list = []
    if image_size != 32:
        eval_list.append(T.Resize((image_size, image_size)))
    eval_list.extend([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return T.Compose(train_list), T.Compose(eval_list)


def unnormalize_to_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] back to [0, 1] and clamp."""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def unnormalize_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] back to [0, 255] uint8."""
    tensor_01 = unnormalize_to_zero_one(tensor)
    return (tensor_01 * 255.0).round().to(torch.uint8)
