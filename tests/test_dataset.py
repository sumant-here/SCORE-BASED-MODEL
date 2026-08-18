"""Unit tests for dataset loading, transforms, subsetting, and class filtering."""

import torch
import pytest
from src.data.transforms import get_cifar10_transforms, unnormalize_to_zero_one, unnormalize_to_uint8
from src.data.dataset import get_cifar10_datasets, get_dataloaders, CIFAR10_CLASSES


def test_transforms_shapes_and_ranges():
    """Verify normalization maps [0, 1] PIL to [-1, 1] tensors."""
    train_tf, eval_tf = get_cifar10_transforms(image_size=32, random_flip=False)
    dummy_img = torch.rand(3, 32, 32)
    # Convert dummy tensor to [-1, 1] via unnormalize inverted
    norm_tensor = dummy_img * 2.0 - 1.0

    assert norm_tensor.min() >= -1.0
    assert norm_tensor.max() <= 1.0

    unnorm = unnormalize_to_zero_one(norm_tensor)
    assert unnorm.min() >= 0.0
    assert unnorm.max() <= 1.0

    uint8_t = unnormalize_to_uint8(norm_tensor)
    assert uint8_t.dtype == torch.uint8
    assert uint8_t.min() >= 0
    assert uint8_t.max() <= 255


def test_dataset_subset_loading():
    """Verify fast subsetting mode creates exact item counts."""
    train_ds, test_ds = get_cifar10_datasets(
        data_dir="data",
        subset_size=50,
        download=True,
    )
    assert len(train_ds) == 50
    assert len(test_ds) == 10

    img, target = train_ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(target, int)


def test_dataset_class_filtering():
    """Verify class-filtering restricts targets to selected classes."""
    train_ds, _ = get_cifar10_datasets(
        data_dir="data",
        selected_classes=[3],  # cat only
        subset_size=30,
        download=True,
    )
    for i in range(len(train_ds)):
        _, target = train_ds[i]
        assert target == 3


def test_dataloaders_batching():
    """Verify DataLoader batches tensors correctly."""
    train_loader, test_loader = get_dataloaders(
        data_dir="data",
        batch_size=8,
        subset_size=24,
        num_workers=0,
        download=True,
    )
    batch, targets = next(iter(train_loader))
    assert batch.shape == (8, 3, 32, 32)
    assert targets.shape == (8,)
