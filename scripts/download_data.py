"""Script to download and verify CIFAR-10 dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_cifar10_datasets
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 dataset for score-based generative modeling.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store CIFAR-10 dataset.")
    parser.add_argument("--subset", type=int, default=None, help="Optional subset size.")
    args = parser.parse_args()

    logger = setup_logger("DownloadData")
    logger.info(f"Downloading/verifying CIFAR-10 in directory: '{args.data_dir}'...")

    train_ds, test_ds = get_cifar10_datasets(
        data_dir=args.data_dir,
        subset_size=args.subset,
        download=True,
    )

    logger.info(f"CIFAR-10 successfully ready! Train items: {len(train_ds)}, Test items: {len(test_ds)}")


if __name__ == "__main__":
    main()
