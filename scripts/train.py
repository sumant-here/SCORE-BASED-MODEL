"""Training script for Score-Based Generative Models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, Config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train DDPM / DDPM++ / NCSN++ under VP, VE, or Sub-VP SDE.")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML configuration.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from.")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda', 'cpu', 'auto').")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device:
        config.device = args.device
    if args.steps:
        if "training" not in config:
            config.training = {}
        config.training.steps = args.steps

    trainer = Trainer(config, resume_path=args.resume)
    results = trainer.train()
    print(f"\n[Training Complete] Results: {results}")


if __name__ == "__main__":
    main()
