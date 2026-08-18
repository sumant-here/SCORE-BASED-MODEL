"""Evaluation CLI script computing Fréchet Inception Distance (FID) and Inception Score (IS)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID and IS on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt.")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of generated samples for metrics.")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--sampler", type=str, default="euler", help="Sampler ('euler', 'pc', 'ode').")
    parser.add_argument("--sampler_steps", type=int, default=100, help="Number of integration steps.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(cfg, checkpoint_path=args.checkpoint)

    metrics = evaluator.evaluate(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        sampler_name=args.sampler,
        sampler_steps=args.sampler_steps,
    )

    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Model:           {metrics.model.upper()}")
    print(f"SDE:             {metrics.sde.upper()}")
    print(f"FID:             {metrics.fid:.3f} (lower is better)")
    print(f"Inception Score: {metrics.inception_score:.3f} +/- {metrics.inception_score_std:.3f}")
    print(f"Parameters:      {metrics.parameters:,}")
    print(f"Sampling Time:   {metrics.sampling_time:.2f}s ({metrics.num_samples} samples)")
    print("=" * 50)


if __name__ == "__main__":
    main()
