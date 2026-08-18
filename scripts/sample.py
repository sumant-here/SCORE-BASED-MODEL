"""Sampling and Image Synthesis script for Score-Based Generative Models."""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.models import get_model
from src.sde import get_sde
from src.training.checkpoint import load_checkpoint
from src.visualization.samples import generate_sample_grid


def main():
    parser = argparse.ArgumentParser(description="Generate image samples from score-based diffusion model.")
    parser.add_argument("--config", type=str, default="configs/dev/ddpm_vp.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt file.")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate.")
    parser.add_argument("--steps", type=int, default=100, help="Discretization sampling steps.")
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "pc", "ode"], help="Sampler algorithm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=str, default="results/generated_samples/sample_grid.png", help="Output PNG path.")
    args = parser.parse_args()

    set_seed(args.seed)
    
    ckpt_data = None
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt_data, dict) and "config" in ckpt_data and ckpt_data["config"]:
            cfg = ckpt_data["config"]
        elif args.config:
            cfg = load_config(args.config)
        else:
            cfg = {}
    else:
        cfg = load_config(args.config)

    device = get_device(cfg.get("device", "auto"))
    sde = get_sde(cfg.get("sde", "vp"), **cfg.get("sde_params", {}))
    model = get_model(cfg.get("model", "ddpm"), **cfg.get("model_params", {})).to(device)

    if ckpt_data is not None:
        print(f"Loading weights from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, model=model, device=device)

    print(f"Generating {args.num_samples} samples using {args.sampler} sampler ({args.steps} steps) on {device}...")
    out_tensor = generate_sample_grid(
        model=model,
        sde=sde,
        device=device,
        num_samples=args.num_samples,
        num_steps=args.steps,
        sampler_name=args.sampler,
        save_path=args.out,
        nrow=int(args.num_samples ** 0.5) if args.num_samples > 4 else args.num_samples,
    )
    print(f"Successfully generated and saved sample grid to: {args.out}")


if __name__ == "__main__":
    main()
