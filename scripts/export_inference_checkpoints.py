"""Export compact inference-only checkpoints (stripping optimizer state and converting to fp16/fp32)
to reduce file size from ~26MB down to ~3MB-6MB for easy direct GitHub commits.
"""

import argparse
from pathlib import Path
import torch


def export_checkpoint(checkpoint_path: Path, output_dir: Path, fp16: bool = False):
    """Strip optimizer states and shadow duplicates to create a tiny checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract model state dict (prefer EMA weights)
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        state_dict = ckpt["ema_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if fp16:
        state_dict = {
            k: v.half() if (isinstance(v, torch.Tensor) and torch.is_floating_point(v)) else v
            for k, v in state_dict.items()
        }

    compact_ckpt = {
        "model_state_dict": state_dict,
        "config": ckpt.get("config", {}),
        "step": ckpt.get("step", 0),
        "fid": ckpt.get("fid", None),
        "is_fp16": fp16,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / checkpoint_path.name
    torch.save(compact_ckpt, out_file)
    
    orig_size = checkpoint_path.stat().st_size / (1024 * 1024)
    new_size = out_file.stat().st_size / (1024 * 1024)
    print(f"Exported {checkpoint_path.name}: {orig_size:.1f} MB -> {new_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export compact checkpoints for easy GitHub upload.")
    parser.add_argument("--input_dir", type=str, default="checkpoints", help="Directory with full checkpoints.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/compact", help="Directory for compact checkpoints.")
    parser.add_argument("--fp16", action="store_true", help="Convert weights to half precision (FP16, ~3.1MB).")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    pt_files = list(in_dir.glob("*_best.pt"))
    if not pt_files:
        pt_files = [p for p in in_dir.glob("*.pt") if "compact" not in str(p)]

    print(f"Found {len(pt_files)} checkpoints to compress (FP16={args.fp16})...")
    for pt in pt_files:
        export_checkpoint(pt, out_dir, fp16=args.fp16)
    print(f"\nAll compact checkpoints saved to: '{out_dir}'")


if __name__ == "__main__":
    main()
