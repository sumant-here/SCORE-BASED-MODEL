"""Model inference service with caching and Base64 image encoding."""

import base64
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.transforms import unnormalize_to_zero_one
from src.models import get_model, count_parameters
from src.sde import get_sde, BaseSDE
from src.diffusion.samplers import get_sampler
from src.training.checkpoint import load_checkpoint


class InferenceService:
    """Singleton-style service caching loaded PyTorch models for fast on-demand inference."""

    def __init__(self, device_pref: str = "auto"):
        self.device = get_device(device_pref)
        self.cache: Dict[str, Tuple[nn.Module, BaseSDE]] = {}

    def get_or_load_model(
        self,
        model_name: str,
        sde_name: str,
        checkpoint_path: Optional[str] = None,
        width: int = 32,
        depth: int = 2,
    ) -> Tuple[nn.Module, BaseSDE]:
        """Retrieve model and SDE from cache or initialize from parameters/checkpoint."""
        cache_key = f"{model_name}_{sde_name}_{checkpoint_path or 'init'}_{width}_{depth}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        sde = get_sde(sde_name)
        model = get_model(
            model_name,
            base_channels=width,
            num_res_blocks=depth,
            channel_multipliers=(1, 2, 2),
            attention_resolutions=(16,),
        ).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(checkpoint_path, model=model, device=self.device)

        model.eval()
        self.cache[cache_key] = (model, sde)
        return model, sde

    def generate(
        self,
        model_name: str,
        sde_name: str,
        num_samples: int = 16,
        steps: int = 100,
        sampler_name: str = "euler",
        seed: Optional[int] = 42,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[str], float]:
        """Generate images and return list of Base64 encoded PNG strings with latency."""
        if seed is not None:
            set_seed(seed)

        model, sde = self.get_or_load_model(model_name, sde_name, checkpoint_path=checkpoint_path)
        sampler_fn = get_sampler(sampler_name)

        start_time = time.time()
        shape = (num_samples, 3, 32, 32)

        with torch.no_grad():
            samples = sampler_fn(
                model=model,
                sde=sde,
                shape=shape,
                device=self.device,
                num_steps=steps,
                show_progress=False,
            )

        samples_01 = unnormalize_to_zero_one(samples)
        elapsed_time = time.time() - start_time

        # Convert to Base64 PNGs
        base64_images = []
        for i in range(num_samples):
            img_tensor = samples_01[i].cpu().permute(1, 2, 0).numpy()
            img_uint8 = (img_tensor * 255.0).round().astype("uint8")
            pil_img = Image.fromarray(img_uint8)

            # Optional upscale for visual clarity in API output
            pil_img = pil_img.resize((128, 128), Image.NEAREST)

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            base64_images.append(f"data:image/png;base64,{b64_str}")

        return base64_images, elapsed_time


# Global inference service instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
