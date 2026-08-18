"""Inception Score (IS) computation for generative models (Salimans et al., 2016)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as T
from typing import Tuple, Optional


class InceptionFeatureExtractor(nn.Module):
    """Wrapper around InceptionV3 or MobileNet for feature and probability extraction."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
            self.model = models.inception_v3(weights=weights, transform_input=False)
            self.model.eval()
            self.is_inception = True
        except Exception:
            # Fallback to lightweight model if InceptionV3 weights cannot be downloaded
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            self.model.eval()
            self.is_inception = False

        self.resize = T.Resize((299, 299), antialias=True) if self.is_inception else T.Resize((224, 224), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract pooled features and classification probabilities.

        Args:
            x: Tensor in [0, 1] range of shape (B, 3, H, W).

        Returns:
            features: 2048-dim feature activations (or 1280 for mobilenet).
            probs: 1000-dim softmax probabilities.
        """
        x_resized = self.resize(x)
        x_norm = self.normalize(x_resized)

        if self.is_inception:
            # Inception V3 forward
            x = self.model.Conv2d_1a_3x3(x_norm)
            x = self.model.Conv2d_2a_3x3(x)
            x = self.model.Conv2d_2b_3x3(x)
            x = self.model.maxpool1(x)
            x = self.model.Conv2d_3b_1x1(x)
            x = self.model.Conv2d_4a_3x3(x)
            x = self.model.maxpool2(x)
            x = self.model.Mixed_5b(x)
            x = self.model.Mixed_5c(x)
            x = self.model.Mixed_5d(x)
            x = self.model.Mixed_6a(x)
            x = self.model.Mixed_6b(x)
            x = self.model.Mixed_6c(x)
            x = self.model.Mixed_6d(x)
            x = self.model.Mixed_6e(x)
            x = self.model.Mixed_7a(x)
            x = self.model.Mixed_7b(x)
            x = self.model.Mixed_7c(x)
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)
            logits = self.model.fc(self.model.dropout(features))
            probs = F.softmax(logits, dim=1)
        else:
            features = self.model.features(x_norm)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            logits = self.model.classifier(features)
            probs = F.softmax(logits, dim=1)

        return features, probs


def calculate_inception_score(
    images: torch.Tensor,
    splits: int = 10,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    feature_extractor: Optional[InceptionFeatureExtractor] = None,
) -> Tuple[float, float]:
    """Calculate Inception Score (mean and standard deviation) on generated images.

    Args:
        images: Generated image tensor in [0, 1] or [-1, 1] range of shape (N, 3, H, W).
        splits: Number of splits for calculating variance.
        batch_size: Batch size for feature extractor.
        device: Device to run evaluation on.
        feature_extractor: Optional pre-loaded InceptionFeatureExtractor.

    Returns:
        (is_mean, is_std)
    """
    if device is None:
        device = images.device

    # Ensure in [0, 1]
    if images.min() < 0:
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)

    if feature_extractor is None:
        feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()

    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            _, probs = feature_extractor(batch)
            preds_list.append(probs.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    N = preds.shape[0]
    split_size = max(1, N // splits)

    scores = []
    for i in range(splits):
        part = preds[i * split_size : (i + 1) * split_size, :]
        if len(part) == 0:
            continue
        # Marginal probability distribution p(y)
        p_y = np.mean(part, axis=0, keepdims=True)
        # KL Divergence: sum p(y|x) * (log p(y|x) - log p(y))
        kl = part * (np.log(part + 1e-10) - np.log(p_y + 1e-10))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))
