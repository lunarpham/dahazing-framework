"""
Loss functions for MSFA-DeNet v2.
Supports: MSE, L1, SSIM, Perceptual, FFT, and weighted combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import calculate_ssim


class SSIMLoss(nn.Module):
    """1 - SSIM(pred, target)."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        return 1.0 - calculate_ssim(pred, target, window_size=self.window_size)


class PerceptualLoss(nn.Module):
    """VGG16 feature-matching loss (up to relu3_3)."""

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights='DEFAULT').features[:16]
        except Exception:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize from [0,1] to ImageNet mean/std."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        self.vgg = self.vgg.to(pred.device)
        pred_feat = self.vgg(self._normalize(pred))
        target_feat = self.vgg(self._normalize(target))
        return F.l1_loss(pred_feat, target_feat)


class FFTLoss(nn.Module):
    """Frequency-domain L1 loss via FFT magnitude comparison."""

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(pred_fft.abs(), target_fft.abs())


class MixedLoss(nn.Module):
    """Weighted combination of loss functions.

    Supported keys: mse, l1, ssim, perceptual, fft
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.weights = weights
        self.losses = nn.ModuleDict()

        if "mse" in weights:
            self.losses["mse"] = nn.MSELoss()
        if "l1" in weights:
            self.losses["l1"] = nn.L1Loss()
        if "ssim" in weights:
            self.losses["ssim"] = SSIMLoss()
        if "perceptual" in weights:
            self.losses["perceptual"] = PerceptualLoss()
        if "fft" in weights:
            self.losses["fft"] = FFTLoss()

    def forward(self, pred, target):
        total = 0.0
        for name, loss_fn in self.losses.items():
            total = total + self.weights[name] * loss_fn(pred, target)
        return total


def build_loss(config: dict) -> nn.Module:
    """Build a loss function from config.

    Uses config["train"]["loss_type"] and config["train"]["mixed_loss_weights"].
    """
    train_config = config.get("train", {})
    loss_type = train_config.get("loss_type", "mse").lower()

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "ssim":
        return SSIMLoss()
    elif loss_type == "perceptual":
        return PerceptualLoss()
    elif loss_type == "mixed":
        weights = train_config.get("mixed_loss_weights", {"mse": 1.0})
        return MixedLoss(weights)
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Supported: 'mse', 'l1', 'ssim', 'perceptual', 'mixed'"
        )