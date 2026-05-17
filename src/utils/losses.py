"""
Configurable loss functions for DehazeNet.
Supports MSE, L1, SSIM, Perceptual, and weighted combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import calculate_ssim


class SSIMLoss(nn.Module):
    """SSIM-based loss: 1 - SSIM(pred, target)."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        return 1.0 - calculate_ssim(pred, target, window_size=self.window_size)


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss.
    Compares high-level feature representations instead of raw pixels,
    producing sharper and more visually natural results.
    Uses VGG16 features up to relu3_3 (layer index 16).
    """

    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights='DEFAULT').features[:16]  # Up to relu3_3
        except Exception:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16]

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        # VGG expects ImageNet-normalized input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """Normalize from [0,1] to ImageNet mean/std."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def forward(self, pred, target):
        self.vgg = self.vgg.to(pred.device)
        pred_feat = self.vgg(self._normalize(pred))
        target_feat = self.vgg(self._normalize(target))
        return F.l1_loss(pred_feat, target_feat)


class DarkChannelLoss(nn.Module):
    """
    Physics-based Dark Channel Prior (DCP) loss.

    Observation: in clear, outdoor, non-sky images, at least one RGB channel
    in a local patch has very low intensity (close to zero). If the predicted
    clean image has a bright dark channel, it still contains haze residue.

    Loss = mean(dark_channel(predicted_clean_image))

    This acts as a physics regularizer, encouraging the network to produce
    outputs consistent with the DCP assumption.

    Reference:
        He, K., Sun, J., & Tang, X. (2010). Single Image Haze Removal Using
        Dark Channel Prior. IEEE TPAMI, 33(12), 2341-2353.
    """

    def __init__(self, patch_size=15):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, pred, target=None):
        """
        Compute DCP loss on the predicted image.
        The target argument is accepted but ignored (for API compatibility
        with MixedLoss which passes both pred and target to all sub-losses).
        """
        # 1. Minimum across color channels
        min_channels, _ = torch.min(pred, dim=1, keepdim=True)

        # 2. Minimum across local spatial patch (using -MaxPool trick)
        pad = self.patch_size // 2
        dark_channel = -F.max_pool2d(
            -min_channels,
            kernel_size=self.patch_size,
            stride=1,
            padding=pad,
        )

        # Mean of dark channel — lower = cleaner image
        return torch.mean(dark_channel)


class MixedLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.
    Example: 1.0 * L1 + 0.2 * (1 - SSIM) + 0.1 * Perceptual + 0.01 * DCP
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
        if "dcp" in weights:
            self.losses["dcp"] = DarkChannelLoss()

    def forward(self, pred, target):
        total = 0.0
        for name, loss_fn in self.losses.items():
            total = total + self.weights[name] * loss_fn(pred, target)
        return total


def build_loss(config: dict) -> nn.Module:
    """
    Build a loss function from configuration.

    Config keys used:
        config["train"]["loss_type"]: "mse" | "l1" | "ssim" | "perceptual" | "mixed"
        config["train"]["mixed_loss_weights"]: dict of {loss_name: weight}

    Returns:
        An nn.Module loss function.
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

