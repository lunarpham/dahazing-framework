"""
Configurable loss functions for DehazeNet.
Supports MSE, L1, SSIM, Perceptual, Sobel, Laplacian, DCP and weighted combinations.
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


class SobelLoss(nn.Module):
    """
    Sobel Gradient Loss — edge-preserving loss that penalizes differences in
    image gradient magnitude between prediction and ground truth.

    Motivation:
        MSE/L1 losses minimize pixel-level intensity differences, which
        encourages the network to produce blurry, averaged predictions near
        depth boundaries. This directly causes halo artifacts.

        SobelLoss enforces gradient consistency: the predicted image must
        have the same edge magnitudes as the ground truth at every boundary,
        which prevents halos (spurious gradient peaks along object edges).

    Implementation:
        Fixed (non-trainable) 3×3 Sobel kernels applied as depthwise
        convolution over all 3 RGB channels. L1 norm of the magnitude
        difference is used (not L2) to avoid averaging-induced blur.

            L_sobel = mean( | |∇pred| - |∇target| | )

    Reference:
        Sobel, I. (1990). An Isotropic 3x3 Image Gradient Operator.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Expand to 3-channel depthwise — non-trainable fixed kernels
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))

    def _gradient_magnitude(self, x):
        """Compute per-pixel gradient magnitude using Sobel operators."""
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        # L2 magnitude with epsilon to avoid NaN in sqrt backward pass
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        grad_pred   = self._gradient_magnitude(pred)
        grad_target = self._gradient_magnitude(target)
        return F.l1_loss(grad_pred, grad_target)


class LaplacianPyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss — multi-scale structural loss.

    Motivation:
        Halos occur at multiple spatial scales simultaneously (macro: building
        vs. sky; micro: vehicle edge details). A single Sobel pass at full
        resolution cannot address both. The Laplacian pyramid decomposes an
        image into frequency bands and enforces structural fidelity at each:

            L_lap = (1/L) * sum_l [ mean(|Lap_l(pred) - Lap_l(target)|) ]

        where Lap_l(I) = I_l - upsample(downsample(I_l)) captures the
        high-frequency residual at pyramid level l.

    Architecture:
        Uses a fixed 5×5 Gaussian kernel for blurring (non-trainable).
        Applied over `num_levels` successive pyramid levels.

    Reference:
        Burt & Adelson (1983). The Laplacian Pyramid as a Compact Image Code.
        IEEE Transactions on Communications.
    """

    def __init__(self, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Fixed 5×5 Gaussian kernel
        g1d = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        g2d = torch.outer(g1d, g1d)
        g2d = g2d / g2d.sum()
        kernel = g2d.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        self.register_buffer('gaussian_kernel', kernel)

    def _blur_downsample(self, x):
        """Gaussian blur then 2× spatial downsample."""
        blurred = F.conv2d(x, self.gaussian_kernel, padding=2, groups=3)
        return blurred[:, :, ::2, ::2]

    def _laplacian_level(self, x):
        """One Laplacian band: x - upsample(downsample(x))."""
        down = self._blur_downsample(x)
        up   = F.interpolate(down, size=x.shape[2:], mode='bilinear',
                             align_corners=False)
        return x - up

    def forward(self, pred, target):
        total = 0.0
        p, t = pred, target
        for _ in range(self.num_levels):
            lap_p = self._laplacian_level(p)
            lap_t = self._laplacian_level(t)
            total = total + F.l1_loss(lap_p, lap_t)
            p = self._blur_downsample(p)
            t = self._blur_downsample(t)
        return total / self.num_levels


class MixedLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.

    Supported keys (in mixed_loss_weights config):
        mse, l1, ssim, perceptual, dcp, sobel, laplacian

    Example config:
        l1 = 1.0
        ssim = 0.1
        sobel = 0.5
        laplacian = 0.3
        dcp = 0.01
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
        if "sobel" in weights:
            self.losses["sobel"] = SobelLoss()
        if "laplacian" in weights:
            self.losses["laplacian"] = LaplacianPyramidLoss()

    def forward(self, pred, target):
        total = 0.0
        for name, loss_fn in self.losses.items():
            total = total + self.weights[name] * loss_fn(pred, target)
        return total


def build_loss(config: dict) -> nn.Module:
    """
    Build a loss function from configuration.

    Config keys used:
        config["train"]["loss_type"]: "mse" | "l1" | "ssim" | "perceptual" |
                                      "sobel" | "laplacian" | "mixed"
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
    elif loss_type == "sobel":
        return SobelLoss()
    elif loss_type == "laplacian":
        return LaplacianPyramidLoss()
    elif loss_type == "mixed":
        weights = train_config.get("mixed_loss_weights", {"mse": 1.0})
        return MixedLoss(weights)
    else:
        raise ValueError(
            f"Unknown loss type: '{loss_type}'. "
            f"Supported: 'mse', 'l1', 'ssim', 'perceptual', 'sobel', 'laplacian', 'mixed'"
        )
