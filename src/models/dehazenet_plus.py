"""
DehazeNet-Plus: A heavier variant of DehazeNet for transmission map estimation.

Key improvements over the original DehazeNet:
- Deeper feature extraction with BatchNorm + LeakyReLU
- Two-stage multi-scale mapping with 4 branches (k1/3/5/7)
- Squeeze-and-Excitation (SE) channel attention for adaptive feature fusion
- Residual skip connection from features to regression head
- Deeper regression head (64 → 32 → 16 → 1)
- Kaiming initialization with warm-start bias on final layer

Estimated parameters: ~280K (vs ~23K for DehazeNet)
Output: (B, 1, H, W) transmission map in [0, 1] — same interface as DehazeNet.
"""

import torch
import torch.nn as nn

from .components import BReLU


# ── Building Blocks ──────────────────────────────────────────────────────────


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + LeakyReLU convenience block."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Learns per-channel scaling factors via global pooling → FC → Sigmoid.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)
        scale = self.excite(scale).view(B, C, 1, 1)
        return x * scale


class MultiScaleBlock(nn.Module):
    """
    4-branch parallel convolution at kernel sizes 1, 3, 5, 7.
    Each branch maps in_channels → branch_channels.
    Outputs are concatenated → 4 * branch_channels.
    """

    def __init__(self, in_channels, branch_channels):
        super().__init__()
        bc = branch_channels
        self.branch1 = ConvBNReLU(in_channels, bc, kernel_size=1, padding=0)
        self.branch3 = ConvBNReLU(in_channels, bc, kernel_size=3, padding=1)
        self.branch5 = ConvBNReLU(in_channels, bc, kernel_size=5, padding=2)
        self.branch7 = ConvBNReLU(in_channels, bc, kernel_size=7, padding=3)

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch7(x),
        ], dim=1)


# ── Main Model ───────────────────────────────────────────────────────────────


class DehazeNetPlus(nn.Module):
    """
    A heavier CNN model for transmission map estimation.
    
    Architecture:
        1. Feature Extraction  — 2× ConvBNReLU (3 → 32 → 64)
        2. Multi-scale Stage 1 — 4 branches × 64ch → 256ch, fused to 64ch via 1×1 conv + SE
        3. Multi-scale Stage 2 — 4 branches × 64ch → 256ch, fused to 64ch via 1×1 conv + SE
        4. Residual merge       — add features from step 1
        5. Regression Head     — 3× Conv (64 → 32 → 16 → 1) with BReLU output
    """

    def __init__(self):
        super().__init__()

        # ── 1. Feature Extraction ────────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
        )

        # ── 2. Multi-scale Stage 1 ──────────────────────────────────────────
        self.ms_block1 = MultiScaleBlock(in_channels=64, branch_channels=64)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se1 = SEBlock(64, reduction=4)

        # ── 3. Multi-scale Stage 2 ──────────────────────────────────────────
        self.ms_block2 = MultiScaleBlock(in_channels=64, branch_channels=64)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se2 = SEBlock(64, reduction=4)

        # ── 4. Regression Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # no BN on output
        )
        self.brelu = BReLU(t_max=1.0)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        # 1. Feature extraction
        feat = self.features(x)                    # (B, 64, H, W)

        # 2. Multi-scale stage 1 + residual
        ms1 = self.ms_block1(feat)                 # (B, 256, H, W)
        ms1 = self.fuse1(ms1)                      # (B, 64, H, W)
        ms1 = self.se1(ms1)                        # (B, 64, H, W)
        ms1 = ms1 + feat                           # residual skip

        # 3. Multi-scale stage 2 + residual
        ms2 = self.ms_block2(ms1)                  # (B, 256, H, W)
        ms2 = self.fuse2(ms2)                      # (B, 64, H, W)
        ms2 = self.se2(ms2)                        # (B, 64, H, W)
        ms2 = ms2 + ms1                            # residual skip

        # 4. Regression
        out = self.head(ms2)                       # (B, 1, H, W)
        out = self.brelu(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Warm-start the final conv bias to 0.5 so the initial transmission
        # map is well within BReLU [0, 1] and above the t0 clamp in Koschmieder.
        final_conv = self.head[-1]
        if isinstance(final_conv, nn.Conv2d) and final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0.5)
