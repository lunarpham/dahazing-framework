"""
DehazeNet-Direct: Predicts the dehazed image directly (no physics inversion).

Instead of estimating a transmission map t(x) and inverting via Koschmieder's
law (which amplifies errors by 1/t for thick haze), this model learns a direct
mapping from hazy → clean.

Architecture mirrors DehazeNetPlus (multi-scale + SE) but outputs 3 channels
(RGB) with a global residual skip connection: output = input + learned_residual.

Estimated parameters: ~280K
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn

from .dehazenet_plus import ConvBNReLU, MultiScaleBlock, SEBlock


class DehazeNetDirect(nn.Module):
    """
    Direct image-to-image dehazing model.

    Architecture:
        1. Feature Extraction  — 2× ConvBNReLU (3 → 32 → 64)
        2. Multi-scale Stage 1 — 4 branches × 64ch → 256ch, fused to 64ch + SE
        3. Multi-scale Stage 2 — 4 branches × 64ch → 256ch, fused to 64ch + SE
        4. Residual merge       — add features from step 1
        5. Regression Head     — 3× Conv (64 → 32 → 16 → 3) with Sigmoid output
        6. Global Residual     — output = head(features) + hazy_input
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
        # Outputs 3 channels (RGB) instead of 1 (transmission)
        self.head = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # no BN on output
        )

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

        # 4. Regression + global residual
        # The head learns a residual correction; adding hazy input makes
        # the network only need to learn the "difference" (much easier).
        out = self.head(ms2) + x                   # (B, 3, H, W)
        out = torch.clamp(out, 0.0, 1.0)

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

        # Initialize the final conv bias to 0 so the initial output ≈ input
        # (since out = head(feat) + x, head starting at ~0 means out ≈ x)
        final_conv = self.head[-1]
        if isinstance(final_conv, nn.Conv2d) and final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0)
