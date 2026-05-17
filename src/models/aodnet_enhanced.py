"""
AOD-Net Enhanced: AOD-Net's K(x) formulation + deeper multi-scale features.

Keeps the core insight from AOD-Net — the unified K(x) parameter that makes
image recovery purely multiplicative/additive (no division by t(x)) — but
replaces the lightweight 5-layer architecture with a deeper feature backbone
using BatchNorm, LeakyReLU, SE channel attention, and residual connections.

This bridges the gap between:
  - AOD-Net (~1.9K params, fast but limited capacity)
  - DehazeNetPlus (~770K params, powerful but divides by t(x))

Physics recap:
    Standard ASM: J(x) = (I(x) - A) / t(x) + A            ← unstable
    AOD-Net:      J(x) = K(x) * I(x) - K(x) + 1           ← stable

Architecture:
    1. Feature Encoder   — 2× ConvBNReLU (3→32→64)
    2. Multi-scale + SE  — 2 stages of 4-branch parallel conv + SE attention
    3. K(x) Head         — 64→32→16→3 (predicts per-pixel K parameter)
    4. Recovery           — J = K*I - K + 1 (no division)

Estimated parameters: ~770K
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn

from .dehazenet_plus import ConvBNReLU, MultiScaleBlock, SEBlock


class AODNetEnhanced(nn.Module):
    """
    Enhanced AOD-Net with deep multi-scale features and SE attention.

    Uses the same encoder backbone as DehazeNetPlus/Direct but predicts
    the unified K(x) parameter instead of t(x) or a direct residual,
    gaining the stability benefits of AOD-Net's reformulation with the
    representational power of a deeper feature extractor.
    """

    def __init__(self):
        super().__init__()

        # ── 1. Feature Encoder ───────────────────────────────────────────────
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

        # ── 4. K(x) Prediction Head ─────────────────────────────────────────
        # Outputs 3 channels: per-channel K(x) for RGB
        self.k_head = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # no BN on output
            nn.ReLU(inplace=True),  # K(x) must be non-negative for stable recovery
        )

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: deep features → K(x) → stable image recovery.

        Args:
            x: Hazy input image (B, 3, H, W) in [0, 1].

        Returns:
            Clean image J(x) = K(x)*I(x) - K(x) + 1, clamped to [0, 1].
        """
        # ── Feature extraction ───────────────────────────────────────────────
        feat = self.features(x)                    # (B, 64, H, W)

        # Multi-scale stage 1 + residual
        ms1 = self.ms_block1(feat)                 # (B, 256, H, W)
        ms1 = self.fuse1(ms1)                      # (B, 64, H, W)
        ms1 = self.se1(ms1)                        # (B, 64, H, W)
        ms1 = ms1 + feat                           # residual skip

        # Multi-scale stage 2 + residual
        ms2 = self.ms_block2(ms1)                  # (B, 256, H, W)
        ms2 = self.fuse2(ms2)                      # (B, 64, H, W)
        ms2 = self.se2(ms2)                        # (B, 64, H, W)
        ms2 = ms2 + ms1                            # residual skip

        # ── K(x) prediction ─────────────────────────────────────────────────
        k = self.k_head(ms2)                       # (B, 3, H, W), non-negative

        # ── Image recovery (no division!) ────────────────────────────────────
        # J(x) = K(x) * I(x) - K(x) + 1
        clean = k * x - k + 1                      # (B, 3, H, W)

        return torch.clamp(clean, 0.0, 1.0)

    def forward_with_k(self, x):
        """
        Same as forward() but also returns K(x) for visualization/analysis.

        Returns:
            dict with keys:
                'output': Final dehazed image (B, 3, H, W)
                'k_map':  Predicted K(x) parameter (B, 3, H, W)
        """
        feat = self.features(x)
        ms1 = self.se1(self.fuse1(self.ms_block1(feat))) + feat
        ms2 = self.se2(self.fuse2(self.ms_block2(ms1))) + ms1

        k = self.k_head(ms2)
        clean = torch.clamp(k * x - k + 1, 0.0, 1.0)

        return {
            'output': clean,
            'k_map': k,
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )

        # Initialize K head final bias so that K starts near 1.
        # When K=1: J = 1*I - 1 + 1 = I (identity), so the network
        # starts by outputting the input image and learns corrections.
        k_final = self.k_head[-2]  # Conv2d before ReLU
        if isinstance(k_final, nn.Conv2d) and k_final.bias is not None:
            nn.init.constant_(k_final.bias, 1.0)
