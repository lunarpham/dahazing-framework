"""
DehazeNet-Hybrid: Dual-Branch Physics + Direct Prediction Network.

Combines two complementary dehazing strategies through a shared encoder:
  - Physics Branch: Estimates transmission map t(x), recovers via Koschmieder's law.
    Excels at thin-to-moderate haze with physically plausible color preservation.
  - Direct Branch:  Learns hazy → clean residual mapping (output = head + input).
    Excels at thick haze where physics inversion amplifies errors (1/t explosion).

A learned fusion module produces per-pixel, per-channel confidence maps to
adaptively blend both outputs, so the network can rely on physics where haze
is thin and fall back to data-driven prediction where haze is thick.

Architecture:
    1. Shared Feature Encoder — 2× ConvBNReLU (3→32→64), 2× MultiScale+SE
    2. Physics Head           — 64→32→16→1, BReLU → t_map → Koschmieder → J_phys
    3. Direct Head            — 64→32→16→3, + input skip → J_direct
    4. Fusion Module          — Concat(J_phys, J_direct, feat) → α confidence
                                J_out = α * J_phys + (1-α) * J_direct

Estimated parameters: ~815K
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn

from .components import BReLU
from .dehazenet_plus import ConvBNReLU, MultiScaleBlock, SEBlock


class FusionModule(nn.Module):
    """
    Learned per-pixel, per-channel confidence gate for blending two branches.

    Input:  concat(J_physics, J_direct, deep_features)  →  (B, 3+3+64, H, W)
    Output: confidence map α (B, 3, H, W) in [0, 1]

    Final blend:  J = α * J_physics + (1 - α) * J_direct
    """

    def __init__(self, feat_channels=64):
        super().__init__()
        in_ch = 3 + 3 + feat_channels  # J_physics + J_direct + features
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, j_physics, j_direct, features):
        x = torch.cat([j_physics, j_direct, features], dim=1)
        alpha = self.gate(x)  # (B, 3, H, W), values in [0, 1]
        return alpha * j_physics + (1 - alpha) * j_direct


class DehazeNetHybrid(nn.Module):
    """
    Dual-branch hybrid dehazing model with learned fusion.

    The physics branch provides structural grounding through Koschmieder's law,
    while the direct branch handles cases where physics inversion is unstable.
    The fusion module learns when to trust each branch.
    """

    def __init__(self):
        super().__init__()

        # ── 1. Shared Feature Encoder ────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
        )

        # Multi-scale Stage 1
        self.ms_block1 = MultiScaleBlock(in_channels=64, branch_channels=64)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se1 = SEBlock(64, reduction=4)

        # Multi-scale Stage 2
        self.ms_block2 = MultiScaleBlock(in_channels=64, branch_channels=64)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.se2 = SEBlock(64, reduction=4)

        # ── 2. Physics Branch (transmission map) ────────────────────────────
        self.physics_head = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # no BN on output
        )
        self.brelu = BReLU(t_max=1.0)

        # ── 3. Direct Branch (clean image residual) ─────────────────────────
        self.direct_head = nn.Sequential(
            ConvBNReLU(64, 32, kernel_size=3, padding=1),
            ConvBNReLU(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # no BN on output
        )

        # ── 4. Fusion Module ────────────────────────────────────────────────
        self.fusion = FusionModule(feat_channels=64)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x, atm_light=None, t0=0.1):
        """
        Forward pass with internal physics recovery.

        Args:
            x:          Hazy input (B, 3, H, W) in [0, 1].
            atm_light:  Pre-computed atmospheric light (B, 3, 1, 1).
                        If None, uses a simple max-based estimate.
            t0:         Minimum transmission for Koschmieder clamping.

        Returns:
            J_out:      Fused dehazed image (B, 3, H, W) in [0, 1].
        """
        # ── Shared Encoder ───────────────────────────────────────────────────
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

        # ── Physics Branch ───────────────────────────────────────────────────
        t_map = self.physics_head(ms2)             # (B, 1, H, W)
        t_map = self.brelu(t_map)                  # clamp to [0, 1]

        # Atmospheric light estimation (simple fallback if not provided)
        if atm_light is None:
            # Per-image top-1% brightness as a quick A estimate
            B, C, H, W = x.shape
            flat = x.view(B, C, -1)                # (B, 3, H*W)
            num_top = max(1, H * W // 100)
            topk_vals, _ = flat.topk(num_top, dim=-1)
            atm_light = topk_vals.median(dim=-1).values  # (B, 3)
            atm_light = atm_light.view(B, C, 1, 1)

        # Koschmieder recovery: J = (I - A) / max(t, t0) + A
        t_clamped = torch.max(
            t_map,
            torch.tensor(t0, device=t_map.device, dtype=t_map.dtype)
        )
        j_physics = (x - atm_light) / t_clamped + atm_light
        j_physics = torch.clamp(j_physics, 0.0, 1.0)

        # ── Direct Branch ────────────────────────────────────────────────────
        j_direct = self.direct_head(ms2) + x       # global residual skip
        j_direct = torch.clamp(j_direct, 0.0, 1.0)

        # ── Fusion ───────────────────────────────────────────────────────────
        j_out = self.fusion(j_physics, j_direct, ms2)
        j_out = torch.clamp(j_out, 0.0, 1.0)

        return j_out

    def forward_with_intermediates(self, x, atm_light=None, t0=0.1):
        """
        Same as forward() but also returns intermediate outputs for
        visualization and debugging.

        Returns:
            dict with keys:
                'output':     Final fused result (B, 3, H, W)
                't_map':      Transmission map from physics branch (B, 1, H, W)
                'j_physics':  Physics-recovered image (B, 3, H, W)
                'j_direct':   Direct-predicted image (B, 3, H, W)
                'atm_light':  Atmospheric light used (B, 3, 1, 1)
        """
        # ── Shared Encoder ───────────────────────────────────────────────────
        feat = self.features(x)
        ms1 = self.se1(self.fuse1(self.ms_block1(feat))) + feat
        ms2 = self.se2(self.fuse2(self.ms_block2(ms1))) + ms1

        # ── Physics Branch ───────────────────────────────────────────────────
        t_map = self.brelu(self.physics_head(ms2))

        if atm_light is None:
            B, C, H, W = x.shape
            flat = x.view(B, C, -1)
            num_top = max(1, H * W // 100)
            topk_vals, _ = flat.topk(num_top, dim=-1)
            atm_light = topk_vals.median(dim=-1).values.view(B, C, 1, 1)

        t_clamped = torch.max(
            t_map,
            torch.tensor(t0, device=t_map.device, dtype=t_map.dtype)
        )
        j_physics = torch.clamp(
            (x - atm_light) / t_clamped + atm_light, 0.0, 1.0
        )

        # ── Direct Branch ────────────────────────────────────────────────────
        j_direct = torch.clamp(self.direct_head(ms2) + x, 0.0, 1.0)

        # ── Fusion ───────────────────────────────────────────────────────────
        j_out = torch.clamp(self.fusion(j_physics, j_direct, ms2), 0.0, 1.0)

        return {
            'output': j_out,
            't_map': t_map,
            'j_physics': j_physics,
            'j_direct': j_direct,
            'atm_light': atm_light,
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

        # Physics head: warm-start bias to 0.5 so initial t_map ~ 0.5
        # (well within BReLU [0,1] and above t0 clamp)
        physics_final = self.physics_head[-1]
        if isinstance(physics_final, nn.Conv2d) and physics_final.bias is not None:
            nn.init.constant_(physics_final.bias, 0.5)

        # Direct head: bias to 0 so initial output ~ input (residual identity)
        direct_final = self.direct_head[-1]
        if isinstance(direct_final, nn.Conv2d) and direct_final.bias is not None:
            nn.init.constant_(direct_final.bias, 0)

        # Fusion gate: bias the final conv to produce ~0.5 initially
        # so both branches contribute equally at the start of training
        fusion_conv = self.fusion.gate[-2]  # Conv2d before Sigmoid
        if isinstance(fusion_conv, nn.Conv2d) and fusion_conv.bias is not None:
            nn.init.constant_(fusion_conv.bias, 0)  # Sigmoid(0) = 0.5
