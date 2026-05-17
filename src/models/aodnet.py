"""
AOD-Net: All-in-One Dehazing Network.

Reformulates the Atmospheric Scattering Model (ASM) to eliminate the unstable
division by t(x). Instead of estimating t(x) and A separately:

    Standard ASM inversion:  J(x) = (I(x) - A) / t(x) + A     ← divides by t(x)!

AOD-Net defines a unified parameter K(x) that absorbs both t(x) and A:

    J(x) = K(x) * I(x) - K(x) + 1                              ← no division!

This makes recovery purely multiplicative/additive, preventing gradient
explosions and division-by-zero in dense haze and sky regions.

The network uses multi-scale feature fusion: each convolutional layer operates
at a different kernel size (1, 3, 5, 7) and concatenates with previous layers
to capture both fine local details and broad global context simultaneously.

Reference:
    Li, B., Peng, X., Wang, Z., Xu, J., & Feng, D. (2017).
    AOD-Net: All-in-One Dehazing Network.
    Proceedings of the IEEE ICCV, 4770-4778.

Estimated parameters: ~1.9K (extremely lightweight)
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn


class AODNet(nn.Module):
    """
    All-in-One Dehazing Network with unified K(x) parameter.

    Architecture (multi-scale concatenation fusion):
        Layer 1: Conv2d 1×1 (3→3)   — point-wise feature extraction
        Layer 2: Conv2d 3×3 (3→3)   — local features
        Layer 3: Conv2d 5×5 (6→3)   — cat(L1,L2) → medium-range context
        Layer 4: Conv2d 7×7 (6→3)   — cat(L2,L3) → wide-range context
        Layer 5: Conv2d 3×3 (12→3)  — cat(L1,L2,L3,L4) → K(x) prediction

    Image recovery:
        J(x) = K(x) * I(x) - K(x) + 1
    """

    def __init__(self):
        super().__init__()

        # ── Multi-scale feature extraction ───────────────────────────────────
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: predict K(x) via multi-scale fusion, then recover J(x).

        Args:
            x: Hazy input image (B, 3, H, W) in [0, 1].

        Returns:
            Clean image J(x) = K(x)*I(x) - K(x) + 1, clamped to [0, 1].
        """
        # ── Multi-scale feature fusion ───────────────────────────────────────
        x1 = self.relu(self.conv1(x))                       # (B, 3, H, W)
        x2 = self.relu(self.conv2(x1))                      # (B, 3, H, W)

        cat1 = torch.cat((x1, x2), dim=1)                  # (B, 6, H, W)
        x3 = self.relu(self.conv3(cat1))                    # (B, 3, H, W)

        cat2 = torch.cat((x2, x3), dim=1)                  # (B, 6, H, W)
        x4 = self.relu(self.conv4(cat2))                    # (B, 3, H, W)

        cat3 = torch.cat((x1, x2, x3, x4), dim=1)          # (B, 12, H, W)

        # ── K(x) prediction ─────────────────────────────────────────────────
        k = self.relu(self.conv5(cat3))                     # (B, 3, H, W)

        # ── Image recovery (no division!) ────────────────────────────────────
        # J(x) = K(x) * I(x) - K(x) + 1
        clean = k * x - k + 1

        return torch.clamp(clean, 0.0, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
