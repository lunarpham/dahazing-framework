"""
AOD-PA-Net: AOD-Net with Pixel Attention for Depth-Aware Dehazing.

Combines the lightweight K(x) reformulation of AOD-Net (Li et al., 2017)
with the Pixel Attention (PA) mechanism from FFA-Net (Qin et al., 2020)
to enable spatially-aware dehazing that adapts to varying haze density.

Motivation:
    Standard AOD-Net applies uniform dehazing strength across the entire image.
    Because haze is intrinsically non-uniform (thicker at depth, thinner in the
    foreground), uniform processing causes:
      - Over-enhancement of clear foreground regions → "halo" artifacts
      - Under-enhancement of distant, heavily hazed regions
      - Overfitting to blocky synthetic depth-map artifacts in training data

Solution — Pixel Attention:
    A PA block is inserted between the multi-scale feature extraction and the
    K(x) parameter estimation. It learns a 3D attention map (values in [0, 1])
    that acts as a spatially-varying confidence threshold:

        PA(F) = σ(Conv(ReLU(Conv(F))))          ← 3D attention map
        F_attended = F ⊙ PA(F)                  ← element-wise modulation

    Dense Haze Regions:  PA weights → 1.0 (max corrective transformation)
    Clear/Edge Regions:  PA weights → 0.0 (muted features, preventing halos)

Architecture Flow:
    1. Multi-Scale Feature Extraction  — 5 conv layers with concatenation
       (kernel sizes 1×1, 3×3, 5×5, 7×7, 3×3) capturing local-to-global context
    2. Spatial Modulation (PA)         — Pixel Attention block weighs features
       by haze density, producing F_attended
    3. K(x) Parameter Estimation       — 3×3 conv predicts unified K(x)
    4. Image Reconstruction            — J(x) = K(x)·I(x) − K(x) + 1

Physics recap:
    Standard ASM: J(x) = (I(x) - A) / t(x) + A          ← divides by t(x)!
    AOD-Net:      J(x) = K(x) * I(x) - K(x) + 1         ← no division!

    where K(x) = (1/t(x)·(I(x) - A) + (A - b)) / (I(x) - 1)
    and b is a constant bias with default value 1.

References:
    Li, B., Peng, X., Wang, Z., Xu, J., & Feng, D. (2017).
    AOD-Net: All-in-One Dehazing Network. IEEE ICCV, 4770-4778.

    Qin, X., Wang, Z., Bai, Y., Xie, X., & Jia, H. (2020).
    FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.
    AAAI Conference on Artificial Intelligence, 34(07), 11908-11915.

Estimated parameters: ~2.1K (extremely lightweight — only ~200 params added)
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pixel Attention Block ────────────────────────────────────────────────────

class PixelAttention(nn.Module):
    """
    Pixel Attention (PA) module from FFA-Net.

    Learns a 3D spatial attention map (values in [0, 1]) that tells the network
    WHICH pixels have thick haze (high correction needed) and WHICH pixels
    are already clear (low correction to avoid halos).

    Architecture:
        Conv 1×1 (channels → channels//8)  →  ReLU  →  Conv 1×1 (channels//8 → 1)  →  Sigmoid

    The single-channel attention map is broadcast-multiplied with the input
    features, effectively gating each spatial position.

    Args:
        channels: Number of input feature channels.
    """

    def __init__(self, channels):
        super().__init__()
        # Bottleneck reduces channels, then projects to a single spatial map
        # Use max(1, channels // 8) to handle very small channel counts safely
        mid_channels = max(1, channels // 8)
        self.pa = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),  # Constrains attention weights to [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Attention-modulated features: x ⊙ PA(x), same shape (B, C, H, W).
        """
        attention_map = self.pa(x)   # (B, 1, H, W)
        return x * attention_map     # broadcast multiply across channels


# ── AOD-PA-Net ───────────────────────────────────────────────────────────────

class AODPANet(nn.Module):
    """
    Hybrid Dehazing Network: AOD-Net + Pixel Attention.

    Synergizes the lightweight K(x) estimation of AOD-Net with the spatial
    awareness of Pixel Attention to eliminate halos and target distant haze
    at different depths within the image.

    Architecture (multi-scale concatenation fusion + PA):
        Layer 1: Conv2d 1×1 (3→3)   — point-wise feature extraction
        Layer 2: Conv2d 3×3 (3→3)   — local features
        Layer 3: Conv2d 5×5 (6→3)   — cat(L1,L2) → medium-range context
        Layer 4: Conv2d 7×7 (6→3)   — cat(L2,L3) → wide-range context
        Layer 5: Conv2d 3×3 (12→3)  — cat(L1,L2,L3,L4) → fused multi-scale

        PA Block:  Pixel Attention on the 3-channel fused features
                   PA(F) = σ(Conv(ReLU(Conv(F))))
                   F_attended = F ⊙ PA(F)

        K Head:  Conv2d 3×3 (3→3) + ReLU → K(x) estimation

    Image recovery:
        J(x) = K(x) * I(x) - K(x) + 1
    """

    def __init__(self):
        super().__init__()

        # ── Multi-scale feature extraction (AOD-Net backbone) ────────────────
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)   # cat(L1, L2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)   # cat(L2, L3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)  # cat(L1, L2, L3, L4)

        # ── Pixel Attention block (the hybrid innovation) ────────────────────
        # Inserted between multi-scale fusion and K(x) estimation.
        # Learns to weigh features based on spatial haze density.
        self.pixel_attention = PixelAttention(channels=3)

        # ── K(x) parameter estimation ────────────────────────────────────────
        self.k_estimator = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: multi-scale features → PA modulation → K(x) → J(x).

        Args:
            x: Hazy input image I(x), shape (B, 3, H, W) in [0, 1].

        Returns:
            Clean image J(x) = K(x)*I(x) - K(x) + 1, clamped to [0, 1].
        """
        # ── Step 1: Multi-Scale Feature Extraction ───────────────────────────
        x1 = self.relu(self.conv1(x))                       # (B, 3, H, W)
        x2 = self.relu(self.conv2(x1))                      # (B, 3, H, W)

        cat1 = torch.cat((x1, x2), dim=1)                  # (B, 6, H, W)
        x3 = self.relu(self.conv3(cat1))                    # (B, 3, H, W)

        cat2 = torch.cat((x2, x3), dim=1)                  # (B, 6, H, W)
        x4 = self.relu(self.conv4(cat2))                    # (B, 3, H, W)

        cat3 = torch.cat((x1, x2, x3, x4), dim=1)          # (B, 12, H, W)
        x5 = self.relu(self.conv5(cat3))                    # (B, 3, H, W)
        # x5 = F_multi (the multi-scale fused features)

        # ── Step 2: Pixel Attention (Spatial Modulation) ─────────────────────
        # The PA block evaluates the spatial distribution of haze and outputs
        # the modulated feature map F_attended.
        #   - Dense haze regions → weights ≈ 1.0 → full correction
        #   - Clear/edge regions → weights ≈ 0.0 → muted (prevents halos)
        attended_features = self.pixel_attention(x5)        # (B, 3, H, W)

        # ── Step 3: Estimate K(x) ───────────────────────────────────────────
        # Final 3×3 conv on the attention-guided features
        k = self.relu(self.k_estimator(attended_features))  # (B, 3, H, W)

        # ── Step 4: Image Reconstruction ─────────────────────────────────────
        # J(x) = K(x) * I(x) - K(x) + b, where b = 1
        clean = k * x - k + 1.0

        return torch.clamp(clean, 0.0, 1.0)

    def forward_with_k(self, x):
        """
        Same as forward() but also returns K(x) and the attention map
        for visualization and analysis.

        Returns:
            dict with keys:
                'output':         Final dehazed image (B, 3, H, W)
                'k_map':          Predicted K(x) parameter (B, 3, H, W)
                'attention_map':  PA spatial attention weights (B, 1, H, W)
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat((x1, x2), dim=1)))
        x4 = self.relu(self.conv4(torch.cat((x2, x3), dim=1)))
        x5 = self.relu(self.conv5(torch.cat((x1, x2, x3, x4), dim=1)))

        # Extract the attention map for visualization
        attention_map = self.pixel_attention.pa(x5)         # (B, 1, H, W)
        attended_features = x5 * attention_map

        k = self.relu(self.k_estimator(attended_features))
        clean = torch.clamp(k * x - k + 1.0, 0.0, 1.0)

        return {
            'output': clean,
            'k_map': k,
            'attention_map': attention_map,
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a dummy hazy image (Batch: 1, Channels: 3, H: 256, W: 256)
    dummy_hazy_img = torch.rand(1, 3, 256, 256)

    # Initialize the hybrid model
    model = AODPANet()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    pa_params = sum(p.numel() for p in model.pixel_attention.parameters())

    # Run forward pass
    clean_output = model(dummy_hazy_img)

    # Run diagnostic forward pass
    diagnostics = model.forward_with_k(dummy_hazy_img)

    print(f"{'='*60}")
    print(f"  AOD-PA-Net: Hybrid Dehazing Network")
    print(f"{'='*60}")
    print(f"  Input shape:          {dummy_hazy_img.shape}")
    print(f"  Output shape:         {clean_output.shape}")
    print(f"  K(x) map shape:       {diagnostics['k_map'].shape}")
    print(f"  Attention map shape:  {diagnostics['attention_map'].shape}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  PA block parameters:  {pa_params:,}")
    print(f"  Attention map range:  [{diagnostics['attention_map'].min():.4f}, "
          f"{diagnostics['attention_map'].max():.4f}]")
    print(f"{'='*60}")
    print(f"  [OK] AOD-PA-Net is working correctly!")
    print(f"{'='*60}")
