"""
AOD-CA-PA-Net: AOD-Net + Channel Attention → Pixel Attention Cascade.

Extends AOD-PA-Net by inserting a Channel Attention (CA) module *before*
the Pixel Attention (PA) block, forming the Feature Attention (FA) cascade
inspired by FFA-Net.

Why the cascade matters:
    In dense haze, feature channels become dominated by uniform atmospheric
    scattering statistics. Pixel Attention fed these uniform features cannot
    differentiate a white vehicle from the ambient fog — it collapses.

    Channel Attention first:
        Identifies which channels carry atmospheric scattering signal vs.
        scene structure, then suppresses the haze-dominated channels via
        learned per-channel weights (squeeze-and-excitation style).

    Pixel Attention second:
        Now operates on channel-cleaned features, where structural edges
        and depth cues are no longer overwhelmed by haze statistics.
        Spatial weights are meaningful and prevent halos.

Architecture Flow:
    I(x)
      ↓
    [Conv1×1 → Conv3×3 → Conv5×5 → Conv7×7] (multi-scale backbone)
      ↓ cat (12 channels)
    [Conv3×3] → F_multi  (3 channels)
      ↓
    ┌── Channel Attention ──────────────────┐
    │  GlobalAvgPool → FC(3→max(1,3//8))   │
    │  → ReLU → FC → Sigmoid               │  → CA weights  ∈ [0,1] per channel
    └───────────────────────────────────────┘
      ↓  F_ca = F_multi ⊙ CA(F_multi)
    ┌── Pixel Attention ─────────────────────┐
    │  Conv1×1 → ReLU → Conv1×1 → Sigmoid  │  → PA map  ∈ [0,1] per pixel
    └────────────────────────────────────────┘
      ↓  F_attended = F_ca ⊙ PA(F_ca)
    [Conv3×3 + ReLU] → K(x)
      ↓
    J(x) = K(x)·I(x) − K(x) + 1   (clamped to [0,1])

Physics:
    Standard ASM:  J(x) = (I(x) - A) / t(x) + A    ← unstable (divides by t)
    AOD-Net:       J(x) = K(x)·I(x) - K(x) + 1     ← no division, stable

References:
    Li et al. (2017). AOD-Net: All-in-One Dehazing Network. ICCV.
    Qin et al. (2020). FFA-Net: Feature Fusion Attention Network. AAAI.

Estimated parameters: ~1,870  (only ~19 more than AOD-PA-Net)
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Channel Attention (Squeeze-and-Excitation style) ─────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) module.

    Performs global context aggregation (squeeze) then learns per-channel
    importance weights (excitation) to suppress channels dominated by
    atmospheric scattering while preserving structurally-informative channels.

    Architecture:
        GlobalAvgPool → FC(C → C//8, min 1) → ReLU → FC(C//8 → C) → Sigmoid

    Args:
        channels: Number of input feature channels.
        reduction: Channel reduction ratio for the bottleneck FC layer.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1) — global squeeze
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=True),
            nn.Sigmoid(),                      # per-channel weight in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Channel-scaled features: x ⊙ CA(x), same shape (B, C, H, W).
        """
        weights = self.ca(x)                  # (B, C)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weights                    # broadcast over H×W


# ── Pixel Attention ───────────────────────────────────────────────────────────

class PixelAttention(nn.Module):
    """
    Pixel Attention (PA) module.

    Learns a 3D spatial attention map (values in [0, 1]) that tells the
    network WHICH pixels have dense haze (apply full correction) and WHICH
    are clear (mute features, preventing halos).

    Architecture:
        Conv1×1 (C → C//8) → ReLU → Conv1×1 (C//8 → 1) → Sigmoid

    Args:
        channels: Number of input feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.pa = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),                     # spatial weight in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W).

        Returns:
            Attention-modulated features: x ⊙ PA(x), same shape (B, C, H, W).
        """
        attn = self.pa(x)                     # (B, 1, H, W)
        return x * attn                       # broadcast across channels


# ── Feature Attention: CA → PA Cascade ───────────────────────────────────────

class FeatureAttention(nn.Module):
    """
    Feature Attention (FA) cascade: Channel Attention followed by Pixel Attention.

    Mirrors the FA module from FFA-Net. The sequential ordering is critical:
        1. CA suppresses entire haze-dominated channels globally.
        2. PA then maps remaining structural features spatially.
    Applying them in reverse (PA first) allows PA to collapse on dense,
    channel-uniform haze before CA can clean it.

    Args:
        channels: Number of input feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=8)
        self.pa = PixelAttention(channels)

    def forward(self, x):
        """
        Args:
            x: F_multi feature map (B, C, H, W).

        Returns:
            F_attended: channel- and pixel-attended features (B, C, H, W).
        """
        x = self.ca(x)   # step 1: suppress haze channels
        x = self.pa(x)   # step 2: spatially weight remaining features
        return x


# ── AOD-CA-PA-Net ─────────────────────────────────────────────────────────────

class AODCAPANet(nn.Module):
    """
    AOD-Net + Channel Attention → Pixel Attention cascade (AOD-CA-PA-Net).

    The CA→PA Feature Attention block replaces the standalone PA block
    of AOD-PA-Net, providing a two-stage attention mechanism that is
    robust to dense haze and suppresses halo artifacts at depth boundaries.

    Architecture (multi-scale backbone + FA cascade):
        Layer 1: Conv2d 1×1  (3→3)   — point-wise feature extraction
        Layer 2: Conv2d 3×3  (3→3)   — local features
        Layer 3: Conv2d 5×5  (6→3)   — cat(L1,L2) → medium-range context
        Layer 4: Conv2d 7×7  (6→3)   — cat(L2,L3) → wide-range context
        Layer 5: Conv2d 3×3  (12→3)  — cat(L1,L2,L3,L4) → F_multi

        FA Block: CA(F_multi) → PA → F_attended

        K Head:  Conv2d 3×3 (3→3) + ReLU → K(x)

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
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)  # cat(L1..L4)

        # ── Feature Attention: CA → PA cascade (the core improvement) ────────
        self.feature_attention = FeatureAttention(channels=3)

        # ── K(x) estimation ──────────────────────────────────────────────────
        self.k_estimator = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: multi-scale backbone → CA→PA cascade → K(x) → J(x).

        Args:
            x: Hazy input image I(x), shape (B, 3, H, W) in [0, 1].

        Returns:
            Clean image J(x) = K(x)*I(x) - K(x) + 1, clamped to [0, 1].
        """
        # ── Step 1: Multi-Scale Feature Extraction ───────────────────────────
        x1 = self.relu(self.conv1(x))                        # (B, 3, H, W)
        x2 = self.relu(self.conv2(x1))                       # (B, 3, H, W)

        cat1 = torch.cat((x1, x2), dim=1)                   # (B, 6, H, W)
        x3 = self.relu(self.conv3(cat1))                     # (B, 3, H, W)

        cat2 = torch.cat((x2, x3), dim=1)                   # (B, 6, H, W)
        x4 = self.relu(self.conv4(cat2))                     # (B, 3, H, W)

        cat3 = torch.cat((x1, x2, x3, x4), dim=1)           # (B, 12, H, W)
        f_multi = self.relu(self.conv5(cat3))                # (B, 3, H, W)

        # ── Step 2: Feature Attention (CA → PA) ──────────────────────────────
        # CA first: suppresses channels dominated by uniform haze statistics.
        # PA second: maps spatial haze density on the channel-cleaned features.
        f_attended = self.feature_attention(f_multi)         # (B, 3, H, W)

        # ── Step 3: K(x) Estimation ─────────────────────────────────────────
        k = self.relu(self.k_estimator(f_attended))          # (B, 3, H, W)

        # ── Step 4: Image Reconstruction ─────────────────────────────────────
        # J(x) = K(x) * I(x) - K(x) + b,  where b = 1
        clean = k * x - k + 1.0

        return torch.clamp(clean, 0.0, 1.0)

    def forward_with_maps(self, x):
        """
        Diagnostic forward pass returning K(x), CA weights, and PA attention map.

        Returns:
            dict with keys:
                'output':       Final dehazed image (B, 3, H, W)
                'k_map':        Predicted K(x) parameter (B, 3, H, W)
                'ca_weights':   Per-channel CA weights (B, 3) in [0, 1]
                'pa_map':       Spatial PA attention map (B, 1, H, W)
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat((x1, x2), dim=1)))
        x4 = self.relu(self.conv4(torch.cat((x2, x3), dim=1)))
        f_multi = self.relu(self.conv5(torch.cat((x1, x2, x3, x4), dim=1)))

        # Extract CA weights (before multiplying)
        fa = self.feature_attention
        ca_weights_flat = fa.ca.ca(f_multi)            # (B, 3)
        ca_weights = ca_weights_flat.unsqueeze(-1).unsqueeze(-1)
        f_ca = f_multi * ca_weights                    # channel-attended features

        # Extract PA map
        pa_map = fa.pa.pa(f_ca)                        # (B, 1, H, W)
        f_attended = f_ca * pa_map

        k = self.relu(self.k_estimator(f_attended))
        clean = torch.clamp(k * x - k + 1.0, 0.0, 1.0)

        return {
            'output':     clean,
            'k_map':      k,
            'ca_weights': ca_weights_flat,             # (B, 3) — one per channel
            'pa_map':     pa_map,                      # (B, 1, H, W) — spatial
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy = torch.rand(1, 3, 256, 256)
    model = AODCAPANet()

    total_params  = sum(p.numel() for p in model.parameters())
    fa_params     = sum(p.numel() for p in model.feature_attention.parameters())
    ca_params     = sum(p.numel() for p in model.feature_attention.ca.parameters())
    pa_params     = sum(p.numel() for p in model.feature_attention.pa.parameters())

    output = model(dummy)
    diag   = model.forward_with_maps(dummy)

    print("=" * 60)
    print("  AOD-CA-PA-Net: CA->PA Cascade Dehazing Network")
    print("=" * 60)
    print(f"  Input shape:        {dummy.shape}")
    print(f"  Output shape:       {output.shape}")
    print(f"  K(x) map shape:     {diag['k_map'].shape}")
    print(f"  CA weights shape:   {diag['ca_weights'].shape}  (per-channel)")
    print(f"  PA map shape:       {diag['pa_map'].shape}  (spatial)")
    print(f"  CA weights (ch):    {diag['ca_weights'].squeeze().tolist()}")
    print(f"  PA map range:       [{diag['pa_map'].min():.4f}, {diag['pa_map'].max():.4f}]")
    print(f"  Total parameters:   {total_params:,}")
    print(f"    FA block:         {fa_params:,}")
    print(f"      CA:             {ca_params:,}")
    print(f"      PA:             {pa_params:,}")
    print("=" * 60)
    print("  [OK] AOD-CA-PA-Net is working correctly!")
    print("=" * 60)
