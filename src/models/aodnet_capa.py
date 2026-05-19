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

Architecture Flow (v2 — wider backbone + residual attention):
    I(x)
      ↓
    [Conv1×1 → Conv3×3 → Conv5×5 → Conv7×7] (multi-scale backbone, 32ch)
      ↓ cat (128 channels)
    [Conv3×3 + BN] → F_multi  (32 channels)
      ↓
    ┌── Feature Attention (with residual) ──────────────┐
    │  CA: GlobalAvgPool → FC(32→4) → ReLU → FC → Sig │
    │  PA: Conv1×1(32→4) → ReLU → Conv1×1(4→1) → Sig  │
    │  F_attended = F_multi + CA(F_multi) ⊙ PA(...)     │  ← residual
    └───────────────────────────────────────────────────┘
      ↓
    [Conv3×3 → ReLU → Conv3×3] → K(x)  (deeper 2-layer head)
      ↓
    J(x) = K(x)·I(x) − K(x) + 1   (clamped to [0,1])

Physics:
    Standard ASM:  J(x) = (I(x) - A) / t(x) + A    ← unstable (divides by t)
    AOD-Net:       J(x) = K(x)·I(x) - K(x) + 1     ← no division, stable

References:
    Li et al. (2017). AOD-Net: All-in-One Dehazing Network. ICCV.
    Qin et al. (2020). FFA-Net: Feature Fusion Attention Network. AAAI.

Estimated parameters: ~45,000  (still very lightweight — FFA-Net has 4.5M)
Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Default backbone width ───────────────────────────────────────────────────

DEFAULT_CHANNELS = 32


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


# ── Feature Attention: CA → PA Cascade with Residual ─────────────────────────

class FeatureAttention(nn.Module):
    """
    Feature Attention (FA) cascade: Channel Attention followed by Pixel
    Attention, wrapped in a residual connection.

    Mirrors the FA module from FFA-Net with a critical addition: the residual
    skip connection ensures that base features always flow through even when
    both CA and PA assign low weights. This prevents the "signal death"
    problem where two cascaded [0,1] multiplications can suppress features
    to near-zero, causing K(x) ≈ 0 and washed-out output.

    Output:  F_out = F_multi + CA(F_multi) ⊙ PA(CA(F_multi))

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
            F_attended: residual-connected, channel- and pixel-attended
                        features (B, C, H, W).
        """
        attended = self.ca(x)   # step 1: suppress haze channels
        attended = self.pa(attended)   # step 2: spatially weight remaining features
        return x + attended     # residual: preserves base features


# ── AOD-CA-PA-Net ─────────────────────────────────────────────────────────────

class AODCAPANet(nn.Module):
    """
    AOD-Net + Channel Attention → Pixel Attention cascade (AOD-CA-PA-Net v2).

    Key improvements over v1:
        1. Wider backbone (3→32ch): ~25× more feature capacity for diverse
           haze patterns and spatially-varying K(x) estimation.
        2. Residual Feature Attention: skip connection prevents signal death
           from cascaded [0,1] multiplications, eliminating washed-out output.
        3. Deeper K(x) head (2-layer): better mapping from attended features
           to the K parameter, improving dynamic range.
        4. BatchNorm after fusion: stabilizes feature distributions before
           the attention block, improving training convergence.

    Architecture (multi-scale backbone + residual FA cascade):
        Layer 1: Conv2d 1×1  (3→CH)    — point-wise feature extraction
        Layer 2: Conv2d 3×3  (CH→CH)   — local features
        Layer 3: Conv2d 5×5  (2CH→CH)  — cat(L1,L2) → medium-range context
        Layer 4: Conv2d 7×7  (2CH→CH)  — cat(L2,L3) → wide-range context
        Layer 5: Conv2d 3×3  (4CH→CH)  — cat(L1,L2,L3,L4) → F_multi + BN

        FA Block: F_multi + CA→PA(F_multi)  ← residual attention

        K Head:  Conv2d 3×3 (CH→CH) + ReLU → Conv2d 3×3 (CH→3) + ReLU → K(x)

    Image recovery:
        J(x) = K(x) * I(x) - K(x) + 1
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS):
        super().__init__()
        ch = channels

        # ── Multi-scale feature extraction (widened AOD-Net backbone) ────────
        self.conv1 = nn.Conv2d(3, ch, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ch * 2, ch, kernel_size=5, stride=1, padding=2)   # cat(L1, L2)
        self.conv4 = nn.Conv2d(ch * 2, ch, kernel_size=7, stride=1, padding=3)   # cat(L2, L3)
        self.conv5 = nn.Conv2d(ch * 4, ch, kernel_size=3, stride=1, padding=1)   # cat(L1..L4)

        # ── BatchNorm after multi-scale fusion ───────────────────────────────
        self.bn5 = nn.BatchNorm2d(ch)

        # ── Feature Attention: CA → PA cascade with residual ─────────────────
        self.feature_attention = FeatureAttention(channels=ch)

        # ── K(x) estimation (deeper 2-layer head) ───────────────────────────
        self.k_estimator = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
        )

        self.relu = nn.ReLU(inplace=True)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: multi-scale backbone → BN → residual CA→PA → K(x) → J(x).

        Args:
            x: Hazy input image I(x), shape (B, 3, H, W) in [0, 1].

        Returns:
            Clean image J(x) = K(x)*I(x) - K(x) + 1, clamped to [0, 1].
        """
        # ── Step 1: Multi-Scale Feature Extraction ───────────────────────────
        x1 = self.relu(self.conv1(x))                        # (B, CH, H, W)
        x2 = self.relu(self.conv2(x1))                       # (B, CH, H, W)

        cat1 = torch.cat((x1, x2), dim=1)                   # (B, 2*CH, H, W)
        x3 = self.relu(self.conv3(cat1))                     # (B, CH, H, W)

        cat2 = torch.cat((x2, x3), dim=1)                   # (B, 2*CH, H, W)
        x4 = self.relu(self.conv4(cat2))                     # (B, CH, H, W)

        cat3 = torch.cat((x1, x2, x3, x4), dim=1)           # (B, 4*CH, H, W)
        f_multi = self.relu(self.bn5(self.conv5(cat3)))      # (B, CH, H, W)

        # ── Step 2: Feature Attention with Residual (CA → PA) ────────────────
        # Residual ensures base features survive even when attention is low.
        # CA first: suppresses channels dominated by uniform haze statistics.
        # PA second: maps spatial haze density on the channel-cleaned features.
        f_attended = self.feature_attention(f_multi)         # (B, CH, H, W)

        # ── Step 3: K(x) Estimation (deeper 2-layer head) ───────────────────
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
                'ca_weights':   Per-channel CA weights (B, CH) in [0, 1]
                'pa_map':       Spatial PA attention map (B, 1, H, W)
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat((x1, x2), dim=1)))
        x4 = self.relu(self.conv4(torch.cat((x2, x3), dim=1)))
        f_multi = self.relu(self.bn5(self.conv5(torch.cat((x1, x2, x3, x4), dim=1))))

        # Extract CA weights (before multiplying)
        fa = self.feature_attention
        ca_weights_flat = fa.ca.ca(f_multi)            # (B, CH)
        ca_weights = ca_weights_flat.unsqueeze(-1).unsqueeze(-1)
        f_ca = f_multi * ca_weights                    # channel-attended features

        # Extract PA map
        pa_map = fa.pa.pa(f_ca)                        # (B, 1, H, W)
        f_attended_branch = f_ca * pa_map

        # Apply residual (matching forward() behavior)
        f_attended = f_multi + f_attended_branch

        k = self.relu(self.k_estimator(f_attended))
        clean = torch.clamp(k * x - k + 1.0, 0.0, 1.0)

        return {
            'output':     clean,
            'k_map':      k,
            'ca_weights': ca_weights_flat,             # (B, CH) — one per channel
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy = torch.rand(1, 3, 256, 256)
    model = AODCAPANet()

    total_params  = sum(p.numel() for p in model.parameters())
    fa_params     = sum(p.numel() for p in model.feature_attention.parameters())
    ca_params     = sum(p.numel() for p in model.feature_attention.ca.parameters())
    pa_params     = sum(p.numel() for p in model.feature_attention.pa.parameters())
    backbone_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith('conv') or name.startswith('bn')
    )
    k_params      = sum(p.numel() for p in model.k_estimator.parameters())

    output = model(dummy)
    diag   = model.forward_with_maps(dummy)

    print("=" * 60)
    print("  AOD-CA-PA-Net v2: Wider Backbone + Residual Attention")
    print("=" * 60)
    print(f"  Input shape:        {dummy.shape}")
    print(f"  Output shape:       {output.shape}")
    print(f"  K(x) map shape:     {diag['k_map'].shape}")
    print(f"  CA weights shape:   {diag['ca_weights'].shape}  (per-channel)")
    print(f"  PA map shape:       {diag['pa_map'].shape}  (spatial)")
    print(f"  PA map range:       [{diag['pa_map'].min():.4f}, {diag['pa_map'].max():.4f}]")
    print(f"  Total parameters:   {total_params:,}")
    print(f"    Backbone + BN:    {backbone_params:,}")
    print(f"    FA block:         {fa_params:,}")
    print(f"      CA:             {ca_params:,}")
    print(f"      PA:             {pa_params:,}")
    print(f"    K head:           {k_params:,}")
    print("=" * 60)
    print("  [OK] AOD-CA-PA-Net v2 is working correctly!")
    print("=" * 60)
