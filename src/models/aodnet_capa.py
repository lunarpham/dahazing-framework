"""
AOD-CA-PA-Net v3: AOD-Net + Channel Attention → Pixel Attention Cascade
                  + Reflection Padding + Color Refinement Tail.

Extends v2 with critical improvements for multi-domain training:

    1. Reflection Padding:
        Replaces zero padding on all spatial convolutions. Zero padding
        injects artificial zeros at image boundaries, which create visible
        edge artifacts when large kernels (5×5, 7×7) are used. Reflection
        padding mirrors edge pixels, producing seamless boundary features.

    2. Stacked Feature Attention (N× FA blocks):
        A single FA block cannot learn domain-specific attention patterns
        for multiple haze types (synthetic RESIDE, real O-Haze/NH-Haze,
        nighttime 3R). Stacking multiple FA blocks in sequence gives the
        network depth to specialize: early blocks handle gross haze
        suppression, later blocks refine domain-specific features.

    3. Color Refinement Tail:
        The AOD-Net formula J = K·I − K + 1 is a per-channel affine
        transform: J_c = K_c·(I_c − 1) + 1. This cannot produce cross-
        channel color interactions. The refinement tail applies a learned
        nonlinear residual:

            J_final = J_k + α · Refine(J_k)

        where Refine is a lightweight ConvNet with Tanh output, and α is
        a learnable scaling factor (initialized to 0.1).

Architecture Flow (v3):
    I(x)
      ↓
    [ReflPad+Conv1×1 → ReflPad+Conv3×3 → ReflPad+Conv5×5 → ReflPad+Conv7×7]
      ↓ cat (4×CH channels)
    [ReflPad+Conv3×3 + InstanceNorm] → F_multi  (CH channels)
      ↓
    ┌── FA Block ×N (stacked, each with residual) ─────────┐
    │  CA: GlobalAvgPool → FC → ReLU → FC → Sig           │
    │  PA: Conv1×1 → ReLU → Conv1×1 → Sig                 │
    │  F = F + CA(F) ⊙ PA(CA(F))                          │
    └──────────────────────────────────────────────────────┘
      ↓
    [ReflPad+Conv3×3 → ReLU → ReflPad+Conv3×3] → K(x)
      ↓
    J_k = K(x)·I(x) − K(x) + 1
      ↓
    J_final = J_k + α · Refine(J_k)   ← color refinement tail
      ↓
    clamp [0, 1]

Physics:
    Standard ASM:  J(x) = (I(x) - A) / t(x) + A    ← unstable (divides by t)
    AOD-Net:       J(x) = K(x)·I(x) - K(x) + 1     ← no division, stable
    v3 tail:       J(x) += α · Refine(J)             ← nonlinear color fix

References:
    Li et al. (2017). AOD-Net: All-in-One Dehazing Network. ICCV.
    Qin et al. (2020). FFA-Net: Feature Fusion Attention Network. AAAI.

Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Default backbone width ───────────────────────────────────────────────────

DEFAULT_CHANNELS = 32
DEFAULT_NUM_FA_BLOCKS = 3


# ── Reflection-padded convolution helper ─────────────────────────────────────

class ReflPadConv2d(nn.Module):
    """
    Conv2d with reflection padding instead of zero padding.

    Zero padding injects artificial zeros at image boundaries, producing
    visible edge artifacts especially with large kernels (5×5, 7×7).
    Reflection padding mirrors edge pixels for seamless boundary features.
    """

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0)  # padding handled by ReflectionPad2d

    def forward(self, x):
        return self.conv(self.pad(x))


# ── Channel Attention (Squeeze-and-Excitation style) ─────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) module.

    Performs global context aggregation (squeeze) then learns per-channel
    importance weights (excitation) to suppress channels dominated by
    atmospheric scattering while preserving structurally-informative channels.

    Architecture:
        GlobalAvgPool → FC(C → C//r, min 1) → ReLU → FC(C//r → C) → Sigmoid

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

    Output:  F_out = F_in + CA(F_in) ⊙ PA(CA(F_in))

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
            x: Feature map (B, C, H, W).

        Returns:
            Residual-connected, channel- and pixel-attended features.
        """
        attended = self.ca(x)   # step 1: suppress haze channels
        attended = self.pa(attended)   # step 2: spatially weight remaining features
        return x + attended     # residual: preserves base features


# ── Color Refinement Tail ────────────────────────────────────────────────────

class ColorRefinementTail(nn.Module):
    """
    Nonlinear color refinement applied after the K(x) formula.

    The AOD formula J = K·I − K + 1 is a per-channel affine transform
    that cannot produce cross-channel color interactions. This tail
    learns a residual correction:

        J_final = J_k + α · Refine(J_k)

    Architecture:
        ReflPad+Conv3×3 (3→mid) → ReLU →
        ReflPad+Conv3×3 (mid→mid) → ReLU →
        ReflPad+Conv3×3 (mid→3) → Tanh

    Args:
        mid_channels: Width of the hidden layers.
    """

    def __init__(self, mid_channels: int = 32):
        super().__init__()
        self.refine = nn.Sequential(
            ReflPadConv2d(3, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflPadConv2d(mid_channels, mid_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflPadConv2d(mid_channels, 3, kernel_size=3),
            nn.Tanh(),   # output in [-1, 1] — allows both + and - corrections
        )
        # Learnable scaling factor — starts small so K-formula output dominates
        # early training, then grows as the network learns useful corrections.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, j_k):
        """
        Args:
            j_k: K-formula output J = K·I − K + 1, shape (B, 3, H, W).

        Returns:
            Color-refined output, same shape (B, 3, H, W). NOT clamped here.
        """
        return j_k + self.alpha * self.refine(j_k)


# ── AOD-CA-PA-Net v3 ─────────────────────────────────────────────────────────

class AODCAPANet(nn.Module):
    """
    AOD-Net + Channel Attention → Pixel Attention cascade (AOD-CA-PA-Net v3).

    Key features:
        1. Reflection padding: eliminates edge artifacts from zero padding.
        2. Stacked FA blocks (default 3): multiple attention refinement stages
           for multi-domain haze handling (synthetic + real + nighttime).
        3. Color refinement tail (3→32→32→3): breaks through the K-formula's
           per-channel linear ceiling for vivid color restoration.
        4. InstanceNorm: preserves per-sample color statistics.
        5. Configurable backbone width via `channels` parameter.

    Args:
        channels: Backbone feature width (default 32).
        num_fa_blocks: Number of stacked Feature Attention blocks (default 3).
        refine_channels: Hidden width of the color refinement tail (default 32).
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS,
                 num_fa_blocks: int = DEFAULT_NUM_FA_BLOCKS,
                 refine_channels: int = 32):
        super().__init__()
        ch = channels

        # ── Multi-scale feature extraction (reflection-padded) ───────────────
        self.conv1 = ReflPadConv2d(3, ch, kernel_size=1)
        self.conv2 = ReflPadConv2d(ch, ch, kernel_size=3)
        self.conv3 = ReflPadConv2d(ch * 2, ch, kernel_size=5)    # cat(L1, L2)
        self.conv4 = ReflPadConv2d(ch * 2, ch, kernel_size=7)    # cat(L2, L3)
        self.conv5 = ReflPadConv2d(ch * 4, ch, kernel_size=3)    # cat(L1..L4)

        # ── InstanceNorm after multi-scale fusion ─────────────────────────────
        self.norm5 = nn.InstanceNorm2d(ch, affine=True)

        # ── Stacked Feature Attention blocks ─────────────────────────────────
        # Multiple FA blocks allow the network to progressively refine
        # attention: early blocks handle gross haze suppression, later
        # blocks refine domain-specific features.
        self.fa_blocks = nn.Sequential(*[
            FeatureAttention(channels=ch) for _ in range(num_fa_blocks)
        ])

        # ── K(x) estimation (deeper 2-layer head, reflection-padded) ─────────
        self.k_estimator = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3),
            nn.ReLU(inplace=True),
            ReflPadConv2d(ch, 3, kernel_size=3),
        )

        # ── Color Refinement Tail ────────────────────────────────────────────
        self.color_refine = ColorRefinementTail(mid_channels=refine_channels)

        self.relu = nn.ReLU(inplace=True)

        # ── Initialization ──────────────────────────────────────────────────
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass: backbone → IN → stacked FA → K(x) → AOD → color refine.

        Args:
            x: Hazy input image I(x), shape (B, 3, H, W) in [0, 1].

        Returns:
            Clean image, clamped to [0, 1].
        """
        # ── Step 1: Multi-Scale Feature Extraction ───────────────────────────
        x1 = self.relu(self.conv1(x))                        # (B, CH, H, W)
        x2 = self.relu(self.conv2(x1))                       # (B, CH, H, W)

        cat1 = torch.cat((x1, x2), dim=1)                   # (B, 2*CH, H, W)
        x3 = self.relu(self.conv3(cat1))                     # (B, CH, H, W)

        cat2 = torch.cat((x2, x3), dim=1)                   # (B, 2*CH, H, W)
        x4 = self.relu(self.conv4(cat2))                     # (B, CH, H, W)

        cat3 = torch.cat((x1, x2, x3, x4), dim=1)           # (B, 4*CH, H, W)
        f_multi = self.relu(self.norm5(self.conv5(cat3)))    # (B, CH, H, W)

        # ── Step 2: Stacked Feature Attention ────────────────────────────────
        f_attended = self.fa_blocks(f_multi)                 # (B, CH, H, W)

        # ── Step 3: K(x) Estimation ──────────────────────────────────────────
        k = self.relu(self.k_estimator(f_attended))          # (B, 3, H, W)

        # ── Step 4: AOD Physics Formula ──────────────────────────────────────
        j_k = k * x - k + 1.0

        # ── Step 5: Color Refinement ─────────────────────────────────────────
        clean = self.color_refine(j_k)

        return torch.clamp(clean, 0.0, 1.0)

    def forward_with_maps(self, x):
        """
        Diagnostic forward pass returning K(x), CA weights, PA map from
        the last FA block, and the pre-refinement K-formula output.

        Returns:
            dict with keys:
                'output':       Final dehazed image (B, 3, H, W)
                'j_k':          Pre-refinement K-formula output (B, 3, H, W)
                'k_map':        Predicted K(x) parameter (B, 3, H, W)
                'ca_weights':   Per-channel CA weights from last FA (B, CH)
                'pa_map':       Spatial PA map from last FA (B, 1, H, W)
                'refine_alpha': Color refinement scaling factor (scalar)
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(torch.cat((x1, x2), dim=1)))
        x4 = self.relu(self.conv4(torch.cat((x2, x3), dim=1)))
        f_multi = self.relu(self.norm5(self.conv5(
            torch.cat((x1, x2, x3, x4), dim=1)
        )))

        # Run through stacked FA blocks, extract maps from the last one
        f = f_multi
        for fa_block in self.fa_blocks:
            f = fa_block(f)

        # Extract diagnostics from the last FA block
        last_fa = self.fa_blocks[-1]
        ca_weights_flat = last_fa.ca.ca(f)
        ca_weights = ca_weights_flat.unsqueeze(-1).unsqueeze(-1)
        f_ca = f * ca_weights
        pa_map = last_fa.pa.pa(f_ca)

        k = self.relu(self.k_estimator(f))
        j_k = k * x - k + 1.0
        clean = torch.clamp(self.color_refine(j_k), 0.0, 1.0)

        return {
            'output':       clean,
            'j_k':          torch.clamp(j_k, 0.0, 1.0),
            'k_map':        k,
            'ca_weights':   ca_weights_flat,
            'pa_map':       pa_map,
            'refine_alpha': self.color_refine.alpha.item(),
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
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy = torch.rand(1, 3, 256, 256)
    model = AODCAPANet(channels=96, num_fa_blocks=3, refine_channels=32)

    total_params  = sum(p.numel() for p in model.parameters())
    backbone_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith('conv') or name.startswith('norm')
    )
    fa_params     = sum(p.numel() for p in model.fa_blocks.parameters())
    k_params      = sum(p.numel() for p in model.k_estimator.parameters())
    refine_params = sum(p.numel() for p in model.color_refine.parameters())

    output = model(dummy)
    diag   = model.forward_with_maps(dummy)

    print("=" * 60)
    print("  AOD-CA-PA-Net v3: Stacked FA + Color Refinement")
    print("=" * 60)
    print(f"  Input shape:        {dummy.shape}")
    print(f"  Output shape:       {output.shape}")
    print(f"  K(x) map shape:     {diag['k_map'].shape}")
    print(f"  J_k (pre-refine):   [{diag['j_k'].min():.4f}, {diag['j_k'].max():.4f}]")
    print(f"  CA weights shape:   {diag['ca_weights'].shape}  (per-channel)")
    print(f"  PA map shape:       {diag['pa_map'].shape}  (spatial)")
    print(f"  PA map range:       [{diag['pa_map'].min():.4f}, {diag['pa_map'].max():.4f}]")
    print(f"  Refine alpha:       {diag['refine_alpha']:.4f}")
    print(f"  Total parameters:   {total_params:,}")
    print(f"    Backbone + Norm:  {backbone_params:,}")
    print(f"    FA blocks (×3):   {fa_params:,}")
    print(f"    K head:           {k_params:,}")
    print(f"    Color refine:     {refine_params:,}")
    print("=" * 60)
    print("  [OK] AOD-CA-PA-Net v3 is working correctly!")
    print("=" * 60)
