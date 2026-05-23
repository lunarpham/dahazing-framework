"""
MSFA-Net: Multi-Scale Feature Attention Network for Image Dehazing.

Combines MSBDN's multi-scale encoder-decoder backbone with FFA-Net's
Feature Attention (Channel Attention → Pixel Attention) mechanism.

Key innovations over AOD-CA-PA-Net:

    1. Multi-Scale Encoder-Decoder (from MSBDN):
        Processes features at 3 scales (full, ½, ¼). The encoder
        downsamples to capture global haze density context; the decoder
        upsamples with skip connections to recover local detail. This
        eliminates the "static K" problem where single-scale models
        apply near-constant corrections regardless of haze density.

    2. Dense Blocks (from MSBDN/DenseNet):
        Each block has 3 conv layers with dense connections — every
        layer receives all previous outputs concatenated. This maximizes
        feature reuse and gradient flow without vanishing gradients.

    3. Feature Attention (from FFA-Net):
        CA → PA cascade at every encoder/decoder level. Channel Attention
        suppresses haze-dominated channels; Pixel Attention spatially
        weights features based on haze density.

    4. Global Residual Learning:
        Predicts the residual (haze component) rather than the clean
        image directly. clean = hazy + residual. More efficient because
        most pixels change little between hazy and clean images.

Architecture:
    Input I(x)  ────────────────────────────────────┐ (global residual)
      ↓ [ReflPadConv 3→CH]                         │
    ┌── Encoder ──────────────────────────────┐     │
    │ Scale 0: DenseBlock + IN + FA×2  ─skip─┐│     │
    │   ↓ stride-2 conv                      ││     │
    │ Scale 1: DenseBlock + IN + FA×2  ─skip┐││     │
    │   ↓ stride-2 conv                     │││     │
    │ Bottleneck: DenseBlock + IN + FA×2    │││     │
    └───────────────────────────────────────┘│││     │
      ↓ upsample + conv                     │││     │
    ┌── Decoder ────────────────────────────┐│││     │
    │ Scale 1: cat(skip1) → Dense+IN+FA×2 ←┘││     │
    │   ↑ upsample + conv                   ││     │
    │ Scale 0: cat(skip0) → Dense+IN+FA×2 ←─┘│     │
    └──────────────────────────────────────────┘     │
      ↓ [ReflPadConv CH→3]                          │
    clean = I(x) + residual  ←───────────────────────┘
    clamp [0, 1]

References:
    Dong et al. (2020). MSBDN: Multi-Scale Boosted Dehazing Network. CVPR.
    Qin et al. (2020). FFA-Net: Feature Fusion Attention Network. AAAI.

Output: (B, 3, H, W) dehazed image in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHANNELS = 64
DEFAULT_NUM_FA = 2
DEFAULT_NUM_DENSE_LAYERS = 3
DEFAULT_NUM_GROUPS = 8  # for GroupNorm (channels must be divisible by this)


# ── Reflection-padded convolution ────────────────────────────────────────────

class ReflPadConv2d(nn.Module):
    """Conv2d with reflection padding. Supports stride for downsampling."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=0)

    def forward(self, x):
        return self.conv(self.pad(x))


# ── Channel Attention (Squeeze-and-Excitation) ───────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention via global average pooling + FC bottleneck.
    Learns per-channel importance to suppress haze-dominated channels.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.ca(x).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


# ── Pixel Attention ──────────────────────────────────────────────────────────

class PixelAttention(nn.Module):
    """
    Pixel Attention via 1×1 conv bottleneck.
    Learns spatial haze density map to weight features per-pixel.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(1, channels // 8)
        self.pa = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.pa(x)  # (B, 1, H, W) broadcast


# ── Feature Attention: CA → PA with Residual ─────────────────────────────────

class FeatureAttention(nn.Module):
    """
    CA → PA cascade with residual: F_out = F_in + PA(CA(F_in))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=8)
        self.pa = PixelAttention(channels)

    def forward(self, x):
        return x + self.pa(self.ca(x))


# ── Dense Block ──────────────────────────────────────────────────────────────

class DenseBlock(nn.Module):
    """
    Dense block with N layers. Each layer receives all previous outputs
    concatenated, maximizing feature reuse (DenseNet-style).

    GroupNorm is applied after each conv to prevent feature explosion
    from the growing concatenated input (CH → 2·CH → 3·CH).

    Architecture (3 layers):
        c1 = ReLU(GN(Conv(x)))                    in: CH      out: CH
        c2 = ReLU(GN(Conv(cat(x, c1))))           in: 2·CH    out: CH
        c3 = ReLU(GN(Conv(cat(x, c1, c2))))       in: 3·CH    out: CH
        output = x + c3                           residual connection

    Args:
        channels: Feature width.
        num_layers: Number of dense layers (default 3).
        num_groups: Groups for GroupNorm (default 8).
    """

    def __init__(self, channels: int, num_layers: int = 3,
                 num_groups: int = DEFAULT_NUM_GROUPS):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = channels * (i + 1)  # cat of all previous outputs
            self.layers.append(nn.Sequential(
                ReflPadConv2d(in_ch, channels, kernel_size=3),
                nn.GroupNorm(num_groups, channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            cat_input = torch.cat(features, dim=1)
            out = layer(cat_input)
            features.append(out)
        # Residual: add last layer output to input
        return x + features[-1]


# ── Encoder Block ────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """DenseBlock → GroupNorm → FA×N at one scale level."""

    def __init__(self, channels: int, num_fa: int = 2,
                 num_dense_layers: int = 3,
                 num_groups: int = DEFAULT_NUM_GROUPS):
        super().__init__()
        self.dense = DenseBlock(channels, num_layers=num_dense_layers,
                                num_groups=num_groups)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.fa_blocks = nn.Sequential(*[
            FeatureAttention(channels) for _ in range(num_fa)
        ])

    def forward(self, x):
        out = self.dense(x)
        out = self.norm(out)
        out = self.fa_blocks(out)
        return out


# ── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Skip fusion (cat + 1×1 conv) → DenseBlock → GroupNorm → FA×N."""

    def __init__(self, channels: int, num_fa: int = 2,
                 num_dense_layers: int = 3,
                 num_groups: int = DEFAULT_NUM_GROUPS):
        super().__init__()
        # Fuse skip (2·CH → CH) with 1×1 conv
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups, channels),
            nn.ReLU(inplace=True),
        )
        self.dense = DenseBlock(channels, num_layers=num_dense_layers,
                                num_groups=num_groups)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.fa_blocks = nn.Sequential(*[
            FeatureAttention(channels) for _ in range(num_fa)
        ])

    def forward(self, x, skip):
        # Match spatial dimensions (handles odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        fused = self.skip_fuse(torch.cat([x, skip], dim=1))
        out = self.dense(fused)
        out = self.norm(out)
        out = self.fa_blocks(out)
        return out


# ── MSFA-Net ─────────────────────────────────────────────────────────────────

class MSFANet(nn.Module):
    """
    Multi-Scale Feature Attention Network for Image Dehazing.

    Combines MSBDN's multi-scale encoder-decoder with FFA-Net's Feature
    Attention (CA→PA) mechanism. Uses global residual learning.

    Args:
        channels: Backbone feature width (default 64).
        num_fa: Number of FA blocks per encoder/decoder level (default 2).
        num_dense_layers: Conv layers per DenseBlock (default 3).
    """

    def __init__(self, channels: int = DEFAULT_CHANNELS,
                 num_fa: int = DEFAULT_NUM_FA,
                 num_dense_layers: int = DEFAULT_NUM_DENSE_LAYERS):
        super().__init__()
        ch = channels

        # ── Input Projection ─────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            ReflPadConv2d(3, ch, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        # ── Encoder (3 levels: full → ½ → ¼) ────────────────────────────────
        self.enc0 = EncoderBlock(ch, num_fa, num_dense_layers)  # full res
        self.down0 = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.enc1 = EncoderBlock(ch, num_fa, num_dense_layers)  # ½ res
        self.down1 = nn.Sequential(
            ReflPadConv2d(ch, ch, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # ── Bottleneck (¼ res) ───────────────────────────────────────────────
        self.bottleneck = EncoderBlock(ch, num_fa, num_dense_layers)

        # ── Decoder (¼ → ½ → full) ──────────────────────────────────────────
        self.up1 = ReflPadConv2d(ch, ch, kernel_size=3)  # post-upsample refine
        self.dec1 = DecoderBlock(ch, num_fa, num_dense_layers)  # ½ res

        self.up0 = ReflPadConv2d(ch, ch, kernel_size=3)  # post-upsample refine
        self.dec0 = DecoderBlock(ch, num_fa, num_dense_layers)  # full res

        # ── Output Projection ────────────────────────────────────────────────
        self.output_proj = ReflPadConv2d(ch, 3, kernel_size=3)

        # Learnable residual scale — starts small so model begins near
        # identity (clean ≈ hazy), then grows as training progresses.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # ── Weight Initialization ────────────────────────────────────────────
        self._initialize_weights()

        # Zero-initialize output projection so residual starts at ~0
        # (EDSR trick: prevents random noise output at epoch 0)
        nn.init.zeros_(self.output_proj.conv.weight)
        nn.init.zeros_(self.output_proj.conv.bias)

    def forward(self, x):
        """
        Forward pass with global residual learning.

        Args:
            x: Hazy input image I(x), shape (B, 3, H, W) in [0, 1].

        Returns:
            Dehazed image, clamped to [0, 1].
        """
        # ── Input Projection ─────────────────────────────────────────────────
        f0 = self.input_proj(x)                           # (B, CH, H, W)

        # ── Encoder ──────────────────────────────────────────────────────────
        e0 = self.enc0(f0)                                # (B, CH, H, W)
        f1 = self.down0(e0)                               # (B, CH, H/2, W/2)

        e1 = self.enc1(f1)                                # (B, CH, H/2, W/2)
        f2 = self.down1(e1)                               # (B, CH, H/4, W/4)

        # ── Bottleneck ───────────────────────────────────────────────────────
        b = self.bottleneck(f2)                           # (B, CH, H/4, W/4)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Scale 1: upsample bottleneck → fuse with enc1 skip
        u1 = F.interpolate(b, scale_factor=2, mode='bilinear',
                           align_corners=False)
        u1 = F.relu(self.up1(u1), inplace=True)
        d1 = self.dec1(u1, e1)                            # (B, CH, H/2, W/2)

        # Scale 0: upsample dec1 → fuse with enc0 skip
        u0 = F.interpolate(d1, scale_factor=2, mode='bilinear',
                           align_corners=False)
        u0 = F.relu(self.up0(u0), inplace=True)
        d0 = self.dec0(u0, e0)                            # (B, CH, H, W)

        # ── Output: Global Residual ──────────────────────────────────────────
        # residual_scale starts at 0.1, keeping initial output near identity.
        # Zero-initialized output_proj ensures residual ≈ 0 at epoch 0.
        residual = self.output_proj(d0)                   # (B, 3, H, W)
        clean = x + self.residual_scale * residual        # gentle correction

        return torch.clamp(clean, 0.0, 1.0)

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

    for ch, label in [(32, "Small"), (64, "Medium"), (96, "Large")]:
        model = MSFANet(channels=ch, num_fa=2, num_dense_layers=3)
        total = sum(p.numel() for p in model.parameters())

        # Component breakdown
        enc_params = sum(
            sum(p.numel() for p in block.parameters())
            for block in [model.enc0, model.enc1, model.bottleneck]
        )
        dec_params = sum(
            sum(p.numel() for p in block.parameters())
            for block in [model.dec0, model.dec1]
        )
        down_params = sum(
            sum(p.numel() for p in block.parameters())
            for block in [model.down0, model.down1]
        )
        up_params = sum(
            sum(p.numel() for p in block.parameters())
            for block in [model.up0, model.up1]
        )
        proj_params = (
            sum(p.numel() for p in model.input_proj.parameters()) +
            sum(p.numel() for p in model.output_proj.parameters())
        )

        output = model(dummy)

        print(f"\n{'='*60}")
        print(f"  MSFA-Net ({label}): channels={ch}")
        print(f"{'='*60}")
        print(f"  Input:           {dummy.shape}")
        print(f"  Output:          {output.shape}")
        print(f"  Range:           [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Total params:    {total:,}")
        print(f"    Encoder (×3):  {enc_params:,}")
        print(f"    Decoder (×2):  {dec_params:,}")
        print(f"    Down (×2):     {down_params:,}")
        print(f"    Up (×2):       {up_params:,}")
        print(f"    Projections:   {proj_params:,}")

    print(f"\n{'='*60}")
    print(f"  [OK] MSFA-Net is working correctly!")
    print(f"{'='*60}")
