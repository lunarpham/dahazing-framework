"""
DCP-Net: Learnable Hybrid Dehazer with Guided Filter Refinement.

A lightweight physics-guided dehazing network combining the Dark Channel
Prior with three learned sub-networks and a differentiable Guided Filter:

1. **Transmission Sub-Network**: Predicts pixel-wise transmission t(x)
   from RGB + DCP features. Output is coarse (blocky).

2. **Guided Image Filter**: Mathematically aligns the coarse t(x) with
   the structural edges of the original image. Unlike feeding Sobel edges
   as CNN input (which the network can ignore), GIF enforces edge alignment
   as a hard constraint → guaranteed halo prevention.

3. **Atmospheric Light Sub-Network**: Learns global A from RGB + DCP via
   CNN + global pooling + FC. Avoids white-object confusion.

4. **Sky Detection Branch**: Learns sky segmentation mask → enforces higher
   transmission floor in sky regions → prevents over-saturation.

Architecture:
    Input I(x) [B, 3, H, W]
        │
        ├── DCP (non-learned) ──→ dark_ch [B, 1, H, W]
        │
        ├── Sky Branch ─────────────────────────────────
        │   cat(I, dark_ch) [B,4] → Conv→ReLU→Conv→σ
        │   → sky_mask ∈ [0, 1]
        │
        ├── Transmission Branch ────────────────────────
        │   cat(I, dark_ch) [B,4] → Conv→ReLU→Conv→ReLU→Conv→σ
        │   → t_raw (coarse)
        │   → Guided Filter(guidance=gray(I), input=t_raw)
        │   → t_refined (edge-aligned, halo-free)
        │   → sky correction: max(t, 0.4) in sky regions
        │   → t_final = clamp(t, min=0.1)
        │
        ├── Atmospheric Light Branch ───────────────────
        │   cat(I, dark_ch) [B,4] → Conv↓2→Conv↓2→GAP→FC→FC→σ
        │   → A ∈ [0, 1]^3 (per-channel global)
        │
        └── Physics Recovery ───────────────────────────
            J(x) = (I(x) - A) / t_final + A
            → clamp [0, 1]

Key Advantages:
    - Guided Filter gives mathematical halo prevention (not just a learned hint)
    - Learned A avoids white-object confusion
    - Sky mask prevents over-saturation
    - DCP as physics-prior feature for transmission CNN
    - Lightweight (~9.5K learnable parameters)

References:
    He, K., Sun, J., & Tang, X. (2010). Single Image Haze Removal Using
    Dark Channel Prior. IEEE TPAMI, 33(12), 2341-2353.

    He, K., Sun, J., & Tang, X. (2013). Guided Image Filtering.
    IEEE TPAMI, 35(6), 1397-1409.

Output: (B, 3, H, W) dehazed image in [0, 1].
"""

from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCPNet(nn.Module):
    """
    Learnable Hybrid Dehazer with Guided Filter Refinement.

    Combines DCP physics with learned transmission, atmospheric light, and
    sky detection sub-networks. Uses a differentiable Guided Image Filter
    to enforce edge-aligned transmission (halo-free).

    Args:
        patch_size: Dark channel local minimum window (default 15).
        refine_channels: Width of internal conv layers (default 16).
        gif_radius: Guided filter spatial radius (default 15).
        gif_eps: Guided filter regularization (default 1e-3).
        t_min_sky: Transmission floor inside sky regions (default 0.4).
        t_min_global: Global transmission floor (default 0.1).
    """

    def __init__(self, patch_size: int = 7, refine_channels: int = 64,
                 gif_radius: int = 8, gif_eps: float = 1e-2,
                 t_min_sky: float = 0.4, t_min_global: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.gif_radius = gif_radius
        self.gif_eps = gif_eps
        self.t_min_sky = t_min_sky
        self.t_min_global = t_min_global
        rc = refine_channels

        # ── 1. Transmission Sub-Network (learned) ────────────────────────────
        # Input: RGB(3) + DCP(1) = 4 channels
        # Output: coarse transmission t_raw, then refined by Guided Filter
        self.t_conv1 = nn.Conv2d(4, rc, kernel_size=3, padding=1, dilation=1)
        self.t_conv2 = nn.Conv2d(rc, rc, kernel_size=3, padding=2, dilation=2)
        self.t_conv3 = nn.Conv2d(rc, rc, kernel_size=3, padding=4, dilation=4)
        self.t_conv4 = nn.Conv2d(rc, 1, kernel_size=3, padding=1, dilation=1)

        # ── 2. Atmospheric Light Sub-Network (learned) ───────────────────────
        # Input: RGB(3) + DCP(1) = 4 channels → downsample → global pool → FC
        self.a_conv1 = nn.Conv2d(4, rc, kernel_size=3, stride=2, padding=1)
        self.a_conv2 = nn.Conv2d(rc, rc * 2, kernel_size=3, stride=2, padding=1)
        self.a_pool = nn.AdaptiveAvgPool2d(1)
        self.a_fc1 = nn.Linear(rc * 2, rc)
        self.a_fc2 = nn.Linear(rc, 3)

        # ── 3. Sky Detection Branch (learned) ────────────────────────────────
        # Input: RGB(3) + DCP(1) = 4 channels
        self.sky_conv1 = nn.Conv2d(4, rc, kernel_size=3, padding=1)
        self.sky_conv2 = nn.Conv2d(rc, 1, kernel_size=3, padding=1)

        # ── Initialization ───────────────────────────────────────────────────
        self._initialize_weights()

    # ── Physics Feature Extraction ───────────────────────────────────────────

    def _compute_dark_channel(self, img):
        """
        Dark channel: min across RGB, then min across local patch.

        Args:
            img: (B, 3, H, W) in [0, 1].
        Returns:
            dark_ch: (B, 1, H, W).
        """
        min_rgb, _ = torch.min(img, dim=1, keepdim=True)
        pad = self.patch_size // 2
        dark_ch = -F.max_pool2d(
            -min_rgb,
            kernel_size=self.patch_size,
            stride=1,
            padding=pad,
        )
        return dark_ch

    # ── Guided Image Filter ──────────────────────────────────────────────────

    def guided_filter(self, I, p):
        """
        Differentiable Guided Image Filter for halo-free transmission.

        Mathematically forces the coarse transmission map 'p' to align
        with the structural edges of guidance image 'I'. Unlike Sobel
        edge input (which the CNN can ignore), GIF is a hard structural
        constraint — it physically prevents halos.

        Fully differentiable: gradients flow through during backprop.

        Args:
            I: Guidance image — grayscale of original (B, 1, H, W).
            p: Input to filter — raw transmission map (B, 1, H, W).
        Returns:
            q: Refined transmission, edge-aligned (B, 1, H, W).
        """
        ks = self.gif_radius * 2 + 1
        pad = self.gif_radius

        def box_filter(x):
            return F.avg_pool2d(x, kernel_size=ks, stride=1, padding=pad, count_include_pad=False)

        mean_I = box_filter(I)
        mean_p = box_filter(p)
        mean_Ip = box_filter(I * p)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = box_filter(I * I)
        var_I = mean_II - mean_I * mean_I

        # Higher gif_eps prevents division-by-zero artifacts in flat areas (like the road)
        a = cov_Ip / (var_I + self.gif_eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a)
        mean_b = box_filter(b)

        return mean_a * I + mean_b

    # ── Forward Pipeline ─────────────────────────────────────────────────────

    def forward(self, x):
        dark_ch = self._compute_dark_channel(x)
        base_features = torch.cat([x, dark_ch], dim=1)

        # Sky Branch
        sky_feat = F.relu(self.sky_conv1(base_features))
        sky_mask = torch.sigmoid(self.sky_conv2(sky_feat))

        # Transmission Branch (Now using 4 layers with dilations)
        t = F.relu(self.t_conv1(base_features))
        t = F.relu(self.t_conv2(t))
        t = F.relu(self.t_conv3(t))
        t_raw = torch.sigmoid(self.t_conv4(t))

        # True Luminance is much better for edge guidance than a simple mean
        # Weights: 0.299 Red, 0.587 Green, 0.114 Blue
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        t_refined = self.guided_filter(gray, t_raw)

        # Sky Adaptive Correction
        t_min_sky = torch.tensor(self.t_min_sky, device=x.device)
        t_corrected = t_refined * (1 - sky_mask) + torch.max(t_refined, t_min_sky) * sky_mask
        t_final = torch.clamp(t_corrected, min=self.t_min_global)

        # Atmospheric Light Branch
        a_feat = F.relu(self.a_conv1(base_features))
        a_feat = F.relu(self.a_conv2(a_feat))
        a_feat = self.a_pool(a_feat).view(x.size(0), -1)
        a_feat = F.relu(self.a_fc1(a_feat))
        A = torch.sigmoid(self.a_fc2(a_feat)).view(-1, 3, 1, 1)

        # Physics Recovery
        clean = (x - A) / t_final + A

        # Store for losses
        self._t_map = t_final
        self._sky_map = sky_mask
        self._gray_img = gray

        return torch.clamp(clean, 0.0, 1.0)

    # ── Auxiliary Loss for Sky Supervision ───────────────────────────────────

    def compute_aux_loss(self, clear_img):
        brightness = clear_img.mean(dim=1, keepdim=True)
        max_rgb, _ = clear_img.max(dim=1, keepdim=True)
        min_rgb, _ = clear_img.min(dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)

        sky_target = ((brightness > 0.6) & (saturation < 0.25)).float()
        return F.binary_cross_entropy(self._sky_map, sky_target)

    def compute_edge_smoothness_loss(self):
        t = self._t_map
        img = self._gray_img

        dt_dx = torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])
        dt_dy = torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])

        dI_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        dI_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        weight_x = torch.exp(-10.0 * dI_dx)
        weight_y = torch.exp(-10.0 * dI_dy)

        loss_x = (dt_dx * weight_x).mean()
        loss_y = (dt_dy * weight_y).mean()

        return loss_x + loss_y 

    # ── Weight Initialization ────────────────────────────────────────────────

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Ensure the new t_conv4 is initialized correctly for Sigmoid
                if name in ['t_conv4', 'sky_conv2', 'a_fc2']:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DCPNet()
    total = sum(p.numel() for p in model.parameters())
    learned = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t_p = sum(p.numel() for n, p in model.named_parameters() if n.startswith('t_'))
    a_p = sum(p.numel() for n, p in model.named_parameters() if n.startswith('a_'))
    s_p = sum(p.numel() for n, p in model.named_parameters() if n.startswith('sky_'))

    dummy_hazy = torch.rand(2, 3, 256, 256)
    dummy_clear = torch.rand(2, 3, 256, 256)

    model.train()
    out = model(dummy_hazy)
    aux = model.compute_aux_loss(dummy_clear)

    print(f"DCP-Net (Hybrid + Guided Filter)")
    print(f"  Learned params:   {learned:,}")
    print(f"    Transmission:   {t_p:,}")
    print(f"    Atm. Light:     {a_p:,}")
    print(f"    Sky Detection:  {s_p:,}")
    print(f"  Output:           {out.shape}, [{out.min():.4f}, {out.max():.4f}]")
    print(f"  t map:            [{model._t_map.min():.4f}, {model._t_map.max():.4f}]")
    print(f"  A:                {model._A[0, :, 0, 0].tolist()}")
    print(f"  Aux loss:         {aux.item():.4f}")
