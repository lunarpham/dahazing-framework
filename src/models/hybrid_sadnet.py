"""
Hybrid MSFA-Transformer (HybridHazeNet)

This architecture directly addresses the synthetic-to-real domain gap by explicitly 
decoupling the image processing pipeline into distinct frequency bands:

1. Shallow CNN Branch (High-Frequency): Utilizes the Feature Attention (FA) mechanisms 
   from MSFA-Net to preserve sharp, localized edges and structural details.
2. Deep Transformer Branch (Low-Frequency): Utilizes Multi-Dconv Head Transposed Attention 
   (MDTA) inspired by Restormer to establish a global receptive field, accurately 
   estimating global atmospheric light and non-uniform transmission maps without quadratic 
   computational overhead.

Input: (B, 3, H, W) hazy image
Output: (B, 3, H, W) dehazed image in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── MSFA-Net CNN Components (Local / High Frequency) ─────────────────────────

class FeatureAttention(nn.Module):
    """
    Dual Attention mechanism (Channel + Spatial) from MSFA-Net.
    Used to dynamically weight important local convolutional features.
    """
    def __init__(self, channels):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # Pixel / Spatial Attention
        self.pa = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_out = self.ca(x)
        pa_out = self.pa(x)
        return x * ca_out * pa_out

class CNNBlock(nn.Module):
    """Basic residual CNN block with Feature Attention."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.fa = FeatureAttention(channels)

    def forward(self, x):
        res = x
        x = self.act(self.norm(self.conv1(x)))
        x = self.conv2(x)
        x = self.fa(x)
        return x + res

# ─── Transformer Components (Global / Low Frequency) ──────────────────────────

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention (Restormer-style).
    Computes self-attention across the feature channel dimension rather than spatial,
    achieving linear complexity O(N) instead of quadratic O(N^2).
    """
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention across CHANNELS
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        # Normalize 
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed Attention (Channel-wise)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        attn = attn.to(torch.float32).softmax(dim=-1).to(attn.dtype)

        out = (attn @ v)
        out = out.view(b, c, h, w)

        return self.project_out(out)

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # Gating mechanism
        x = F.gelu(x1) * x2
        return self.project_out(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Norm1 -> Attn -> Res
        x_norm = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.transpose(1, 2).view(b, c, h, w)
        x = x + self.attn(x_norm)
        
        # Norm2 -> FFN -> Res
        x_norm = x.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.transpose(1, 2).view(b, c, h, w)
        x = x + self.ffn(x_norm)
        
        return x


# ─── Hybrid Architecture ──────────────────────────────────────────────────────

class HybridHazeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_heads=8, num_vit_blocks=4):
        super().__init__()
        
        # 1. Shallow Feature Extraction
        self.embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # 2. Local CNN Branch (Operates at 1x resolution)
        self.cnn_branch = nn.Sequential(
            CNNBlock(dim),
            CNNBlock(dim)
        )
        
        # 3. Global Transformer Branch (Operates at 0.5x resolution for global context)
        self.downsample = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
        self.vit_blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads) for _ in range(num_vit_blocks)
        ])
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)
        
        # 4. Fusion Module
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            FeatureAttention(dim), # Use attention to pick between local vs global details
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
        # 5. Reconstruction
        self.reconstruct = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        
        # Initialize
        self._initialize_weights()

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Shallow embedding
        feat = self.embed(x)
        
        # Branch 1: CNN Local Details (Preserves high frequencies)
        local_feat = self.cnn_branch(feat)
        
        # Branch 2: ViT Global Context (Extracts global atmospheric light and depth mappings)
        global_feat_down = self.downsample(feat)
        global_feat_processed = self.vit_blocks(global_feat_down)
        global_feat = self.upsample(global_feat_processed)
        
        # Handle spatial mismatch if the original image has odd dimensions
        if global_feat.shape[2:] != local_feat.shape[2:]:
            global_feat = F.interpolate(global_feat, size=local_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Feature Fusion
        fused = torch.cat([local_feat, global_feat], dim=1)
        refined_feat = self.fuse(fused)
        
        # Global Residual Mapping
        residual = self.reconstruct(refined_feat)
        
        # Tanh bounds the mathematical output of the transformer
        residual = torch.tanh(residual)
        
        clean = x + residual
        
        # THE ARCHITECTURAL FIX: 
        # Do NOT clamp during training! Clamping sets the gradient to 0.0 for pixels 
        # that overshoot 1.0. This prevents the L1 and Highlight losses from physically
        # reaching the white blobs to pull them down.
        if self.training:
            return clean
        else:
            return torch.clamp(clean, 0.0, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # THE INITIALIZATION FIX:
        # Zero out the final layer. This guarantees that at epoch 0, residual = 0 
        # and clean = x. The network starts by outputting the exact hazy image 
        # instead of astronomical white noise, stabilizing early training.
        nn.init.constant_(self.reconstruct.weight, 0)
        if self.reconstruct.bias is not None:
            nn.init.constant_(self.reconstruct.bias, 0)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing HybridHazeNet...")
    model = HybridHazeNet(dim=64, num_vit_blocks=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    
    # Dummy hazy image (Batch, Channels, Height, Width)
    dummy_input = torch.rand(2, 3, 256, 256) 
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert dummy_input.shape == output.shape, "Output shape mismatch!"
    print("Test passed successfully!")