"""Quick verification of AOD-CA-PA-Net v2 and loss functions."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.aodnet_capa import AODCAPANet
from src.utils.losses import (
    MixedLoss, ColorConsistencyLoss, build_loss
)

# ── Test 1: Model architecture ──────────────────────────────────────────────
print("=" * 60)
print("  Test 1: AOD-CA-PA-Net v2 Architecture")
print("=" * 60)

model = AODCAPANet()
dummy = torch.rand(1, 3, 256, 256)

out = model(dummy)
diag = model.forward_with_maps(dummy)

total_params = sum(p.numel() for p in model.parameters())
fa_params = sum(p.numel() for p in model.feature_attention.parameters())
k_params = sum(p.numel() for p in model.k_estimator.parameters())

print(f"  Output shape:       {out.shape}")
print(f"  K map shape:        {diag['k_map'].shape}")
print(f"  CA weights shape:   {diag['ca_weights'].shape}")
print(f"  PA map shape:       {diag['pa_map'].shape}")
print(f"  Output range:       [{out.min():.4f}, {out.max():.4f}]")
print(f"  K range:            [{diag['k_map'].min():.4f}, {diag['k_map'].max():.4f}]")
print(f"  Total parameters:   {total_params:,}")
print(f"    FA block:         {fa_params:,}")
print(f"    K head:           {k_params:,}")

assert out.shape == (1, 3, 256, 256), "Output shape mismatch!"
assert total_params > 10000, f"Too few params: {total_params}"
print("  [PASS] Model architecture OK")

# ── Test 2: ColorConsistencyLoss ─────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  Test 2: ColorConsistencyLoss")
print("=" * 60)

color_loss = ColorConsistencyLoss()
pred = torch.rand(2, 3, 64, 64)
target = torch.rand(2, 3, 64, 64)
loss_val = color_loss(pred, target)
print(f"  Loss value: {loss_val.item():.6f}")
assert loss_val.item() >= 0, "Loss should be non-negative"

# Identical inputs should give ~0 loss
loss_same = color_loss(pred, pred)
print(f"  Same-input loss: {loss_same.item():.8f}")
assert loss_same.item() < 1e-6, "Same-input loss should be ~0"
print("  [PASS] ColorConsistencyLoss OK")

# ── Test 3: MixedLoss with color key ────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  Test 3: MixedLoss with 'color' key")
print("=" * 60)

mixed = MixedLoss({"l1": 1.0, "color": 0.2})
loss_val = mixed(pred, target)
print(f"  Mixed loss value: {loss_val.item():.6f}")
assert "color" in mixed.losses, "'color' not registered in MixedLoss!"
print("  [PASS] MixedLoss with color OK")

# ── Test 4: build_loss with full CAPA config ─────────────────────────────────
print(f"\n{'=' * 60}")
print("  Test 4: build_loss with CAPA training config")
print("=" * 60)

config = {
    "train": {
        "loss_type": "mixed",
        "mixed_loss_weights": {
            "l1": 1.0,
            "ssim": 0.5,
            "sobel": 0.5,
            "laplacian": 0.3,
            "dcp": 0.01,
            "perceptual": 0.1,
            "color": 0.2,
        }
    }
}
criterion = build_loss(config)
loss_val = criterion(pred, target)
print(f"  Full mixed loss: {loss_val.item():.6f}")
print(f"  Active losses: {list(criterion.losses.keys())}")
assert "color" in criterion.losses
assert "perceptual" in criterion.losses
print("  [PASS] build_loss with full config OK")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  ALL TESTS PASSED!")
print(f"{'=' * 60}")
