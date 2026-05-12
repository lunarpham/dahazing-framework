"""
DehazeNet Inference Script
Usage: python scripts/infer.py options/infer_dehazenet.toml
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, DIRECT_MODELS
from src.core import get_dark_channel, estimate_atmospheric_light, recover_image
from src.utils import load_image, to_tensor, save_image, parse_config, post_process


def find_images(path: str) -> list:
    """Find all image files in a path (single file or directory)."""
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
        return sorted(files)
    else:
        raise FileNotFoundError(f"Input path not found: {path}")


def infer(config_path: str):
    """Main inference pipeline driven by TOML config."""

    # ── Parse Config ─────────────────────────────────────────────────────────
    config = parse_config(config_path)
    physics_cfg = config.get("physics", {})
    path_cfg = config.get("path", {})

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  DehazeNet Inference")
    print(f"  Device:  {device}")
    print(f"  Config:  {config.get('_config_path', config_path)}")
    print(f"{'='*60}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)

    model_path = path_cfg.get("model_path")
    if model_path and os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Loaded weights: {model_path}")
    else:
        print("Warning: No weights provided, using randomly initialized model.")

    model.eval()

    model_type = config.get('network', {}).get('type', 'dehazenet').lower()
    is_direct = model_type in DIRECT_MODELS
    print(f"Model type: {model_type} ({'direct prediction' if is_direct else 'transmission → physics'})")

    # ── Input / Output ───────────────────────────────────────────────────────
    input_path = path_cfg.get("input", "test_images/")
    output_dir = path_cfg.get("output", "results/")
    os.makedirs(output_dir, exist_ok=True)

    images = find_images(input_path)
    if not images:
        print(f"No images found at: {input_path}")
        return

    print(f"Found {len(images)} image(s) to process.\n")

    # ── Process Each Image ───────────────────────────────────────────────────
    with torch.no_grad():
        for idx, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_dehazed{ext}")

            # Load image
            img_np = load_image(img_path)
            img_tensor = to_tensor(img_np).to(device)

            # Forward pass
            if is_direct:
                # Direct model: output IS the clean image
                dehazed = model(img_tensor)
            else:
                # Transmission-based model: predict t(x), then physics
                t_pred = model(img_tensor)

                # Estimate atmospheric light via DCP
                dark_channel = get_dark_channel(
                    img_tensor,
                    window_size=physics_cfg.get("dark_channel_window", 15)
                )
                atm_light = estimate_atmospheric_light(
                    img_tensor, dark_channel,
                    top_percent=physics_cfg.get("atm_light_top_percent", 0.001)
                )

                # Reconstruct image via Koschmieder's law
                dehazed = recover_image(
                    img_tensor, t_pred, atm_light,
                    t0=physics_cfg.get("t_min", 0.1)
                )

            # Save result
            save_image(output_path, dehazed)
            print(f"  [{idx+1}/{len(images)}] {filename}  →  {output_path}")

    print(f"\nDone! Results saved to: {output_dir}")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run DehazeNet inference with TOML configuration',
        usage='python scripts/infer.py <config.toml>'
    )
    parser.add_argument(
        'config', type=str,
        help='Path to TOML configuration file (e.g., options/infer_dehazenet.toml)'
    )
    args = parser.parse_args()
    infer(args.config)
