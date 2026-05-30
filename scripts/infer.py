"""
MSFA-DeNet v2 Inference Script
Usage: python scripts/infer.py <config.toml>
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.utils import load_image, to_tensor, save_image, parse_config


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


def infer(config_path: str, passes_override: int = None):
    """Run inference on images using a trained model."""

    config = parse_config(config_path)
    path_cfg = config.get("path", {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = config.get('name', 'inference')

    infer_cfg = config.get("infer", {})
    num_passes = passes_override if passes_override is not None else infer_cfg.get("passes", 1)

    print(f"\n{'='*60}")
    print(f"  MSFA-DeNet v2 Inference")
    print(f"  Experiment:  {exp_name}")
    print(f"  Device:      {device}")
    print(f"  Passes:      {num_passes}{'  (iterative)' if num_passes > 1 else ''}")
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

    ckpt_suffix = Path(model_path).stem if model_path else "msfa_denet_v2"

    # ── Input / Output ───────────────────────────────────────────────────────
    input_path = path_cfg.get("input", "test_images/")
    default_output = os.path.join("experiments", exp_name, "results")
    output_dir = path_cfg.get("output", default_output)
    os.makedirs(output_dir, exist_ok=True)

    images = find_images(input_path)
    if not images:
        print(f"No images found at: {input_path}")
        return

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Found {len(images)} image(s) to process.\n")

    # ── Process ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        for idx, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_{ckpt_suffix}{ext}")

            img_np = load_image(img_path)
            current = to_tensor(img_np).to(device)

            for _ in range(num_passes):
                current = model(current)

            save_image(output_path, current)
            print(f"  [{idx+1}/{len(images)}] {filename} → {output_path}")

    print(f"\nDone! {len(images)} image(s) saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run MSFA-DeNet v2 inference',
        usage='python scripts/infer.py <config.toml>'
    )
    parser.add_argument('config', type=str, help='Path to TOML config file')
    parser.add_argument('--passes', type=int, default=None,
                        help='Number of iterative dehazing passes (overrides config)')
    args = parser.parse_args()
    infer(args.config, passes_override=args.passes)
