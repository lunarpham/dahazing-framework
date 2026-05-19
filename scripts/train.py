"""
DehazeNet Training Script
Usage: python scripts/train.py options/train_dehazenet.toml
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, DIRECT_MODELS
from src.datasets import DehazeDataset
from src.core import get_dark_channel, estimate_atmospheric_light, recover_image
from src.utils import (
    calculate_psnr,
    calculate_ssim,
    build_loss,
    parse_config,
    make_experiment_dirs,
    copy_config,
    save_training_state,
    load_training_state,
    TrainingLogger,
)


# ── Optimizer Builder ────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Build optimizer from config."""
    opt_config = config["train"]["optimizer"]
    opt_type = opt_config.get("type", "adam").lower()
    lr = opt_config.get("lr", 1e-4)
    weight_decay = opt_config.get("weight_decay", 0.0)
    betas = tuple(opt_config.get("betas", [0.9, 0.999]))

    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, betas=betas,
                          weight_decay=weight_decay)
    elif opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, betas=betas,
                           weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_config.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: '{opt_type}'")


# ── Scheduler Builder ────────────────────────────────────────────────────────

def build_scheduler(optimizer: optim.Optimizer, config: dict):
    """Build LR scheduler from config. Returns None if type is 'none'."""
    sched_config = config["train"].get("scheduler", {})
    sched_type = sched_config.get("type", "none").lower()

    if sched_type == "cosine":
        T_max = sched_config.get("T_max", config["train"].get("epochs", 50))
        eta_min = sched_config.get("eta_min", 1e-7)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif sched_type == "step":
        step_size = sched_config.get("step_size", 20)
        gamma = sched_config.get("gamma", 0.5)
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif sched_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: '{sched_type}'")


# ── Validation Loop ──────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device, physics_config, metrics_list,
             is_direct=False):
    """
    Run validation and return average metrics.
    
    Args:
        is_direct: If True, model outputs clean image directly (no physics).
    
    Returns:
        dict of {metric_name: average_value}
    """
    model.eval()
    totals = {m: 0.0 for m in metrics_list}
    count = 0

    for batch in val_loader:
        hazy = batch['hazy'].to(device)
        clear = batch['clear'].to(device)

        if is_direct:
            # Direct model: output IS the clean image
            J_pred = model(hazy)
        else:
            # Transmission-based model: predict t(x), then physics recovery
            t_pred = model(hazy)
            dark_channel = get_dark_channel(
                hazy, window_size=physics_config.get("dark_channel_window", 15)
            )
            atm_light = estimate_atmospheric_light(
                hazy, dark_channel,
                top_percent=physics_config.get("atm_light_top_percent", 0.001)
            )
            J_pred = recover_image(
                hazy, t_pred, atm_light,
                t0=physics_config.get("t_min", 0.1)
            )

        # Calculate metrics per sample in batch
        batch_size = hazy.size(0)
        for i in range(batch_size):
            pred_i = J_pred[i:i+1]
            clear_i = clear[i:i+1]

            if "psnr" in metrics_list:
                totals["psnr"] += calculate_psnr(pred_i, clear_i)
            if "ssim" in metrics_list:
                totals["ssim"] += calculate_ssim(pred_i, clear_i).item()

        count += batch_size

    model.train()

    if count == 0:
        return {m: 0.0 for m in metrics_list}

    return {m: totals[m] / count for m in metrics_list}


# ── Training Pipeline ────────────────────────────────────────────────────────

def train(config_path: str):
    """Main training pipeline driven by TOML config."""

    # ── Parse Config ─────────────────────────────────────────────────────────
    config = parse_config(config_path)
    config = make_experiment_dirs(config)
    copy_config(config)

    exp_name = config["name"]
    train_cfg = config["train"]
    dataset_cfg = config["datasets"]
    physics_cfg = config["physics"]
    val_cfg = config.get("val", {})
    logger_cfg = config["logger"]
    path_cfg = config.get("path", {})

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  Device:     {device}")
    print(f"  Config:     {config['_config_path']}")
    print(f"  Output:     {config['_paths']['exp_root']}")
    print(f"{'='*60}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds_cfg = dataset_cfg["train"]
    train_dataset = DehazeDataset(
        hazy_dir=train_ds_cfg["hazy_dir"],
        clear_dir=train_ds_cfg["clear_dir"],
        patch_size=train_ds_cfg.get("patch_size", 128),
        mode='train',
        strategy=train_ds_cfg.get("pairing_strategy", "reside"),
        augmentation=train_ds_cfg.get("augmentation", ["hflip", "vflip"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_ds_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_ds_cfg.get("num_workers", 4),
    )
    print(f"Training set:   {len(train_dataset)} pairs")

    # Validation dataset (optional)
    val_loader = None
    if "val" in dataset_cfg:
        val_ds_cfg = dataset_cfg["val"]
        val_dataset = DehazeDataset(
            hazy_dir=val_ds_cfg["hazy_dir"],
            clear_dir=val_ds_cfg["clear_dir"],
            patch_size=val_ds_cfg.get("patch_size", 256),
            mode='val',
            strategy=val_ds_cfg.get("pairing_strategy", "reside"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        print(f"Validation set: {len(val_dataset)} pairs")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    model_type = config['network']['type'].lower()
    is_direct = model_type in DIRECT_MODELS
    print(f"Model:          {model_type}")
    print(f"Mode:           {'direct prediction' if is_direct else 'transmission → physics'}")

    # Load pretrained weights if specified
    pretrain_path = path_cfg.get("pretrain")
    if pretrain_path and os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))
        print(f"Loaded pretrained weights from: {pretrain_path}")

    # ── Optimizer, Scheduler, Loss ───────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config).to(device)   # move buffers (Sobel/Gaussian kernels) to GPU

    print(f"Optimizer:      {train_cfg['optimizer']['type']}, lr={train_cfg['optimizer']['lr']}")
    print(f"Scheduler:      {train_cfg.get('scheduler', {}).get('type', 'none')}")
    print(f"Loss:           {train_cfg.get('loss_type', 'mse')}")
    print(f"Epochs:         {train_cfg['epochs']}")
    print()

    # ── Training Logger ──────────────────────────────────────────────────────
    training_logger = TrainingLogger(
        log_dir=config["_paths"]["logs"],
        exp_name=exp_name,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:     {total_params:,}")
    print(f"Logging to:     {config['_paths']['logs']}")
    print()

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0.0
    resume_path = path_cfg.get("resume_state")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        state = load_training_state(resume_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state.get("epoch", 0)
        best_psnr = state.get("best_metric", 0.0)
        print(f"Resumed at epoch {start_epoch}, best PSNR: {best_psnr:.2f}")

    # ── Training Loop ────────────────────────────────────────────────────────
    epochs = train_cfg["epochs"]
    print_freq = logger_cfg.get("print_freq", 10)
    save_freq = logger_cfg.get("save_checkpoint_freq", 5)
    val_freq = val_cfg.get("val_freq", 1)
    metrics_list = val_cfg.get("metrics", ["psnr", "ssim"])
    grad_clip = train_cfg.get("grad_clip", False)
    grad_clip_value = train_cfg.get("grad_clip_value", 1.0)

    model.train()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            hazy = batch['hazy'].to(device)
            clear = batch['clear'].to(device)

            optimizer.zero_grad()

            if is_direct:
                # Direct model: output IS the clean image prediction
                J_pred = model(hazy)
            else:
                # Transmission-based model: predict t(x), then physics
                t_pred = model(hazy)

                # Physics: estimate atmospheric light via DCP
                dark_channel = get_dark_channel(
                    hazy, window_size=physics_cfg.get("dark_channel_window", 15)
                )
                atm_light = estimate_atmospheric_light(
                    hazy, dark_channel,
                    top_percent=physics_cfg.get("atm_light_top_percent", 0.001)
                )

                # Reconstruct clear image via Koschmieder's law
                J_pred = recover_image(
                    hazy, t_pred, atm_light,
                    t0=physics_cfg.get("t_min", 0.1)
                )

            # Loss against ground truth clean image
            loss = criterion(J_pred, clear)
            loss.backward()

            # Optional gradient clipping
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()
            epoch_loss += loss.item()

            # Batch logging
            if batch_idx % print_freq == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}  "
                    f"LR: {lr_current:.2e}"
                )

        # ── Epoch Summary ────────────────────────────────────────────────────
        avg_loss = epoch_loss / max(len(train_loader), 1)
        lr_current = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = None
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            val_metrics = validate(
                model, val_loader, device, physics_cfg, metrics_list,
                is_direct=is_direct
            )
            metrics_str = "  ".join(
                f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()
            )
            print(f"  Validation  ->  {metrics_str}")

            # Track best model by PSNR
            current_psnr = val_metrics.get("psnr", 0.0)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_path = os.path.join(
                    config["_paths"]["checkpoints"], "best.pth"
                )
                torch.save(model.state_dict(), best_path)
                print(f"  * New best PSNR: {best_psnr:.4f}  ->  saved to {best_path}")

        # ── Log Epoch Metrics ────────────────────────────────────────────────
        training_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            lr=lr_current,
            val_metrics=val_metrics,
        )

        # ── Save Checkpoint ──────────────────────────────────────────────────
        if (epoch + 1) % save_freq == 0:
            # Save model weights
            ckpt_path = os.path.join(
                config["_paths"]["checkpoints"],
                f"epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

            # Save full training state for resume
            save_training_state(
                config=config,
                epoch=epoch + 1,
                current_iter=(epoch + 1) * len(train_loader),
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict() if scheduler else None,
                best_metric=best_psnr,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

        print()

    # ── Final Save ───────────────────────────────────────────────────────────
    final_path = os.path.join(config["_paths"]["checkpoints"], "final.pth")
    torch.save(model.state_dict(), final_path)

    # ── Generate Training Analysis ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Generating training analysis...")
    print(f"{'='*60}\n")
    training_logger.generate_analysis(total_params=total_params)

    print(f"\n{'='*60}")
    print(f"  Training completed!")
    print(f"  Final model:  {final_path}")
    if best_psnr > 0:
        print(f"  Best PSNR:    {best_psnr:.4f}")
    print(f"  Logs & plots: {config['_paths']['logs']}")
    print(f"{'='*60}")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DehazeNet with TOML configuration',
        usage='python scripts/train.py <config.toml>'
    )
    parser.add_argument(
        'config', type=str,
        help='Path to TOML configuration file (e.g., options/train_dehazenet.toml)'
    )
    args = parser.parse_args()
    train(args.config)
