"""
MSFA-DeNet v2 Training Script
Usage: python scripts/train.py options/train_msfa_denet_v2.toml
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.datasets import DehazeDataset, build_domain_balanced_sampler
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
    save_image,
)


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


@torch.no_grad()
def validate(model, val_loader, device, metrics_list,
             save_dir=None, epoch=None, max_save=0):
    """Run validation and return average metrics. Optionally saves comparison images."""
    model.eval()
    totals = {m: 0.0 for m in metrics_list}
    count = 0
    saved_count = 0

    for batch in val_loader:
        hazy = batch['hazy'].to(device)
        clear = batch['clear'].to(device)

        output = model(hazy)
        J_pred = output[0] if isinstance(output, tuple) else output

        batch_size = hazy.size(0)
        for i in range(batch_size):
            pred_i = J_pred[i:i+1]
            clear_i = clear[i:i+1]

            if "psnr" in metrics_list:
                totals["psnr"] += calculate_psnr(pred_i, clear_i)
            if "ssim" in metrics_list:
                totals["ssim"] += calculate_ssim(pred_i, clear_i).item()

            # Save side-by-side: hazy | dehazed | ground truth
            if save_dir is not None and epoch is not None and (max_save == 0 or saved_count < max_save):
                comparison = torch.cat([
                    hazy[i:i+1], J_pred[i:i+1], clear[i:i+1]
                ], dim=3)

                hazy_path = batch.get('hazy_path', [None])
                if isinstance(hazy_path, (list, tuple)) and i < len(hazy_path) and hazy_path[i]:
                    base_name = os.path.splitext(os.path.basename(hazy_path[i]))[0]
                else:
                    base_name = f"val_{saved_count:03d}"

                img_dir = os.path.join(save_dir, base_name)
                os.makedirs(img_dir, exist_ok=True)

                fname = f"{base_name}_epoch_{epoch:03d}.png"
                save_image(os.path.join(img_dir, fname), comparison)
                saved_count += 1

        count += batch_size

    if save_dir is not None and saved_count > 0:
        print(f"  Saved {saved_count} validation images to: {save_dir}")

    model.train()

    if count == 0:
        return {m: 0.0 for m in metrics_list}

    return {m: totals[m] / count for m in metrics_list}


def train(config_path: str, resume_arg: str = None):
    """Main training pipeline driven by TOML config."""

    config = parse_config(config_path)
    config = make_experiment_dirs(config)
    copy_config(config)

    exp_name = config["name"]
    train_cfg = config["train"]
    dataset_cfg = config["datasets"]
    val_cfg = config.get("val", {})
    logger_cfg = config["logger"]
    path_cfg = config.get("path", {})

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
        augmentation=train_ds_cfg.get("augmentation", ["hflip", "vflip"]),
    )

    use_balanced = train_ds_cfg.get("domain_balanced_sampling", False)
    if use_balanced:
        sampler, domain_counts = build_domain_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_ds_cfg.get("batch_size", 8),
            sampler=sampler,
            num_workers=train_ds_cfg.get("num_workers", 4),
        )
        print(f"Domain-balanced sampling enabled:")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain:>12s}: {count} samples")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_ds_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=train_ds_cfg.get("num_workers", 4),
        )
    print(f"Training set:   {len(train_dataset)} pairs")

    val_loader = None
    if "val" in dataset_cfg:
        val_ds_cfg = dataset_cfg["val"]
        val_dataset = DehazeDataset(
            hazy_dir=val_ds_cfg["hazy_dir"],
            clear_dir=val_ds_cfg["clear_dir"],
            patch_size=val_ds_cfg.get("patch_size", 256),
            mode='val',

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
    print(f"Model:          {config['network']['type']}")

    pretrain_path = path_cfg.get("pretrain")
    if pretrain_path and os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))
        print(f"Loaded pretrained weights: {pretrain_path}")

    # ── Optimizer, Scheduler, Loss ───────────────────────────────────────────
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config).to(device)

    print(f"Optimizer:      {train_cfg['optimizer']['type']}, lr={train_cfg['optimizer']['lr']}")
    print(f"Scheduler:      {train_cfg.get('scheduler', {}).get('type', 'none')}")
    print(f"Loss:           {train_cfg.get('loss_type', 'mse')}")
    print(f"Epochs:         {train_cfg['epochs']}")

    training_logger = TrainingLogger(
        log_dir=config["_paths"]["logs"],
        exp_name=exp_name,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:     {total_params:,}")
    print(f"Logging to:     {config['_paths']['logs']}\n")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr = 0.0
    resume_path = resume_arg if resume_arg else path_cfg.get("resume_state")

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        state = load_training_state(resume_path)

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = state.get("epoch", 0)
            best_psnr = state.get("best_metric", 0.0)
            print(f"Resumed at epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
        else:
            model.load_state_dict(state)
            import re
            match = re.search(r'epoch_(\d+)', os.path.basename(resume_path))
            if match:
                start_epoch = int(match.group(1))
            print(f"Loaded weights, starting at epoch {start_epoch}")

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

            output = model(hazy)
            J_pred = output[0] if isinstance(output, tuple) else output

            loss = criterion(J_pred, clear)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % print_freq == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(
                    f"  Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}  "
                    f"LR: {lr_current:.2e}"
                )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        lr_current = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")

        if scheduler is not None:
            scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = None
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            torch.cuda.empty_cache()
            val_metrics = validate(
                model, val_loader, device, metrics_list,
                save_dir=config["_paths"]["results"],
                epoch=epoch + 1,
                max_save=val_cfg.get("save_images", 0),
            )
            metrics_str = "  ".join(
                f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()
            )
            print(f"  Validation  ->  {metrics_str}")

            current_psnr = val_metrics.get("psnr", 0.0)
            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_path = os.path.join(config["_paths"]["checkpoints"], "best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  * New best PSNR: {best_psnr:.4f} -> {best_path}")

        training_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            lr=lr_current,
            val_metrics=val_metrics,
        )

        # ── Save Checkpoint ──────────────────────────────────────────────────
        if (epoch + 1) % save_freq == 0:
            ckpt_path = os.path.join(
                config["_paths"]["checkpoints"], f"epoch_{epoch+1}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MSFA-DeNet v2 with TOML configuration',
        usage='python scripts/train.py <config.toml>'
    )
    parser.add_argument(
        'config', type=str,
        help='Path to TOML config file (e.g., options/train_msfa_denet_v2.toml)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a checkpoint (.pth) to resume training from'
    )
    args = parser.parse_args()
    train(args.config, args.resume)
