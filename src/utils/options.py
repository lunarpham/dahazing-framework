"""TOML configuration parser for experiment configs."""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Python 3.11+ has tomllib built-in; older versions need tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise ImportError(
            "Please install 'tomli' for Python < 3.11: pip install tomli"
        )


# ── Default Configuration ────────────────────────────────────────────────────

DEFAULTS = {
    "name": "msfa_denet_v2_experiment",

    "datasets": {
        "train": {
            "patch_size": 128,
            "batch_size": 8,
            "num_workers": 4,
            "augmentation": ["hflip", "vflip"],
        },
        "val": {
            "patch_size": 256,
        },
    },

    "network": {
        "type": "msfa_denet_v2",
    },

    "train": {
        "epochs": 50,
        "loss_type": "mse",
        "mixed_loss_weights": {"mse": 1.0, "ssim": 0.1},
        "grad_clip": False,
        "grad_clip_value": 1.0,
        "optimizer": {
            "type": "adam",
            "lr": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
        },
        "scheduler": {
            "type": "cosine",
            "T_max": 50,
            "eta_min": 1e-7,
        },
    },

    "val": {
        "val_freq": 1,
        "metrics": ["psnr", "ssim"],
    },

    "logger": {
        "print_freq": 10,
        "save_checkpoint_freq": 5,
        "save_dir": "./experiments",
    },

    "path": {},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge 'override' into 'base'.
    Values in 'override' take precedence.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_config(toml_path: str) -> dict:
    """
    Parse a TOML configuration file and merge with defaults.

    Args:
        toml_path: Path to the .toml configuration file.

    Returns:
        A complete configuration dictionary with all defaults filled in.
    """
    toml_path = Path(toml_path).resolve()

    if not toml_path.exists():
        raise FileNotFoundError(f"Config file not found: {toml_path}")

    if not toml_path.suffix == ".toml":
        raise ValueError(f"Config file must be a .toml file, got: {toml_path}")

    with open(toml_path, "rb") as f:
        user_config = tomllib.load(f)

    # Merge user config on top of defaults
    config = _deep_merge(DEFAULTS, user_config)

    # Store the original config path for reference
    config["_config_path"] = str(toml_path)

    return config


def make_experiment_dirs(config: dict) -> dict:
    """
    Create the experiment directory structure and return updated config with paths.

    Structure:
        experiments/<name>/
            ├── checkpoints/
            ├── results/
            ├── logs/
            └── config.toml  (copy of original)
    """
    save_dir = Path(config["logger"]["save_dir"])
    exp_name = config["name"]
    exp_root = save_dir / exp_name

    dirs = {
        "exp_root": str(exp_root),
        "checkpoints": str(exp_root / "checkpoints"),
        "results": str(exp_root / "results"),
        "logs": str(exp_root / "logs"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Update config with resolved paths
    config["_paths"] = dirs

    return config


def copy_config(config: dict) -> None:
    """
    Copy the original TOML config file into the experiment directory
    for reproducibility.
    """
    src = config.get("_config_path")
    if src and os.path.exists(src):
        dst_dir = config.get("_paths", {}).get("exp_root", ".")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(dst_dir, f"config_{timestamp}.toml")
        shutil.copy2(src, dst)


def save_training_state(config: dict, epoch: int, current_iter: int,
                        model_state: dict, optimizer_state: dict,
                        scheduler_state: dict = None,
                        best_metric: float = None) -> str:
    """
    Save a full training state for resume support.

    Returns:
        Path to the saved state file.
    """
    import torch

    state = {
        "epoch": epoch,
        "iter": current_iter,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
    }

    if scheduler_state is not None:
        state["scheduler_state_dict"] = scheduler_state

    if best_metric is not None:
        state["best_metric"] = best_metric

    state_path = os.path.join(
        config["_paths"]["checkpoints"],
        "training_state.pth"
    )
    torch.save(state, state_path)
    return state_path


def load_training_state(state_path: str) -> dict:
    """Load a previously saved training state."""
    import torch
    return torch.load(state_path, map_location="cpu", weights_only=False)
