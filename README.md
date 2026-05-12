# DehazeNet Framework

A PyTorch-based framework for image dehazing, inspired by DehazeNet. Uses a hybrid approach where a CNN estimates the transmission map, the Dark Channel Prior estimates atmospheric light, and Koschmieder's law reconstructs the clear image.

**Fully configurable via TOML** — duplicate a config, tweak parameters, run experiments.

## Features

- **TOML Configuration** — all hyperparameters (model, dataset, optimizer, scheduler, loss, physics) are set in a single `.toml` file. Easily run multiple experiments by duplicating configs.
- **Smart Data Loader** — automatically pairs hazy/clear images using filename matching (RESIDE convention or strict match). On-the-fly augmentation.
- **Koschmieder Integration** — the loss is driven by the physical model: $J(x) = \frac{I(x) - A}{\max(t(x), t_0)} + A$. No ground-truth transmission map needed.
- **Validation Loop** — PSNR and SSIM tracked during training with automatic best-model saving.
- **LR Scheduler** — supports Cosine Annealing, Step decay, or no scheduling.
- **Resume Training** — full training state (model, optimizer, scheduler, epoch, best metric) is saved and can be resumed.
- **Experiment Management** — each experiment gets its own directory with checkpoints, logs, and a copy of the config used.
- **Extensible** — model registry for adding new architectures, loss builder for custom losses.

## Project Structure

```
DehazeNet/
├── options/                          # TOML config templates
│   ├── train_dehazenet.toml          #   Training config
│   └── infer_dehazenet.toml          #   Inference config
├── scripts/
│   ├── train.py                      # Training entry point
│   └── infer.py                      # Inference entry point
├── src/
│   ├── core/                         # Physical model
│   │   ├── dcp.py                    #   Dark Channel Prior
│   │   └── koschmieder.py            #   Koschmieder's law
│   ├── datasets/                     # Data loading
│   │   ├── dataset.py                #   PyTorch Dataset
│   │   └── pairing.py                #   Image pair matching
│   ├── models/                       # Neural networks
│   │   ├── builder.py                #   Model registry
│   │   ├── components.py             #   Custom layers (BReLU, Maxout)
│   │   └── dehazenet.py              #   DehazeNet architecture
│   └── utils/                        # Utilities
│       ├── options.py                #   TOML config parser
│       ├── losses.py                 #   Loss function builder
│       ├── metrics.py                #   PSNR, SSIM
│       └── image_io.py               #   Image I/O
├── experiments/                      # Auto-created experiment outputs
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

1. Edit the config template to point to your dataset:
   ```bash
   cp options/train_dehazenet.toml options/my_experiment.toml
   # Edit my_experiment.toml: set hazy_dir, clear_dir, etc.
   ```

2. Start training:
   ```bash
   python scripts/train.py options/my_experiment.toml
   ```

3. Results are saved to `experiments/<name>/`:
   - `checkpoints/best.pth` — best model (by validation PSNR)
   - `checkpoints/epoch_N.pth` — periodic checkpoints
   - `checkpoints/training_state.pth` — full state for resuming
   - `config_*.toml` — copy of the config used

### Resume Training

Set `resume_state` in your config's `[path]` section:
```toml
[path]
resume_state = "experiments/my_experiment/checkpoints/training_state.pth"
```
Then run the same command:
```bash
python scripts/train.py options/my_experiment.toml
```

### Inference

```bash
python scripts/infer.py options/infer_dehazenet.toml
```

Supports both single images and directories. Edit `[path]` in the config:
```toml
[path]
model_path = "experiments/my_experiment/checkpoints/best.pth"
input = "test_images/"
output = "results/"
```

## Configuration Reference

All settings are in the TOML file. See `options/train_dehazenet.toml` for a fully commented template. Key sections:

| Section | Key Settings |
|---------|-------------|
| `[datasets.train]` | `hazy_dir`, `clear_dir`, `patch_size`, `batch_size`, `augmentation` |
| `[datasets.val]` | `hazy_dir`, `clear_dir`, `patch_size` |
| `[network]` | `type` (e.g., `"dehazenet"`) |
| `[physics]` | `dark_channel_window`, `atm_light_top_percent`, `t_min` |
| `[train]` | `epochs`, `loss_type`, `grad_clip` |
| `[train.optimizer]` | `type`, `lr`, `betas`, `weight_decay` |
| `[train.scheduler]` | `type`, `T_max`, `eta_min` |
| `[val]` | `val_freq`, `metrics` |
| `[logger]` | `print_freq`, `save_checkpoint_freq`, `save_dir` |
| `[path]` | `pretrain`, `resume_state` |
