"""
Training Logger for DehazeNet Framework.

Provides structured CSV logging of training metrics (loss, LR, validation
PSNR/SSIM per epoch) and generates analysis plots upon completion.

Outputs:
    logs/training_log.csv     — Structured metrics per epoch
    logs/loss_curve.png       — Training loss over epochs
    logs/psnr_curve.png       — Validation PSNR over epochs
    logs/ssim_curve.png       — Validation SSIM over epochs
    logs/lr_schedule.png      — Learning rate schedule
    logs/training_summary.txt — Final text summary with analysis
"""

import os
import csv
from datetime import datetime


class TrainingLogger:
    """
    Logs training metrics to CSV and generates post-training analysis.

    Usage:
        logger = TrainingLogger(log_dir="./experiments/my_exp/logs",
                                exp_name="msfa_net_multidomain")
        # In training loop:
        logger.log_epoch(epoch=1, train_loss=0.05, lr=1e-4,
                         val_metrics={"psnr": 22.5, "ssim": 0.85})
        # After training:
        logger.generate_analysis()
    """

    CSV_FILENAME = "training_log.csv"
    SUMMARY_FILENAME = "training_summary.txt"

    def __init__(self, log_dir: str, exp_name: str = "experiment"):
        self.log_dir = log_dir
        self.exp_name = exp_name
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, self.CSV_FILENAME)
        self.summary_path = os.path.join(log_dir, self.SUMMARY_FILENAME)

        # In-memory history for analysis
        self.history = {
            "epoch": [],
            "train_loss": [],
            "lr": [],
            "val_psnr": [],
            "val_ssim": [],
        }

        # Initialize CSV with header
        self._init_csv()

        # Track timing
        self.start_time = datetime.now()
        self.best_psnr = 0.0
        self.best_psnr_epoch = 0
        self.best_ssim = 0.0
        self.best_ssim_epoch = 0

    def _init_csv(self):
        """Create CSV file with header row."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "learning_rate",
                "val_psnr", "val_ssim", "timestamp"
            ])

    def log_epoch(self, epoch: int, train_loss: float, lr: float,
                  val_metrics: dict = None):
        """
        Log one epoch's metrics to CSV and in-memory history.

        Args:
            epoch: Current epoch number (1-indexed).
            train_loss: Average training loss for this epoch.
            lr: Learning rate at end of epoch.
            val_metrics: Optional dict with "psnr" and/or "ssim" keys.
        """
        val_psnr = val_metrics.get("psnr", None) if val_metrics else None
        val_ssim = val_metrics.get("ssim", None) if val_metrics else None

        # Update in-memory history
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["lr"].append(lr)
        self.history["val_psnr"].append(val_psnr)
        self.history["val_ssim"].append(val_ssim)

        # Track bests
        if val_psnr is not None and val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            self.best_psnr_epoch = epoch
        if val_ssim is not None and val_ssim > self.best_ssim:
            self.best_ssim = val_ssim
            self.best_ssim_epoch = epoch

        # Append to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{lr:.2e}",
                f"{val_psnr:.4f}" if val_psnr is not None else "",
                f"{val_ssim:.4f}" if val_ssim is not None else "",
                timestamp,
            ])

    def generate_analysis(self, total_params: int = None):
        """
        Generate post-training analysis: plots + text summary.

        Args:
            total_params: Total model parameter count (for summary).
        """
        self._generate_plots()
        self._generate_summary(total_params)

    def _generate_plots(self):
        """Generate training curve plots using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARNING] matplotlib not installed. Skipping plot generation.")
            print("  Install with: pip install matplotlib")
            return

        epochs = self.history["epoch"]
        if len(epochs) < 2:
            print("  [WARNING] Not enough epochs to generate plots.")
            return

        # Shared style
        plt.rcParams.update({
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor": "#16213e",
            "axes.edgecolor": "#e94560",
            "axes.labelcolor": "#eee",
            "xtick.color": "#aaa",
            "ytick.color": "#aaa",
            "text.color": "#eee",
            "grid.color": "#333",
            "grid.alpha": 0.4,
            "font.size": 11,
        })

        # ── 1. Training Loss Curve ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, self.history["train_loss"],
                color="#e94560", linewidth=1.8, label="Train Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss - {self.exp_name}", fontweight="bold")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # ── 2. Validation PSNR Curve ─────────────────────────────────────────
        psnr_values = self.history["val_psnr"]
        psnr_epochs = [e for e, v in zip(epochs, psnr_values) if v is not None]
        psnr_vals = [v for v in psnr_values if v is not None]

        if psnr_vals:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(psnr_epochs, psnr_vals,
                    color="#0f3460", linewidth=1.8, marker="o", markersize=3,
                    label="Val PSNR")
            # Highlight best
            ax.axhline(y=self.best_psnr, color="#e94560", linestyle="--",
                       linewidth=1, alpha=0.7,
                       label=f"Best: {self.best_psnr:.2f} dB (ep {self.best_psnr_epoch})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title(f"Validation PSNR - {self.exp_name}", fontweight="bold")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, "psnr_curve.png"), dpi=150)
            plt.close(fig)

        # ── 3. Validation SSIM Curve ─────────────────────────────────────────
        ssim_values = self.history["val_ssim"]
        ssim_epochs = [e for e, v in zip(epochs, ssim_values) if v is not None]
        ssim_vals = [v for v in ssim_values if v is not None]

        if ssim_vals:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ssim_epochs, ssim_vals,
                    color="#533483", linewidth=1.8, marker="o", markersize=3,
                    label="Val SSIM")
            ax.axhline(y=self.best_ssim, color="#e94560", linestyle="--",
                       linewidth=1, alpha=0.7,
                       label=f"Best: {self.best_ssim:.4f} (ep {self.best_ssim_epoch})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("SSIM")
            ax.set_title(f"Validation SSIM - {self.exp_name}", fontweight="bold")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir, "ssim_curve.png"), dpi=150)
            plt.close(fig)

        # ── 4. Learning Rate Schedule ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, self.history["lr"],
                color="#00b4d8", linewidth=1.8, label="Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.set_title(f"LR Schedule - {self.exp_name}", fontweight="bold")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

        # ── 5. Combined Overview (2x2 grid) ─────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axes[0, 0].plot(epochs, self.history["train_loss"],
                        color="#e94560", linewidth=1.5)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # PSNR
        if psnr_vals:
            axes[0, 1].plot(psnr_epochs, psnr_vals,
                            color="#0f3460", linewidth=1.5)
            axes[0, 1].axhline(y=self.best_psnr, color="#e94560",
                               linestyle="--", linewidth=1, alpha=0.7)
        axes[0, 1].set_title("Validation PSNR (dB)")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("PSNR")
        axes[0, 1].grid(True)

        # SSIM
        if ssim_vals:
            axes[1, 0].plot(ssim_epochs, ssim_vals,
                            color="#533483", linewidth=1.5)
            axes[1, 0].axhline(y=self.best_ssim, color="#e94560",
                               linestyle="--", linewidth=1, alpha=0.7)
        axes[1, 0].set_title("Validation SSIM")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("SSIM")
        axes[1, 0].grid(True)

        # LR
        axes[1, 1].plot(epochs, self.history["lr"],
                        color="#00b4d8", linewidth=1.5)
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True)

        fig.suptitle(f"Training Overview - {self.exp_name}",
                     fontsize=14, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(self.log_dir, "training_overview.png"), dpi=150)
        plt.close(fig)

        print(f"  Plots saved to: {self.log_dir}")

    def _generate_summary(self, total_params: int = None):
        """Generate a text summary with training analysis."""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        epochs = self.history["epoch"]
        losses = self.history["train_loss"]
        psnr_vals = [v for v in self.history["val_psnr"] if v is not None]
        ssim_vals = [v for v in self.history["val_ssim"] if v is not None]

        lines = []
        lines.append("=" * 70)
        lines.append(f"  TRAINING ANALYSIS REPORT")
        lines.append(f"  Experiment: {self.exp_name}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # ── Training Summary ─────────────────────────────────────────────────
        lines.append("--- Training Summary ---")
        lines.append(f"  Total epochs:       {len(epochs)}")
        lines.append(f"  Training time:      {hours}h {minutes}m {seconds}s")
        if total_params is not None:
            lines.append(f"  Model parameters:   {total_params:,}")
        lines.append(f"  Initial loss:       {losses[0]:.6f}")
        lines.append(f"  Final loss:         {losses[-1]:.6f}")
        if losses[0] > 0:
            reduction = (1 - losses[-1] / losses[0]) * 100
            lines.append(f"  Loss reduction:     {reduction:.1f}%")
        lines.append("")

        # ── Loss Analysis ────────────────────────────────────────────────────
        lines.append("--- Loss Analysis ---")
        lines.append(f"  Minimum loss:       {min(losses):.6f} (epoch {losses.index(min(losses)) + 1})")
        lines.append(f"  Maximum loss:       {max(losses):.6f} (epoch {losses.index(max(losses)) + 1})")

        # Convergence analysis: check last 10% of epochs
        tail_size = max(1, len(losses) // 10)
        tail_losses = losses[-tail_size:]
        if len(tail_losses) > 1:
            tail_std = (sum((x - sum(tail_losses)/len(tail_losses))**2
                           for x in tail_losses) / len(tail_losses)) ** 0.5
            lines.append(f"  Final {tail_size} epochs std: {tail_std:.6f}")
            if tail_std < 0.001:
                lines.append(f"  Convergence:        CONVERGED (stable)")
            elif tail_std < 0.01:
                lines.append(f"  Convergence:        NEAR-CONVERGED")
            else:
                lines.append(f"  Convergence:        STILL TRAINING (consider more epochs)")
        lines.append("")

        # ── Validation PSNR Analysis ─────────────────────────────────────────
        if psnr_vals:
            lines.append("--- Validation PSNR Analysis ---")
            lines.append(f"  Best PSNR:          {self.best_psnr:.4f} dB (epoch {self.best_psnr_epoch})")
            lines.append(f"  Final PSNR:         {psnr_vals[-1]:.4f} dB")
            lines.append(f"  PSNR range:         [{min(psnr_vals):.4f}, {max(psnr_vals):.4f}] dB")
            psnr_improvement = psnr_vals[-1] - psnr_vals[0]
            lines.append(f"  Total improvement:  {psnr_improvement:+.4f} dB")

            # Overfitting check: is final PSNR much worse than best?
            if len(psnr_vals) > 10:
                gap = self.best_psnr - psnr_vals[-1]
                if gap > 1.0:
                    lines.append(f"  [!] WARNING: PSNR dropped {gap:.2f} dB from best.")
                    lines.append(f"      Possible overfitting detected.")
                    lines.append(f"      Consider using the best checkpoint (epoch {self.best_psnr_epoch}).")
                elif gap > 0.3:
                    lines.append(f"  [~] PSNR slightly below best by {gap:.2f} dB.")
                    lines.append(f"      Minor fluctuation - likely normal.")
                else:
                    lines.append(f"  [OK] PSNR stable near best value.")
            lines.append("")

        # ── Validation SSIM Analysis ─────────────────────────────────────────
        if ssim_vals:
            lines.append("--- Validation SSIM Analysis ---")
            lines.append(f"  Best SSIM:          {self.best_ssim:.4f} (epoch {self.best_ssim_epoch})")
            lines.append(f"  Final SSIM:         {ssim_vals[-1]:.4f}")
            lines.append(f"  SSIM range:         [{min(ssim_vals):.4f}, {max(ssim_vals):.4f}]")
            ssim_improvement = ssim_vals[-1] - ssim_vals[0]
            lines.append(f"  Total improvement:  {ssim_improvement:+.4f}")
            lines.append("")

        # ── Quality Assessment ───────────────────────────────────────────────
        lines.append("--- Quality Assessment ---")
        if psnr_vals:
            final_psnr = psnr_vals[-1]
            if final_psnr >= 30:
                lines.append(f"  PSNR Grade:         EXCELLENT (>= 30 dB)")
            elif final_psnr >= 25:
                lines.append(f"  PSNR Grade:         GOOD (25-30 dB)")
            elif final_psnr >= 20:
                lines.append(f"  PSNR Grade:         FAIR (20-25 dB)")
            else:
                lines.append(f"  PSNR Grade:         POOR (< 20 dB)")

        if ssim_vals:
            final_ssim = ssim_vals[-1]
            if final_ssim >= 0.95:
                lines.append(f"  SSIM Grade:         EXCELLENT (>= 0.95)")
            elif final_ssim >= 0.90:
                lines.append(f"  SSIM Grade:         GOOD (0.90-0.95)")
            elif final_ssim >= 0.80:
                lines.append(f"  SSIM Grade:         FAIR (0.80-0.90)")
            else:
                lines.append(f"  SSIM Grade:         POOR (< 0.80)")
        lines.append("")

        # ── Learning Rate ────────────────────────────────────────────────────
        lrs = self.history["lr"]
        lines.append("--- Learning Rate ---")
        lines.append(f"  Initial LR:         {lrs[0]:.2e}")
        lines.append(f"  Final LR:           {lrs[-1]:.2e}")
        lines.append(f"  Min LR:             {min(lrs):.2e}")
        lines.append("")

        lines.append("=" * 70)
        lines.append(f"  CSV log:    {self.csv_path}")
        lines.append(f"  Plots dir:  {self.log_dir}")
        lines.append("=" * 70)

        summary_text = "\n".join(lines)

        # Write to file
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # Print to console
        print(summary_text)
