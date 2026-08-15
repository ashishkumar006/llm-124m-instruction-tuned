"""
Main Training Script for 124M Parameter Language Model.

This script handles:
1. Model initialization
2. Loading data (streaming from HuggingFace)
3. Training loop with:
   - Gradient accumulation for large effective batch sizes
   - Learning rate scheduling (warmup + cosine decay)
   - Time-based + step-based checkpointing (Lightning AI 4hr limit)
   - Auto-resume from last checkpoint on restart
   - Loss, perplexity, gradient norm, GPU memory tracking
   - CSV + JSON log files (persistent across sessions)
   - Terminal output saved to file (appended on resume)
   - Text sample generation at regular intervals
4. Validation
5. Model saving

Usage:
    python train.py              # Full training run (auto-detects GPU)
    python train.py --quick-test # Quick test run (50M tokens, ~5-10 min)

Supported platforms: Lightning AI, Google Colab, Kaggle, Local

Lightning AI resume:
    Just run `python train.py` again — it auto-loads the last checkpoint,
    appends to existing logs, and continues where it left off.
"""

import os
import sys
import csv
import json
import math
import time
import logging
import itertools
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW

# Import our modules
import config
from model import LanguageModel
from dataloader import get_data_loaders, get_validation_batches

# ---------------------------------------------------------------------------
# CLI flag: python train.py --quick-test
# ---------------------------------------------------------------------------
if "--quick-test" in sys.argv:
    config.QUICK_TEST = True
    config.total_tokens = 50_000_000
    config.save_interval = 100
    config.eval_interval = 50
    config.log_interval = 5
    config.shuffle_buffer_size = 100
    config.checkpoint_every_minutes = 10   # more frequent for testing
    config.sample_interval = 25            # generate samples often during test
    config.num_sample_prompts = 2
    config.dataloader_workers = 0          # workers add startup time; skip for test
    config.compile_model = False           # skip compilation overhead for test
    print("\n⚡ QUICK TEST MODE (--quick-test): 50M tokens, ~5-10 minutes\n")

# ============================================================================
# LOGGING SETUP — Console + File (appended on resume)
# ============================================================================

# Create log directory early so file handler can write to it
Path(config.log_dir).mkdir(parents=True, exist_ok=True)

_log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Root logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FILE HANDLER: saves ALL terminal output to logs/train_output.log
# Uses mode='a' so it APPENDS when you restart — full history preserved.
_file_handler = logging.FileHandler(
    Path(config.log_dir) / "train_output.log",
    mode='a',        # append, not overwrite!
    encoding='utf-8'
)
_file_handler.setFormatter(_log_fmt)
_file_handler.setLevel(logging.INFO)
logger.addHandler(_file_handler)

# Also attach to root logger so ALL libraries' output is captured
logging.getLogger().addHandler(_file_handler)

# Log session start marker
logger.info(f"\n{'#'*70}")
logger.info(f"# NEW SESSION STARTED: {datetime.now().isoformat()}")
logger.info(f"# Command: {' '.join(sys.argv)}")
logger.info(f"{'#'*70}")

# ============================================================================
# DEVICE & DISTRIBUTED SETUP
# ============================================================================

def setup_device():
    """Setup CUDA device and print info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️  No CUDA available. Training on CPU (will be slow).")
    
    return device

# ============================================================================
# GPU STATS LOGGER — Periodic GPU utilization/temperature/memory/power to file
# ============================================================================

class GPUStatsLogger:
    """
    Logs GPU utilisation, temperature, memory, power consumption, and system RAM to a CSV file.

    File: logs/gpu_stats.csv
    Columns: timestamp, step, gpu_util_pct, gpu_temp_c, gpu_power_w, mem_used_gb,
             mem_total_gb, mem_pct, sys_ram_used_gb, sys_ram_pct

    GPU Power: Requires nvidia-smi (W)
    System RAM: Uses psutil if available, otherwise logs empty
    Falls back gracefully if nvidia-smi or psutil is not available.
    """

    CSV_COLUMNS = [
        "timestamp", "step", "gpu_util_pct", "gpu_temp_c", "gpu_power_w",
        "mem_used_gb", "mem_total_gb", "mem_pct", "sys_ram_used_gb", "sys_ram_pct",
    ]

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.csv_path = self.log_dir / "gpu_stats.csv"
        self._has_nvidia_smi = self._check_nvidia_smi()
        self._has_psutil = self._check_psutil()

        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.CSV_COLUMNS)

    @staticmethod
    def _check_nvidia_smi() -> bool:
        """Check if nvidia-smi is available."""
        try:
            import subprocess
            subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    @staticmethod
    def _check_psutil() -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False

    def log(self, step: int = 0):
        """Log one GPU stats snapshot."""
        if not torch.cuda.is_available():
            return

        row = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "gpu_util_pct": "",
            "gpu_temp_c": "",
            "gpu_power_w": "",
            "mem_used_gb": f"{torch.cuda.memory_allocated() / 1e9:.2f}",
            "mem_total_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
            "mem_pct": f"{torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}",
            "sys_ram_used_gb": "",
            "sys_ram_pct": "",
        }

        # Try nvidia-smi for utilisation + temperature + power
        if self._has_nvidia_smi:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 2:
                        row["gpu_util_pct"] = parts[0].strip()
                        row["gpu_temp_c"]   = parts[1].strip()
                    if len(parts) >= 3:
                        row["gpu_power_w"]  = parts[2].strip()
            except Exception:
                pass

        # Try psutil for system RAM usage
        if self._has_psutil:
            try:
                import psutil
                ram = psutil.virtual_memory()
                row["sys_ram_used_gb"] = f"{ram.used / 1e9:.2f}"
                row["sys_ram_pct"] = f"{ram.percent:.1f}"
            except Exception:
                pass

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.CSV_COLUMNS).writerow(row)


# ============================================================================
# INFERENCE SAMPLE GENERATOR — Monitor model quality over training
# ============================================================================

class SampleGenerator:
    """
    Generates text samples at regular intervals during training.
    Saves to logs/samples.txt (appended on resume).

    This lets you visually monitor progress:
    - Early training: complete gibberish
    - Mid training: grammatical but nonsensical
    - Late training: coherent paragraphs
    """

    # Fixed prompts used every time — lets you compare across steps
    DEFAULT_PROMPTS = [
        "The meaning of life is",
        "In a groundbreaking scientific discovery,",
        "Once upon a time in a small village,",
        "The fundamental principles of machine learning",
        "India is a country known for",
    ]

    def __init__(self, log_dir: str, tokenizer, max_new_tokens: int = 100,
                 num_prompts: int = 3):
        self.log_dir = Path(log_dir)
        self.sample_path = self.log_dir / "samples.txt"
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompts = self.DEFAULT_PROMPTS[:num_prompts]

        # Log initial state
        if not self.sample_path.exists():
            with open(self.sample_path, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("  INFERENCE SAMPLES — Generated during training\n")
                f.write("  Prompts are fixed so you can compare across steps\n")
                f.write("=" * 70 + "\n\n")

    @torch.no_grad()
    def generate_samples(self, model: LanguageModel, step: int,
                         loss: float, device: torch.device) -> str:
        """
        Generate text from the model and save to file.

        Args:
            model: The language model
            step: Current training step
            loss: Current training loss
            device: torch device

        Returns:
            Formatted string with all generated samples
        """
        model.eval()

        header = (
            f"\n{'─'*70}\n"
            f"  Step {step} | Loss {loss:.4f} | PPL {math.exp(min(loss, 20)):.1f}"
            f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'─'*70}\n"
        )
        outputs = [header]

        for i, prompt in enumerate(self.prompts, 1):
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate
            try:
                generated = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.8,
                    top_k=50
                )
                text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            except Exception as e:
                text = f"[Generation failed: {e}]"

            entry = f"  Prompt {i}: \"{prompt}\"\n  Output : {text}\n"
            outputs.append(entry)

        result = "\n".join(outputs) + "\n"

        # Append to file
        with open(self.sample_path, "a", encoding="utf-8") as f:
            f.write(result)

        model.train()
        logger.info(f"  \U0001f4dd Saved {len(self.prompts)} text samples at step {step} -> {self.sample_path}")
        return result

# ============================================================================
# TRAINING LOGGER — Writes CSV + JSON for loss curves & resume
# ============================================================================

class TrainingLogger:
    """
    Persistent training logger that writes metrics to CSV and JSON files.

    Files created in log_dir:
        training_log.csv  — one row per log_interval (for plotting)
        training_meta.json — run metadata (config, GPU, timestamps)

    CSV columns:
        step, loss, perplexity, lr, tokens_per_sec, total_tokens,
        grad_norm, gpu_mem_used_gb, gpu_mem_total_gb, elapsed_sec,
        val_loss, val_perplexity

    Usage:
        tlog = TrainingLogger("./logs")
        tlog.log_step(step=100, loss=4.5, lr=6e-4, tok_sec=50000, ...)
        tlog.log_validation(step=100, val_loss=4.6)
    """

    CSV_COLUMNS = [
        "step", "loss", "perplexity", "lr", "tokens_per_sec",
        "total_tokens", "grad_norm", "gpu_mem_used_gb", "gpu_mem_total_gb",
        "elapsed_sec", "val_loss", "val_perplexity",
    ]

    def __init__(self, log_dir: str, config_module=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path  = self.log_dir / "training_log.csv"
        self.meta_path = self.log_dir / "training_meta.json"

        self.start_time = time.time()
        self.rows: List[Dict] = []  # in-memory buffer for quick access

        # If CSV already exists (resumed run), load existing rows
        if self.csv_path.exists():
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.rows.append(row)
            logger.info(f"📄 Loaded {len(self.rows)} existing log entries from {self.csv_path}")
        else:
            # Write CSV header
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
            logger.info(f"📄 Training log: {self.csv_path.resolve()}")

        # Write metadata JSON (overwritten each run)
        meta = {
            "start_time": datetime.now().isoformat(),
            "platform": Checkpoint.detect_platform(),
        }
        if config_module is not None:
            meta["config"] = {
                "hidden_dim": config_module.hidden_dim,
                "num_layers": config_module.num_layers,
                "num_attention_heads": config_module.num_attention_heads,
                "num_kv_heads": config_module.num_kv_heads,
                "max_seq_length": config_module.max_seq_length,
                "vocab_size": config_module.vocab_size,
                "total_tokens": config_module.total_tokens,
                "global_batch_size": config_module.global_batch_size,
                "peak_lr": config_module.peak_lr,
                "dtype": config_module.dtype,
                "seq_per_micro_batch": config_module.seq_per_micro_batch,
                "gradient_accumulation_steps": config_module.gradient_accumulation_steps,
            }
        if torch.cuda.is_available():
            meta["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def log_step(self, step: int, loss: float, lr: float,
                 tokens_per_sec: float, total_tokens: int,
                 grad_norm: float = 0.0) -> Dict:
        """Log one training step to CSV and return the row dict."""
        perplexity = math.exp(min(loss, 20))  # cap to avoid overflow

        gpu_used = gpu_total = 0.0
        if torch.cuda.is_available():
            gpu_used  = torch.cuda.memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        elapsed = time.time() - self.start_time

        row = {
            "step":             step,
            "loss":             f"{loss:.4f}",
            "perplexity":       f"{perplexity:.2f}",
            "lr":               f"{lr:.2e}",
            "tokens_per_sec":   f"{tokens_per_sec:.0f}",
            "total_tokens":     total_tokens,
            "grad_norm":        f"{grad_norm:.4f}",
            "gpu_mem_used_gb":  f"{gpu_used:.2f}",
            "gpu_mem_total_gb": f"{gpu_total:.1f}",
            "elapsed_sec":      f"{elapsed:.1f}",
            "val_loss":         "",
            "val_perplexity":   "",
        }
        self._append_row(row)
        return row

    def log_validation(self, step: int, val_loss: float):
        """Append validation metrics to the most recent row for this step."""
        val_ppl = math.exp(min(val_loss, 20))

        # Update the last row that matches this step
        for row in reversed(self.rows):
            if int(row["step"]) == step:
                row["val_loss"]       = f"{val_loss:.4f}"
                row["val_perplexity"] = f"{val_ppl:.2f}"
                break
        else:
            # Step not found — append a new row with just val metrics
            row = {col: "" for col in self.CSV_COLUMNS}
            row["step"]            = step
            row["val_loss"]        = f"{val_loss:.4f}"
            row["val_perplexity"]  = f"{val_ppl:.2f}"
            self.rows.append(row)

        # Rewrite entire CSV (small file, no performance issue)
        self._rewrite_csv()

    def _append_row(self, row: Dict):
        self.rows.append(row)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writerow(row)

    def _rewrite_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self) -> str:
        """Return a human-readable summary of the training run."""
        if not self.rows:
            return "No training data logged yet."
        last = self.rows[-1]
        elapsed = time.time() - self.start_time
        hours   = elapsed / 3600
        lines = [
            f"\n{'='*60}",
            f"  Training Summary",
            f"{'='*60}",
            f"  Steps completed : {last['step']}",
            f"  Final loss      : {last['loss']}",
            f"  Final perplexity: {last['perplexity']}",
            f"  Total tokens    : {int(last['total_tokens']):,}",
            f"  Wall time       : {hours:.2f} hours",
            f"  Log file        : {self.csv_path.resolve()}",
        ]
        if last.get("val_loss"):
            lines.append(f"  Last val loss   : {last['val_loss']}")
            lines.append(f"  Last val ppl    : {last['val_perplexity']}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate(model: LanguageModel, val_batches: list, device: torch.device,
             max_batches: int = 20, dtype=torch.float32,
             use_mixed_precision: bool = True) -> float:
    """
    Run validation on cached batches and return average loss.

    Uses pre-cached validation batches (from get_validation_batches) so
    evaluation is fast, consistent, and separate from training data.

    Args:
        model: The language model
        val_batches: List of cached validation batch dicts
        device: torch device
        max_batches: Max batches to evaluate (safety limit)
        dtype: AMP dtype
        use_mixed_precision: Whether to use AMP

    Returns:
        Average cross-entropy loss over evaluated batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_batches:
        if num_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets   = batch["targets"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_mixed_precision):
            _, loss = model(input_ids, targets)

        total_loss += loss.item()
        num_batches += 1

    model.train()

    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================

def get_lr_schedule(step: int, total_steps: int,
                   peak_lr: float, min_lr: float, warmup_steps: int,
                   decay_start_step: int = None) -> float:
    """
    Learning rate schedule with warmup and cosine decay.

    Schedule:
    1. Warmup  (0  → warmup_steps)        : linear 0 -> peak_lr
    2. Stable  (warmup → decay_start_step) : constant peak_lr
    3. Decay   (decay_start_step → end)    : cosine decay to min_lr
    """
    # Phase 1: Warmup
    if step < warmup_steps:
        return peak_lr * (step / max(warmup_steps, 1))

    # Default: decay starts right after warmup if not specified
    if decay_start_step is None:
        decay_start_step = warmup_steps

    # Phase 2: Stable (constant peak LR)
    if step < decay_start_step:
        return peak_lr

    # Phase 3: Cosine decay from peak_lr to min_lr
    decay_steps = max(total_steps - decay_start_step, 1)
    decay_progress = (step - decay_start_step) / decay_steps
    decay_progress = min(decay_progress, 1.0)  # Clamp at 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_progress))  # 1.0 → 0.0
    return min_lr + (peak_lr - min_lr) * coeff


def set_learning_rate(optimizer: AdamW, lr: float):
    """Update learning rate for all param groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class Checkpoint:
    """
    Handles saving and loading model checkpoints.
    Supports local disk, Kaggle /kaggle/working, and Google Drive (Colab).
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints", max_checkpoints: int = 30):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        logger.info(f"📁 Checkpoint directory: {self.checkpoint_dir.resolve()}")
        logger.info(f"   Max checkpoints kept on disk: {self.max_checkpoints}")

    @staticmethod
    def detect_platform() -> str:
        """Auto-detect if running on Lightning AI, Kaggle, Colab, or local."""
        # Lightning AI Studios
        if (os.path.exists("/teamspace")
                or os.environ.get("LIGHTNING_CLOUD_URL")
                or os.environ.get("LIGHTNING_STUDIO_ID")):
            return "lightning"
        # Kaggle
        if os.path.exists("/kaggle/working"):
            return "kaggle"
        # Colab
        try:
            import google.colab  # noqa
            return "colab"
        except ImportError:
            pass
        return "local"

    @staticmethod
    def mount_google_drive(mount_path: str = "/content/drive") -> Optional[str]:
        """
        Mount Google Drive inside Colab.
        Returns the drive path if successful, None otherwise.

        HOW TO USE IN COLAB:
        ---------------------
        The Checkpoint class calls this automatically.
        Just set checkpoint_dir to a Google Drive path:

            from train import Checkpoint
            ckpt = Checkpoint("/content/drive/MyDrive/llm_checkpoints")

        Or manually:
            ckpt_path = Checkpoint.mount_google_drive()
            if ckpt_path:
                ckpt = Checkpoint(ckpt_path + "/llm_checkpoints")
        """
        try:
            from google.colab import drive
            drive.mount(mount_path)
            drive_path = mount_path + "/MyDrive"
            logger.info(f"✅ Google Drive mounted at {mount_path}")
            return drive_path
        except Exception as e:
            logger.warning(f"⚠️  Could not mount Google Drive: {e}")
            return None

    def save(self, model: LanguageModel, optimizer: AdamW, step: int,
             loss: float, scaler=None, total_loss: float = 0.0,
             micro_step_global: int = 0, wall_time_sec: float = 0.0):
        """
        Save checkpoint with ALL state needed for perfect resume.

        Saves:
        - Model weights, optimizer state, scaler state
        - Training step counters (opt_step, micro_step)
        - Accumulated loss, wall time
        - Timestamp for debugging

        Files:
        - checkpoint_step_{step}.pt  — named checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        checkpoint = {
            "step": step,
            "loss": loss,
            "total_loss": total_loss,
            "micro_step_global": micro_step_global,
            "wall_time_sec": wall_time_sec,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✅ Saved checkpoint at step {step} -> {checkpoint_path}")
        logger.info(f"   (total_loss={total_loss:.4f}, micro_step={micro_step_global}, wall={wall_time_sec:.0f}s)")

        # Clean up old checkpoints, keep only the most recent N
        self._cleanup_old_checkpoints()

        # On Kaggle: also copy to /kaggle/working so it persists after session
        platform = self.detect_platform()
        if platform == "kaggle":
            import shutil
            kaggle_out = Path("/kaggle/working/checkpoints")
            kaggle_out.mkdir(exist_ok=True)
            shutil.copy2(checkpoint_path, kaggle_out / f"checkpoint_step_{step}.pt")
            logger.info(f"   Kaggle copy saved to /kaggle/working/checkpoints/")

    def _cleanup_old_checkpoints(self):
        """
        Delete old named checkpoints, keeping only the most recent N.
        Never deletes latest.pt (used for auto-resume).
        """
        # Find all checkpoint_step_*.pt files
        ckpt_files = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        if len(ckpt_files) > self.max_checkpoints:
            to_delete = ckpt_files[:len(ckpt_files) - self.max_checkpoints]
            for f in to_delete:
                f.unlink()
                logger.info(f"🗑️  Deleted old checkpoint: {f.name}")
            logger.info(f"   Kept {self.max_checkpoints} most recent checkpoints")

    def load_latest(self, model: LanguageModel, optimizer: AdamW,
                    scaler=None) -> Dict:
        """
        Load the latest valid checkpoint_step_*.pt file.

        Finds all checkpoint_step_*.pt files, picks the one with the
        highest step number, and loads it. If corrupted, tries the next.

        Returns dict with keys:
            step, loss, total_loss, micro_step_global, wall_time_sec
        Or empty dict if no checkpoint found.
        """
        import re

        # Collect checkpoint_step_*.pt files from all search dirs
        search_dirs = [self.checkpoint_dir]
        platform = self.detect_platform()
        if platform == "kaggle":
            kaggle_dir = Path("/kaggle/working/checkpoints")
            if kaggle_dir.exists():
                search_dirs.insert(0, kaggle_dir)

        all_ckpts = []
        for d in search_dirs:
            for f in d.glob("checkpoint_step_*.pt"):
                m = re.search(r"checkpoint_step_(\d+)\.pt$", f.name)
                if m:
                    all_ckpts.append((int(m.group(1)), f))

        # Sort by step number descending (newest first)
        all_ckpts.sort(key=lambda x: x[0], reverse=True)

        if not all_ckpts:
            return {}

        # Try loading from newest to oldest, skip any corrupted files
        for step_num, ckpt_path in all_ckpts:
            try:
                logger.info(f"📥 Loading checkpoint: {ckpt_path.name} (step {step_num})")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if scaler is not None and "scaler_state_dict" in ckpt:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])

                step = ckpt["step"]
                saved_ts = ckpt.get("timestamp", "unknown")
                logger.info(f"✅ Resumed from step {step} (saved at {saved_ts})")
                return {
                    "step": step,
                    "loss": ckpt.get("loss", float('inf')),
                    "total_loss": ckpt.get("total_loss", 0.0),
                    "micro_step_global": ckpt.get("micro_step_global", step * config.gradient_accumulation_steps),
                    "wall_time_sec": ckpt.get("wall_time_sec", 0.0),
                }
            except Exception as e:
                logger.warning(f"⚠️  Checkpoint {ckpt_path.name} is corrupted: {e}")
                logger.warning(f"   Skipping, trying next checkpoint...")
                continue

        logger.warning("⚠️  All checkpoints corrupted! Starting fresh.")
        return {}


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(
    model: LanguageModel,
    train_loader,
    optimizer: AdamW,
    device: torch.device,
    config,
    num_training_steps: int,
    checkpoint_manager: Checkpoint,
    use_mixed_precision: bool = True,
    training_logger: Optional[TrainingLogger] = None,
    gpu_logger: Optional[GPUStatsLogger] = None,
    sample_generator: Optional[SampleGenerator] = None,
    val_batches: Optional[list] = None,
):
    """
    Main training loop with:
    - Gradient accumulation (fixed)
    - Time-based checkpointing every N minutes (Lightning AI safety)
    - Step-based checkpointing every save_interval steps
    - Validation every eval_interval steps
    - Text sample generation every sample_interval steps
    - GPU stats logging
    - Full auto-resume from checkpoint

    CHECKPOINT STRATEGY:
    Lightning AI can kill your session after 4 hours.
    We checkpoint every 30 minutes (configurable) so you lose at most
    30 minutes of work. On restart, `python train.py` auto-resumes.
    """
    # Use torch.amp (not deprecated torch.cuda.amp)
    # Dtype depends on GPU: T4 needs float16, modern GPUs use bfloat16
    if config.dtype == "float16":
        amp_dtype = torch.float16
    elif config.dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    if use_mixed_precision and amp_dtype == torch.bfloat16:
        # Safety check: fall back to float16 if bfloat16 not supported
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("bfloat16 not supported on this GPU, falling back to float16")
            amp_dtype = torch.float16
            config.dtype = "float16"

    dtype = amp_dtype if use_mixed_precision else torch.float32
    # GradScaler: essential for float16 (prevents underflow).
    # For bfloat16 it's a harmless no-op.
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_mixed_precision)

    model = model.to(device)
    model.train()

    # ── Resume from checkpoint ──────────────────────────────────────
    ckpt_data = checkpoint_manager.load_latest(model, optimizer, scaler)
    if ckpt_data:
        start_opt_step      = ckpt_data["step"]
        total_loss           = ckpt_data["total_loss"]
        micro_step_global    = ckpt_data["micro_step_global"]
        prev_wall_time       = ckpt_data["wall_time_sec"]
        logger.info(f"  Resuming: opt_step={start_opt_step}, "
                    f"total_loss={total_loss:.4f}, prev_wall={prev_wall_time:.0f}s")
    else:
        start_opt_step    = 0
        total_loss        = 0.0
        micro_step_global = 0
        prev_wall_time    = 0.0
        logger.info("  Starting fresh training (no checkpoint found).")

    window_loss = []           # losses over the last log_interval opt steps
    t_last      = time.time()  # for tokens/sec calculation
    tokens_since_log = 0
    last_grad_norm   = 0.0     # track gradient norm for logging
    train_start_time = time.time()  # wall clock for this session
    last_ckpt_time   = time.time()  # for time-based checkpointing

    acc_steps = config.gradient_accumulation_steps
    ckpt_interval_sec = getattr(config, 'checkpoint_every_minutes', 30) * 60
    sample_interval   = getattr(config, 'sample_interval', 200)

    logger.info(f"\n{'='*70}")
    logger.info(f"  Starting training  opt_step={start_opt_step}/{num_training_steps}")
    logger.info(f"  Gradient accumulation  : {acc_steps} micro-steps per opt-step")
    logger.info(f"  Mixed precision        : {use_mixed_precision} (dtype={dtype})")
    logger.info(f"  Time-based checkpoint  : every {ckpt_interval_sec//60:.0f} minutes")
    logger.info(f"  Step-based checkpoint  : every {config.save_interval} steps")
    if val_batches:
        logger.info(f"  Validation             : every {config.eval_interval} steps ({len(val_batches)} cached batches)")
    else:
        logger.info(f"  Validation             : disabled (no val batches)")
    logger.info(f"  Sample generation      : every {sample_interval} steps")
    logger.info(f"  Previous wall time     : {prev_wall_time/3600:.2f} hours")
    logger.info(f"{'='*70}")

    opt_step = start_opt_step
    optimizer.zero_grad()

    # ── Fast-forward dataloader on resume ───────────────────────────
    # Streaming datasets have no random access, so when we resume we
    # must consume (and discard) the batches that were already used in
    # the previous session.  This guarantees that the model never sees
    # the same data twice — critical when resuming many times.
    #
    # Cost: ~1-5 minutes of CPU work per 1000 skipped micro-steps
    # (no GPU used, just streaming + tokenizing).
    #
    # HOW IT WORKS:
    # 1. We seeded the shuffle with config.seed (fixed)
    # 2. DataLoader iterates in deterministic order
    # 3. Fast-forward pulls the same batches in the same order
    # 4. Training loop continues from exact position, zero overlap
    micro_steps_to_skip = micro_step_global  # restored from checkpoint
    if micro_steps_to_skip > 0:
        logger.info(f"\n⏩ Resuming: fast-forwarding dataloader past {micro_steps_to_skip:,} "
                    f"micro-steps already consumed...")
        logger.info(f"   (This happens because streaming datasets can't random-access)")
        skip_start = time.time()
        
        # Use itertools.islice for efficient skipping
        skipped = 0
        for i, _batch in enumerate(itertools.islice(train_loader, micro_steps_to_skip)):
            skipped = i + 1
            if skipped % 500 == 0 or skipped == micro_steps_to_skip:
                elapsed = time.time() - skip_start
                rate = skipped / max(elapsed, 0.1)
                eta_sec = (micro_steps_to_skip - skipped) / max(rate, 1)
                logger.info(f"   ... {skipped:,}/{micro_steps_to_skip:,} batches "
                           f"({elapsed:.1f}s, ETA: {eta_sec:.0f}s)")
        
        skip_elapsed = time.time() - skip_start
        logger.info(f"   ✅ Fast-forwarded {skipped:,} batches in {skip_elapsed:.1f}s "
                    f"({skipped/max(skip_elapsed,1):.0f} batches/s)")
        logger.info(f"   Data stream is now at the correct position (deterministic shuffle)")
        logger.info(f"   Resuming training with ZERO data overlap.\n")

    def _save_checkpoint(reason: str):
        """Helper to save checkpoint with all state."""
        wall = prev_wall_time + (time.time() - train_start_time)
        avg_loss = total_loss / max(opt_step, 1)
        checkpoint_manager.save(
            model, optimizer, opt_step, avg_loss,
            scaler=scaler,
            total_loss=total_loss,
            micro_step_global=micro_step_global,
            wall_time_sec=wall,
        )
        logger.info(f"   Reason: {reason} | Wall time: {wall/3600:.2f}h")

    try:
        for batch in train_loader:
            if opt_step >= num_training_steps:
                break

            # --- micro-step (forward + backward) ---
            input_ids = batch["input_ids"].to(device)
            targets   = batch["targets"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_mixed_precision):
                _, loss = model(input_ids, targets)
                # Scale loss by accumulation steps so gradient magnitude
                # is the same as computing it over the full batch at once.
                loss = loss / acc_steps

            scaler.scale(loss).backward()

            tokens_since_log    += input_ids.numel()
            micro_step_global   += 1

            # --- optimizer step (every acc_steps micro-steps) ---
            if micro_step_global % acc_steps == 0:
                if config.use_grad_clip:
                    scaler.unscale_(optimizer)
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip_norm
                    ).item()
                else:
                    last_grad_norm = 0.0  # skip expensive norm when not clipping

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update learning rate (opt_step+1 so LR aligns with logged step)
                lr = get_lr_schedule(
                    opt_step + 1, num_training_steps,
                    config.peak_lr, config.min_lr, config.warmup_steps,
                    decay_start_step=config.lr_decay_start_step
                )
                set_learning_rate(optimizer, lr)

                # Tracking (use unscaled loss for display)
                real_loss = loss.item() * acc_steps
                total_loss  += real_loss
                window_loss.append(real_loss)

                # Logging
                if (opt_step + 1) % config.log_interval == 0:
                    avg_loss = sum(window_loss) / len(window_loss)
                    elapsed  = time.time() - t_last
                    tok_sec  = tokens_since_log / max(elapsed, 1e-6)
                    total_tok = (opt_step + 1) * config.global_batch_size
                    ppl      = math.exp(min(avg_loss, 20))  # cap to avoid overflow
                    session_time = time.time() - train_start_time
                    total_wall   = prev_wall_time + session_time

                    # GPU memory
                    gpu_mem_str = ""
                    if torch.cuda.is_available():
                        used  = torch.cuda.memory_allocated() / 1e9
                        total = torch.cuda.get_device_properties(0).total_memory / 1e9
                        gpu_mem_str = f"| mem {used:.1f}/{total:.0f}GB "

                    logger.info(
                        f"  step {opt_step+1:6d}/{num_training_steps} "
                        f"| loss {avg_loss:.4f} "
                        f"| ppl {ppl:.1f} "
                        f"| lr {lr:.2e} "
                        f"| grad {last_grad_norm:.2f} "
                        f"| {tok_sec/1e3:.1f}k tok/s "
                        f"{gpu_mem_str}"
                        f"| {total_tok/1e9:.3f}B tok "
                        f"| wall {total_wall/3600:.2f}h"
                    )

                    # Write to CSV log
                    if training_logger is not None:
                        training_logger.log_step(
                            step=opt_step + 1,
                            loss=avg_loss,
                            lr=lr,
                            tokens_per_sec=tok_sec,
                            total_tokens=total_tok,
                            grad_norm=last_grad_norm,
                        )

                    # GPU stats to separate CSV
                    if gpu_logger is not None:
                        gpu_logger.log(step=opt_step + 1)

                    window_loss      = []
                    t_last           = time.time()
                    tokens_since_log = 0

                # Validation (on cached held-out batches)
                if val_batches and (opt_step + 1) % config.eval_interval == 0:
                    val_loss = evaluate(
                        model, val_batches, device,
                        max_batches=20, dtype=dtype,
                        use_mixed_precision=use_mixed_precision,
                    )
                    val_ppl = math.exp(min(val_loss, 20))
                    logger.info(
                        f"  ✅ EVAL step {opt_step+1} "
                        f"| val_loss {val_loss:.4f} "
                        f"| val_ppl {val_ppl:.1f}"
                    )
                    if training_logger is not None:
                        training_logger.log_validation(opt_step + 1, val_loss)

                # Text sample generation
                if sample_generator is not None and (opt_step + 1) % sample_interval == 0:
                    current_loss = total_loss / max(opt_step + 1, 1)
                    sample_generator.generate_samples(
                        model, opt_step + 1, current_loss, device
                    )

                # Step-based checkpointing
                if (opt_step + 1) % config.save_interval == 0:
                    _save_checkpoint(f"step-based (every {config.save_interval} steps)")

                # Time-based checkpointing (Lightning AI safety net)
                if time.time() - last_ckpt_time >= ckpt_interval_sec:
                    _save_checkpoint(f"time-based (every {ckpt_interval_sec//60:.0f} min)")
                    last_ckpt_time = time.time()

                opt_step += 1

    except KeyboardInterrupt:
        logger.info(f"\n⚠️  Interrupted at opt_step={opt_step}")
        _save_checkpoint("keyboard interrupt")

    # Final checkpoint
    logger.info("\n✅ Training complete!")
    _save_checkpoint("training complete")

    # Generate final samples
    if sample_generator is not None:
        current_loss = total_loss / max(opt_step, 1)
        sample_generator.generate_samples(model, opt_step, current_loss, device)

    # Print training summary
    if training_logger is not None:
        logger.info(training_logger.summary())


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training entry point."""

    # Auto-detect platform and set checkpoint dir
    platform = Checkpoint.detect_platform()
    logger.info(f"\n🖥️  Detected platform: {platform}")

    if platform == "lightning":
        # Lightning AI: files persist across sessions — just use local dir
        ckpt_dir = config.checkpoint_dir
        logger.info(f"   Lightning AI: checkpoints saved to {ckpt_dir} (persists across sessions!)")
    elif platform == "colab":
        drive_root = Checkpoint.mount_google_drive()
        if drive_root:
            ckpt_dir = drive_root + "/llm_124m_checkpoints"
            logger.info(f"   Colab: saving checkpoints to Google Drive at {ckpt_dir}")
        else:
            ckpt_dir = config.checkpoint_dir
            logger.warning("   Colab: Drive mount failed, saving locally (will be lost on disconnect!)")
    elif platform == "kaggle":
        ckpt_dir = "/kaggle/working/checkpoints"
        logger.info(f"   Kaggle: saving to /kaggle/working (persists in notebook output)")
    else:
        ckpt_dir = config.checkpoint_dir
        logger.info(f"   Local: saving to {ckpt_dir}")

    # Auto-detect GPU and configure batch size / dtype
    detected_gpu = config.configure_for_gpu()  # auto-detects GPU model
    if detected_gpu:
        logger.info(f"   GPU profile applied: {detected_gpu}")
    else:
        logger.info(f"   Using default config (no GPU profile matched)")

    max_ckpts = getattr(config, 'max_checkpoints', 30)
    checkpoint_manager = Checkpoint(ckpt_dir, max_checkpoints=max_ckpts)

    # Initialize training logger (CSV + JSON metrics)
    training_logger = TrainingLogger(config.log_dir, config_module=config)

    # Initialize GPU stats logger
    gpu_logger = GPUStatsLogger(config.log_dir)

    # Setup device early (needed for torch.compile decision)
    device = setup_device()

    # Reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"  Random seed: {config.seed}")

    # Enable TF32 on Ampere+ GPUs (free precision-preserving speedup)
    torch.set_float32_matmul_precision('high')

    logger.info(f"\n{'='*70}")
    logger.info(f"  124M LLM Training Run")
    logger.info(f"{'='*70}")

    # Initialize model
    logger.info("\n📦 Initializing model...")
    model = LanguageModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))  # after weight tying
    logger.info(f"  Total param tensors : {total_params/1e6:.1f}M")
    logger.info(f"  Unique params (tied): {unique_params/1e6:.1f}M")

    # Compile model for faster training (PyTorch 2.0+)
    # First ~2 minutes will be slow (JIT compilation), then ~20-30% faster.
    if getattr(config, 'compile_model', True) and hasattr(torch, 'compile') and device.type == 'cuda':
        logger.info("  ⚡ Compiling model with torch.compile (first ~2 min will be slow)...")
        model = torch.compile(model)

    # Optimizer — separate weight decay parameters
    # Rule: don't apply weight decay to bias / norm params (they're 1D or are scale params)
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params,    "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=config.peak_lr,
                      betas=(config.beta1, config.beta2), eps=config.eps)
    logger.info(f"  Optimizer: AdamW | decay params={len(decay_params)} | no-decay={len(no_decay_params)}")

    # Load data (streaming from HuggingFace — no download needed)
    logger.info("\n📊 Loading data (streaming from HuggingFace)...")
    train_loader, _ = get_data_loaders(
        config,
        batch_size=config.seq_per_micro_batch,
    )

    # Note: integer division drops remainder (up to global_batch_size-1 tokens).
    # Not harmful — just be aware when comparing total_tokens vs actual.
    num_training_steps = config.total_tokens // config.global_batch_size
    logger.info(f"  Training steps: {num_training_steps:,}")

    # Cache fixed validation batches (separate stream from training data)
    logger.info("\n📊 Caching validation batches...")
    val_batches = get_validation_batches(config, batch_size=config.seq_per_micro_batch)

    # Initialize sample generator (generates text at regular intervals)
    from dataloader import get_tokenizer
    tokenizer = get_tokenizer("gpt2")
    sample_generator = SampleGenerator(
        log_dir=config.log_dir,
        tokenizer=tokenizer,
        max_new_tokens=getattr(config, 'max_gen_length', 100),
        num_prompts=getattr(config, 'num_sample_prompts', 3),
    )

    logger.info(f"\n📁 Output files:")
    logger.info(f"   Checkpoints  : {ckpt_dir}/")
    logger.info(f"   Training CSV : {config.log_dir}/training_log.csv")
    logger.info(f"   Terminal log : {config.log_dir}/train_output.log")
    logger.info(f"   GPU stats    : {config.log_dir}/gpu_stats.csv")
    logger.info(f"   Text samples : {config.log_dir}/samples.txt")
    logger.info(f"   Run metadata : {config.log_dir}/training_meta.json")

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        config=config,
        num_training_steps=num_training_steps,
        checkpoint_manager=checkpoint_manager,
        use_mixed_precision=config.use_mixed_precision,
        training_logger=training_logger,
        gpu_logger=gpu_logger,
        sample_generator=sample_generator,
        val_batches=val_batches,
    )


if __name__ == "__main__":
    main()
