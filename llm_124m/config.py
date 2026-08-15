"""
Configuration file for the 124M Parameter Transformer Model.
All hyperparameters and architectural settings are defined here.
"""

# ============================================================================
# ARCHITECTURE CONFIGURATION
# ============================================================================

# Model dimensions
hidden_dim = 768              # d_model: The main embedding dimension
num_layers = 12               # Number of transformer blocks
num_attention_heads = 12      # Number of attention heads (each 64-dim)
num_kv_heads = 4              # KV heads for Grouped Query Attention (GQA)
ffn_expansion = 4             # FFN hidden dim = hidden_dim * ffn_expansion * 2/3 for SwiGLU
vocab_size = 50304            # Vocabulary size (multiple of 64 for GPU efficiency)
# CONTEXT LENGTH: You can safely increase this!
# - 1024 : Standard GPT-2 size. Fast, works on 6GB GPU.
# - 2048 : 2x context. Needs ~2x VRAM for attention.
# - 4096 : Llama-style. Needs A100 or gradient checkpointing.
# RoPE means the model can be used at inference with LONGER contexts
# than it was trained on (with slight quality drop).
max_seq_length = 2048         # Context window. 2048 is a good balance for 2026.

# Compute dimensions (derived)
ffn_hidden_dim = int(hidden_dim * ffn_expansion * 2 / 3)  # SwiGLU formula
head_dim = hidden_dim // num_attention_heads  # Each head dimension (should be 64)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Learning rate schedule
peak_lr = 6e-4                # Peak learning rate
min_lr = peak_lr * 0.1        # Minimum LR after decay
warmup_steps = 2000           # Steps to linearly increase LR to peak
lr_decay_start_step = 3675    # Step at which cosine LR decay begins
total_tokens = 4_000_000_000 # 4 billion tokens (Chinchilla-optimal for 124M)

# Optimizer settings
beta1 = 0.9                   # AdamW: exponential decay for 1st moment
beta2 = 0.95                  # AdamW: exponential decay for 2nd moment
weight_decay = 0.1            # L2 regularization
eps = 1e-8                    # Numerical stability for AdamW

# Batch configuration
global_batch_size = 524_288   # Total tokens per gradient update (~512K, power of 2)
seq_per_micro_batch = 4       # Number of SEQUENCES per forward pass on your 4050/P100
                               # (actual tokens = seq_per_micro_batch * max_seq_length)
# BUG FIX: old formula used max_micro_batch_size as tokens, but it's the number
# of sequences. The correct formula is:
# grad_accum = global_batch_tokens / (sequences_per_step * tokens_per_sequence)
gradient_accumulation_steps = global_batch_size // (seq_per_micro_batch * max_seq_length)

# Gradient Checkpointing: Trade compute for memory.
# Set True if you're running out of GPU memory.
# Effect: saves ~60% activation memory, costs ~33% extra compute.
use_gradient_checkpointing = False  # Set True on 6GB GPU with context > 1024

# Training dynamics
use_grad_clip = True          # Gradient clipping (CRITICAL: prevents exploding gradients)
grad_clip_norm = 1.0          # Clip gradients to this L2 norm
use_mixed_precision = True    # Use bfloat16 (2x speed, half memory, same quality)

# Quick test mode (reduces everything to test that the code actually runs)
# Set True when running first time to verify no crashes before long training!
QUICK_TEST = False            # ← SET TO True FOR FIRST TEST RUN

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Dataset settings
dataset_name = "HuggingFaceFW/fineweb-edu"  # FineWeb-Edu from HuggingFace
dataset_config = "sample-10BT"               # 10 billion token sample
train_split = "train"
use_streaming = True          # Stream data instead of downloading all at once
shuffle_buffer_size = 10000   # Size of shuffle buffer for streaming

dataloader_workers = 2        # Parallel data-loading workers (0=serial, 2=recommended)
                               # CRITICAL for throughput! Without this the GPU starves
                               # waiting for the CPU to tokenize the next batch.

# ============================================================================
# CHECKPOINTING & LOGGING
# ============================================================================

checkpoint_dir = "./checkpoints"
log_dir = "./logs"
save_interval = 100           # Save checkpoint every N steps (non-interruptible: no need for frequent saves)
eval_interval = 100           # Evaluate every N steps
log_interval = 10             # Log metrics every N steps
max_checkpoints = 30          # Keep only the N most recent checkpoint files on disk
checkpoint_every_minutes = 30 # TIME-BASED checkpoint (Lightning AI 4hr limit!)
sample_interval = 200         # Generate text samples every N steps to monitor quality
num_sample_prompts = 3        # How many prompts to generate per sample interval

# ============================================================================
# HARDWARE & COMPUTATION
# ============================================================================

device = "cuda"               # Use GPU (cuda) or CPU (cpu)
dtype = "bfloat16"            # "bfloat16" (A100/H100/H200/L40S) or "float16" (T4)
                              # auto_detect_gpu() below will set this correctly!
seed = 42                     # Random seed for reproducibility
compile_model = True          # torch.compile: ~20-30% speedup (PyTorch 2.0+, CUDA only)
                               # First ~2 min of training is slow (JIT compilation), then fast.
num_devices = 1               # Number of GPUs (single GPU setup)

# ============================================================================
# INFERENCE & GENERATION (for later use)
# ============================================================================

max_gen_length = 256          # Max tokens to generate
temperature = 0.7             # Sampling temperature
top_k = 50                    # Top-K sampling
top_p = 0.9                   # Top-P (nucleus) sampling

# ============================================================================
# QUICK TEST MODE (used by test_model.py and first sanity check)
# ============================================================================
if QUICK_TEST:
    total_tokens = 50_000_000        # Only 50M tokens (finishes in ~1 min on GPU)
    save_interval = 100
    eval_interval = 50
    log_interval = 5
    shuffle_buffer_size = 100
    checkpoint_every_minutes = 10    # More frequent for testing
    sample_interval = 25             # Generate samples often during test
    num_sample_prompts = 2
    dataloader_workers = 0           # Workers add startup time; skip for quick test
    compile_model = False            # Skip compilation overhead for quick test

# ============================================================================
# CALCULATED METRICS (for reference)
# ============================================================================

# Approx parameter count (weight-tied: lm_head shares token_embed weights)
# So we do NOT count lm_head separately
_attn_params_per_layer  = (
    hidden_dim * (num_attention_heads * head_dim) +  # Q
    hidden_dim * (num_kv_heads * head_dim) +         # K
    hidden_dim * (num_kv_heads * head_dim) +         # V
    (num_attention_heads * head_dim) * hidden_dim    # out_proj
)
_ffn_params_per_layer   = hidden_dim * ffn_hidden_dim * 3   # gate, up, down
_norm_params_per_layer  = hidden_dim * 2                    # 2 RMSNorm per block
_total_block_params     = num_layers * (_attn_params_per_layer + _ffn_params_per_layer + _norm_params_per_layer)
_embed_params           = vocab_size * hidden_dim            # token_embed (tied with lm_head)
total_parameters        = _embed_params + _total_block_params + hidden_dim  # +final norm

print(f"\n{'=' * 60}")
print(f"[==] 124M LLM Configuration Summary")
print(f"{'=' * 60}")
print(f"  Architecture:")
print(f"    Layers         : {num_layers}")
print(f"    Hidden dim     : {hidden_dim}")
print(f"    Attention heads: {num_attention_heads} Q / {num_kv_heads} KV (GQA {num_attention_heads//num_kv_heads}:1)")
print(f"    Context length : {max_seq_length}")
print(f"    FFN hidden     : {ffn_hidden_dim} (SwiGLU)")
print(f"    ~Parameters    : {total_parameters/1e6:.1f}M  (weight-tied)")
print(f"  Training:")
print(f"    Dtype          : {dtype}")
print(f"    Target tokens  : {total_tokens/1e9:.1f}B")
print(f"    Global batch   : {global_batch_size:,} tokens")
print(f"    Micro-batch    : {seq_per_micro_batch} seqs × {max_seq_length} = {seq_per_micro_batch*max_seq_length:,} tok")
print(f"    Grad accum     : {gradient_accumulation_steps} steps")
print(f"    Training steps : {int(total_tokens // global_batch_size):,}")
print(f"═" * 60 + "\n")


# ============================================================================
# GPU PROFILES FOR LIGHTNING AI
# ============================================================================
# Each GPU has different VRAM, so we need different micro-batch sizes.
# The GLOBAL batch size stays the same (512K tokens) — only the number
# of gradient accumulation steps changes.
#
# Learning rate does NOT change per GPU because the effective batch
# is always 512K tokens. LR scaling only matters if you change the
# global batch size.

GPU_PROFILES = {
    "T4": {
        "vram_gb": 16,
        "seq_per_micro_batch": 12,      # 12 × 2048 = 24K tokens/micro-step
        "dtype": "float16",             # T4 (Turing) doesn't support bfloat16!
        "use_gradient_checkpointing": True,  # ON: Saves activation memory when needed
        "free_hours": 80,
        "notes": "Best value: 80 free hours.",
    },
    "L4": {
        "vram_gb": 24,
        "seq_per_micro_batch": 8,       # 8 × 2048 = 16K tokens/micro-step
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        "free_hours": 31,
        "notes": "Good balance of speed and free time.",
    },
    "L40S": {
        "vram_gb": 48,
        "seq_per_micro_batch": 24,      # 24 × 2048 = 49K tokens/micro-step
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        "free_hours": 5,
        "notes": "Fast but only 5 hours. Good for short intensive runs.",
    },
    "A100": {
        "vram_gb": 40,                  # 40GB or 80GB variant
        "seq_per_micro_batch": 16,      # 16 × 2048 = 32K tokens/micro-step
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        "free_hours": 3,
        "notes": "Only 3 hours free. Use for fast experiments.",
    },
    "H100": {
        "vram_gb": 80,
        "seq_per_micro_batch": 48,      # 48 × 2048 = 98K tokens/micro-step
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        "free_hours": 1,
        "notes": "Only 1 hour! Great for quick verification.",
    },
    "H200": {
        "vram_gb": 141,
        "seq_per_micro_batch": 64,      # 64 × 2048 = 131K tokens/micro-step
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        "free_hours": 4,
        "notes": "Fastest GPU. 4 hours can do ~5B tokens.",
    },
}


def auto_detect_gpu():
    """
    Detect the GPU model and return the matching profile name.
    Returns None if no GPU or no matching profile.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        gpu_name = torch.cuda.get_device_name(0).upper()
        # Check profiles in order (most specific first)
        for profile_name in ["H200", "H100", "A100", "L40S", "L4", "T4"]:
            if profile_name.upper() in gpu_name:
                return profile_name
        return None
    except Exception:
        return None


def configure_for_gpu(gpu_name=None):
    """
    Apply GPU-specific settings to this config module.

    Call this BEFORE creating the model/optimizer/dataloader.
    Changes: seq_per_micro_batch, dtype, gradient_accumulation_steps,
             use_gradient_checkpointing.

    Args:
        gpu_name: One of "T4", "L4", "L40S", "A100", "H100", "H200".
                  If None, auto-detects from CUDA device.

    Returns:
        str: The GPU name that was configured (or None if no match).
    """
    global seq_per_micro_batch, dtype, gradient_accumulation_steps
    global use_gradient_checkpointing

    if gpu_name is None:
        gpu_name = auto_detect_gpu()

    if gpu_name is None or gpu_name not in GPU_PROFILES:
        print(f"⚠️  No GPU profile found (gpu_name={gpu_name}). Using defaults.")
        return None

    profile = GPU_PROFILES[gpu_name]
    seq_per_micro_batch = profile["seq_per_micro_batch"]
    dtype = profile["dtype"]
    use_gradient_checkpointing = profile["use_gradient_checkpointing"]
    gradient_accumulation_steps = global_batch_size // (seq_per_micro_batch * max_seq_length)

    print(f"\n🎮 GPU Configured: {gpu_name}")
    print(f"   VRAM              : {profile['vram_gb']} GB")
    print(f"   Micro-batch       : {seq_per_micro_batch} seqs × {max_seq_length} = {seq_per_micro_batch * max_seq_length:,} tokens")
    print(f"   Grad accumulation : {gradient_accumulation_steps} steps")
    print(f"   Dtype             : {dtype}")
    print(f"   Free hours        : {profile['free_hours']}")
    print(f"   Note              : {profile['notes']}")

    return gpu_name
