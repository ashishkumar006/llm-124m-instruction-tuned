"""
Model Architecture for 124M Parameter Transformer.
Implements modern 2026-style Transformer with:
- RMSNorm (Pre-LN): More stable than LayerNorm
- RoPE (Rotary Positional Embeddings): Allows context extrapolation
- Grouped Query Attention (GQA): Reduces KV cache memory
- SwiGLU Activation: More expressive than GeLU
- Flash Attention 3: Optimized attention kernel
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from typing import Optional, Tuple

# Try to import flash-attn for fast attention (optional but recommended)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️  Flash Attention not installed. Using standard attention (slower).")
    print("   Install with: pip install flash-attn")

# ============================================================================
# 1. RMSNORM - Stable Layer Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More stable and faster than LayerNorm. Used in Llama, Qwen, etc.
    
    Formula: y = (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # Learnable scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
        
        Returns:
            Normalized tensor with same shape
        """
        # Calculate RMS along the hidden dimension
        rms = torch.sqrt((x.pow(2).mean(-1, keepdim=True)) + self.eps)
        
        # Normalize and scale
        normalized = x / rms
        return normalized * self.gamma


# ============================================================================
# 2. ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Encodes position information by rotating the query and key vectors.
    
    Advantages over absolute position embeddings:
    - Enables context extrapolation (train on 1024 tokens, use 2048 at inference)
    - More stable gradients
    - Standard in 2026 models (Llama 3, Qwen, etc.)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Pre-compute rotation matrices
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            x: Input tensor (used only for device/dtype information)
            seq_len: Sequence length
        
        Returns:
            cos and sin matrices for rotation
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Compute outer product: (seq_len,) @ (dim/2,) -> (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Concatenate: [freqs, freqs] to get (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Return cos and sin for rotation
        return emb.cos(), emb.sin()


def apply_rotary_embeddings(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, 
                            sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to queries and keys.
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        cos: Cosine values (seq_len, head_dim)
        sin: Sine values (seq_len, head_dim)
    
    Returns:
        Rotated query and key tensors
    """
    # Add batch and head dimensions for broadcasting.
    # Shape: (1, 1, seq_len, head_dim) — broadcasts against any num_heads.
    # BUG FIX: old code did .expand(q.shape[0], q.shape[1], ...) which locked
    # the head count to Q's heads. K has fewer heads (GQA), causing a mismatch.
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim)
    
    # Split into two halves for rotation formula: [x, y] -> [-y, x]
    q_rot = torch.cat([-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]], dim=-1)
    k_rot = torch.cat([-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]], dim=-1)
    
    # Apply rotation: q_new = q*cos + q_rot*sin
    q_out = (q * cos) + (q_rot * sin)
    k_out = (k * cos) + (k_rot * sin)
    
    return q_out, k_out


# ============================================================================
# 3. GROUPED QUERY ATTENTION (GQA)
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Reduces KV cache memory by using fewer K,V heads than Q heads.
    
    Standard Attention: num_kv_heads = num_q_heads (wastes memory)
    GQA: num_kv_heads < num_q_heads (e.g., 4 KV heads, 12 Q heads = 3:1 ratio)
    
    Formula:
        Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Benefit: 30-50% reduction in KV cache with minimal performance loss.
    """
    
    def __init__(self, hidden_dim: int, num_q_heads: int, num_kv_heads: int, 
                 head_dim: int, rope: RotaryEmbedding):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope = rope
        
        # Query projection (full heads)
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
        
        # Key-Value projections (reduced heads)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(num_q_heads * head_dim, hidden_dim, bias=False)
        
        # Scaling factor
        self.scale = head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            causal_mask: Causal attention mask (optional)
        
        Returns:
            Attention output (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_embeddings(q, k, cos, sin)
        
        # Use Flash Attention if available, else PyTorch's built-in SDPA
        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects (batch, seq, num_heads, head_dim)
            # flash_attn_func handles GQA natively when kv heads < q heads
            out = flash_attn_func(
                q.transpose(1, 2),   # (batch, seq, num_q_heads, head_dim)
                k.transpose(1, 2),   # (batch, seq, num_kv_heads, head_dim)
                v.transpose(1, 2),   # (batch, seq, num_kv_heads, head_dim)
                causal=True
            )
            # out: (batch, seq, num_q_heads, head_dim) -> (batch, num_q_heads, seq, head_dim)
            out = out.transpose(1, 2)
        else:
            # PyTorch 2.0+ scaled_dot_product_attention (SDPA)
            # This automatically selects the best kernel:
            #   1. FlashAttention (if GPU supports it)
            #   2. Memory-efficient attention (xFormers-like)
            #   3. Math fallback (manual, O(N^2) — last resort)
            #
            # KEY: This does NOT build the full (seq, seq) attention matrix
            # in memory like our old manual code did. That's why it fits in
            # 16GB VRAM while the old code OOM'd.
            #
            # For GQA: expand KV heads to match Q heads before calling SDPA
            repeat_factor = self.num_q_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(repeat_factor, dim=1)  # (batch, num_q_heads, seq, head_dim)
            v_exp = v.repeat_interleave(repeat_factor, dim=1)  # (batch, num_q_heads, seq, head_dim)

            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=None,
                is_causal=True,       # applies causal mask internally
                scale=self.scale,
            )
        
        # Merge heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


# ============================================================================
# 4. FEEDFORWARD WITH SWIGLU - Gated Feed-Forward Network
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-Forward Network using SwiGLU activation.
    This is the canonical implementation used in Llama 3, Qwen, Mistral, etc.

    SwiGLU formula:
        FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    Three linear layers:
      - gate_proj : hidden_dim  -> ffn_hidden_dim   (gating branch)
      - up_proj   : hidden_dim  -> ffn_hidden_dim   (value branch)
      - down_proj : ffn_hidden_dim -> hidden_dim    (projection back)

    WHY 3 LAYERS NOT 2:
      Standard FFN: x -> Linear -> GELU -> Linear
      SwiGLU FFN:   x -> gate_proj  -> SiLU  ─┐
                    x -> up_proj             ─ * -> down_proj
      The gate decides WHICH features matter. More expressive!

    KEY BUG FIX: The previous SwiGLU + nn.Sequential was double-projecting
    and had mismatched dimensions. This canonical 3-layer design is correct.
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        # SiLU(gate) * up  — this is the SwiGLU gate operation
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 5. TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block (one layer).
    
    Architecture:
    1. Pre-LN: RMSNorm(x)
    2. Multi-head attention
    3. Residual connection: x + Attention(...)
    4. Pre-LN: RMSNorm(x)
    5. Feed-forward (SwiGLU)
    6. Residual connection: x + FFN(...)
    """
    
    def __init__(self, hidden_dim: int, num_q_heads: int, num_kv_heads: int, 
                 head_dim: int, ffn_hidden_dim: int, rope: RotaryEmbedding):
        super().__init__()
        
        # Pre-normalization (applied BEFORE attention and FFN, not after)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

        # Attention
        self.attention = GroupedQueryAttention(hidden_dim, num_q_heads, num_kv_heads,
                                              head_dim, rope)

        # BUG FIX: Replace broken nn.Sequential(Linear, SwiGLU, Linear) with
        # the canonical 3-projection FeedForward. The old design double-projected
        # and had mismatched dimensions that would crash on first forward pass.
        self.ffn = FeedForward(hidden_dim, ffn_hidden_dim)
    
    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            causal_mask: Causal attention mask
        
        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        # Attention with pre-norm and residual
        x = x + self.attention(self.norm1(x), causal_mask)
        
        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))
        
        return x


# ============================================================================
# 6. FULL LANGUAGE MODEL
# ============================================================================

class LanguageModel(nn.Module):
    """
    Complete 124M Parameter Transformer Language Model.
    
    Architecture:
    1. Token Embeddings: vocab_size -> hidden_dim
    2. 12 Transformer Blocks
    3. Output Layer Norm
    4. LM Head: project back to vocab_size for next-token prediction
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Rotary embeddings (shared across all layers — no extra params)
        self.rope = RotaryEmbedding(config.head_dim, max_seq_len=config.max_seq_length)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_q_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                rope=self.rope
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = RMSNorm(config.hidden_dim)

        # LM head (weights will be TIED to token_embed in _init_weights)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Whether to use gradient checkpointing (saves memory, slower)
        self.use_grad_checkpoint = getattr(config, 'use_gradient_checkpointing', False)

        # Initialize weights (includes weight tying)
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using GPT-NeoX / Llama-style scaling.

        Key insight: residual projections (out_proj, down_proj) need to be
        scaled DOWN by 1/sqrt(2 * num_layers) to prevent the residual stream
        from growing too large as you go deeper. Without this, deep models
        can have exploding activations at initialization.

        Formula: std = 0.02 / sqrt(2 * num_layers)
        """
        std = 0.02
        residual_std = 0.02 / math.sqrt(2 * self.config.num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Residual projection layers need smaller init
                if 'out_proj' in name or 'down_proj' in name:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

        # WEIGHT TYING: Share weights between token embedding and LM head.
        # This is a key trick:
        # - Saves ~38M parameters (vocab_size × hidden_dim)
        # - Improves quality because embedding and output use same space
        # - Used in GPT-2, Llama, almost all modern LLMs
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)
            **kwargs: Additional arguments (e.g., attention_mask) - ignored for compatibility
        
        Returns:
            logits: Predictions (batch, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided, else None)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embed(input_ids)  # (batch, seq_len, hidden_dim)

        # Forward through transformer blocks
        # Gradient checkpointing: recompute activations during backward instead of
        # storing them. Saves up to 60% of activation memory at cost of ~33% slower.
        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (batch * seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss
    
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        """
        Prepare inputs for generation (required by PEFT for LoRA compatibility).
        For our model, we just return the input_ids as-is.
        """
        return {"input_ids": input_ids}
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k most likely tokens (if None, no filtering)
        
        Returns:
            output_ids: Generated token IDs (batch, seq_len + max_new_tokens)
        """
        max_ctx = self.config.max_seq_length

        for _ in range(max_new_tokens):
            # Truncate to max context length to prevent O(n^2) memory blowup
            # RoPE allows this — we just lose very old context
            ctx = input_ids if input_ids.shape[1] <= max_ctx else input_ids[:, -max_ctx:]

            # Forward pass
            logits, _ = self(ctx)
            
            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature
            
            # Top-K filtering
            if top_k is not None:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, -1:]] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
        
        return input_ids
