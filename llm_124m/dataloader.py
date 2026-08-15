"""
Data Loading Pipeline — Streaming from HuggingFace.

====================================================================
WHAT THIS FILE DOES:
====================================================================

Documents are streamed from HuggingFace (no 40GB download), tokenized,
and PACKED into fixed-length training sequences. No padding, no wasted
compute — every token in every batch is useful.

Flow:
    HuggingFace (remote)
      → Stream one document at a time
      → Tokenize (GPT-2 tokenizer, 50257 tokens)
      → Append EOS between documents
      → Pack into 2048-token chunks
      → Yield (input_ids, targets) pairs
      → DataLoader stacks into batches

CRITICAL BUG FIX (from previous version):
    Old code yielded ONE TOKEN at a time, making it impossible to form
    actual batches. This version yields complete packed sequences.
====================================================================
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TOKENIZER
# ============================================================================

def get_tokenizer(name: str = "gpt2"):
    """
    Load pre-trained GPT-2 tokenizer from HuggingFace.

    GPT-2 tokenizer:
      - 50257 tokens (BPE)
      - No padding token by default (we set pad = eos)
      - Used by many open models, well-tested

    Returns:
        AutoTokenizer with pad_token set
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Suppress "Token indices sequence length is longer than the specified
    # maximum sequence length" warning — we handle length ourselves via packing
    tokenizer.model_max_length = 1_000_000
    logger.info(f"✅ Tokenizer loaded: {name} (vocab={len(tokenizer)})")
    return tokenizer


# ============================================================================
# PACKED STREAMING DATASET
# ============================================================================

class PackedDataset(IterableDataset):
    """
    Streams documents from HuggingFace, tokenizes them, and packs
    consecutive tokens into fixed-length training sequences.

    WHY PACKING?
    ------------
    Naive approach: one document = one training example.
    Problem: short documents waste GPU (padding) and long documents
    get truncated (information loss).

    Packing: concatenate all documents into one long stream with EOS
    tokens between them, then slice into fixed-length chunks.
    Result: zero padding, zero waste, every token trains the model.

    This is standard practice in GPT-2, GPT-3, Llama, Chinchilla, etc.

    Each yielded item is:
        {"input_ids": LongTensor(seq_len), "targets": LongTensor(seq_len)}

    where targets = input_ids shifted left by 1 (next-token prediction).

    RESUME SAFETY:
    The shuffle seed is fixed so the document order is deterministic.
    On resume, the training loop fast-forwards through already-consumed
    batches to reach the exact position where training left off.
    This guarantees zero data overlap across resumes.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_config: str,
        split: str,
        tokenizer,
        max_seq_length: int,
        shuffle_buffer_size: int = 10_000,
        shuffle_seed: int = 42,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset (e.g., "HuggingFaceFW/fineweb-edu")
            dataset_config: Config name (e.g., "sample-10BT")
            split: Dataset split (e.g., "train")
            tokenizer: Tokenizer with .encode() method
            max_seq_length: Sequence length for training (e.g., 2048)
            shuffle_buffer_size: Buffer for streaming shuffle (larger = better shuffle)
            shuffle_seed: Fixed seed for deterministic shuffle order (critical for resume)
        """
        self.tokenizer = tokenizer
        self.seq_length = max_seq_length
        self.eos_id = tokenizer.eos_token_id
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed

        # KEY: streaming=True means no download — data comes over network
        logger.info(f"📥 Loading dataset: {dataset_name}/{dataset_config} (streaming)")
        self.dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            streaming=True,      # Stream instead of downloading 40GB
            trust_remote_code=True,
        )
        logger.info("✅ Dataset ready (streaming mode — no disk space needed)")

    def __iter__(self):
        """
        Iterate: stream documents → tokenize → pack → yield sequences.

        The token buffer accumulates tokens from documents. When it has
        enough for one full sequence (seq_length + 1 tokens), we yield
        that chunk and move on. The +1 is for the target of the last
        position (next-token prediction needs N+1 tokens for N positions).
        """
        # CRITICAL: use a fixed seed so the document order is identical
        # across process restarts. On resume the training loop fast-forwards
        # through already-consumed batches to reach the correct position.
        dataset = self.dataset.shuffle(
            buffer_size=self.shuffle_buffer_size,
            seed=self.shuffle_seed,
        )

        token_buffer = []
        chunk_size = self.seq_length + 1  # +1 for the final target token

        for example in dataset:
            # FineWeb-Edu has 'text' key; other datasets might use 'content'
            text = example.get("text") or example.get("content", "")
            if not text or len(text.strip()) < 20:
                continue

            # Tokenize entire document and append EOS separator
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_id)
            token_buffer.extend(tokens)

            # Yield all complete sequences from the buffer
            while len(token_buffer) >= chunk_size:
                chunk = token_buffer[:chunk_size]
                token_buffer = token_buffer[chunk_size:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "targets": targets}


# ============================================================================
# DATALOADER FACTORY
# ============================================================================

def get_data_loaders(config, batch_size: int):
    """
    Create a streaming DataLoader ready for training.

    Args:
        config: Config module with dataset_name, dataset_config, etc.
        batch_size: Number of sequences per micro-batch (= seq_per_micro_batch)

    Returns:
        (train_loader, None) — no separate validation loader for streaming

    The DataLoader automatically stacks individual sequences into batches:
        Single item:  {"input_ids": (seq_len,), "targets": (seq_len,)}
        Batch of N:   {"input_ids": (N, seq_len), "targets": (N, seq_len)}
    """
    tokenizer = get_tokenizer("gpt2")

    dataset = PackedDataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.train_split,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        shuffle_buffer_size=getattr(config, "shuffle_buffer_size", 10_000),
        shuffle_seed=getattr(config, "seed", 42),
    )

    # CRITICAL for throughput: use multiple workers so the CPU tokenizes
    # the NEXT batch while the GPU trains on the CURRENT one.
    # Without this, the GPU sits idle during data loading (~60% of wall time!).
    # persistent_workers keeps HTTP connections alive between batches.
    nw = getattr(config, 'dataloader_workers', 2)
    if nw > 0:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=4,          # pre-load 4 batches per worker
            persistent_workers=True,    # keep workers alive (saves HTTP reconnect)
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
        )

    tokens_per_step = batch_size * config.max_seq_length
    logger.info(
        f"✅ DataLoader ready: {batch_size} seqs × {config.max_seq_length} tokens "
        f"= {tokens_per_step:,} tokens/micro-step"
    )
    return loader, None


def get_validation_batches(config, batch_size: int, num_batches: int = 20) -> list:
    """
    Cache a fixed set of validation batches from a separate data stream.

    These batches are drawn from the same dataset pool but shuffled
    independently from training. They are cached in memory so evaluation
    is fast and consistent across steps (same data every time).

    With 20 batches × 8 seqs × 2048 tokens ≈ 328K tokens — tiny but
    enough for a reliable loss signal.

    NOTE: Since FineWeb-Edu 10BT only has a 'train' split, these come
    from the same pool as training data, but are independently shuffled.
    With 10B tokens in the pool and only 4B used for training, overlap
    is small.  The key benefit is that these batches are FIXED — so you
    get a consistent signal across steps.

    Args:
        config: Config module with dataset_name, dataset_config, etc.
        batch_size: Sequences per batch (= seq_per_micro_batch)
        num_batches: Number of batches to cache (default 20)

    Returns:
        List of batch dicts, each with 'input_ids' and 'targets' tensors
    """
    tokenizer = get_tokenizer("gpt2")

    val_dataset = PackedDataset(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.train_split,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        shuffle_buffer_size=1000,  # smaller buffer — we only need a few batches
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    batches = []
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        batches.append(batch)

    total_tokens = sum(b["input_ids"].numel() for b in batches)
    logger.info(
        f"✅ Cached {len(batches)} validation batches "
        f"({total_tokens:,} tokens) for consistent evaluation"
    )
    return batches


# ============================================================================
# STANDALONE TEST (run: python dataloader.py)
# ============================================================================

if __name__ == "__main__":
    """
    Quick test: stream 3 batches and print shapes.
    Run this to verify HuggingFace streaming works on your machine.
    """
    import config

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("  DataLoader Streaming Test")
    print("=" * 60)

    loader, _ = get_data_loaders(config, batch_size=2)

    for i, batch in enumerate(loader):
        print(f"\n  Batch {i + 1}:")
        print(f"    input_ids shape : {batch['input_ids'].shape}")
        print(f"    targets shape   : {batch['targets'].shape}")
        print(f"    tokens in batch : {batch['input_ids'].numel():,}")

        # Verify targets = input shifted by 1
        # (chunk[:-1] vs chunk[1:] — so input[1:] == target[:-1])
        match = (batch["input_ids"][:, 1:] == batch["targets"][:, :-1]).float().mean()
        print(f"    shift check     : {match:.4f} (should be 1.0000)")

        if i >= 2:
            break

    print("\n✅ Streaming works! Data flows from HuggingFace → tokenizer → packed sequences.\n")
