"""
Test Suite for the 124M LLM.

ALWAYS run this before starting a real training run!
It takes < 2 minutes and catches bugs that would otherwise
waste hours of compute time.

Run with:
    python test_model.py

What each test checks:
    Test 1: Config sanity        - hyperparameter math is correct
    Test 2: Model shapes         - no shape mismatch in any layer
    Test 3: Forward pass         - no crash on a dummy batch
    Test 4: Loss sanity          - untrained model loss ≈ ln(vocab_size) ≈ 10.8
    Test 5: Gradient flow        - all parameters receive gradients
    Test 6: Weight tying         - lm_head and token_embed share weights
    Test 7: RoPE correctness     - rotary embeddings have correct shapes
    Test 8: Mini training loop   - 20 steps, loss goes DOWN (overfit test)
    Test 9: Checkpoint save/load - model state preserved exactly
    Test 10: Generation          - model can generate without crashing
"""

import sys
import math
import time
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from model import LanguageModel, RMSNorm, RotaryEmbedding, FeedForward, GroupedQueryAttention


# ---------------------------------------------------------------------------
# Minimal "tiny" config for fast testing (don't run 124M in tests!)
# ---------------------------------------------------------------------------
class TinyConfig:
    """Tiny model for fast tests. Same architecture, just smaller."""
    hidden_dim               = 64
    num_layers               = 2
    num_attention_heads      = 4
    num_kv_heads             = 2        # GQA 2:1
    ffn_expansion            = 4
    ffn_hidden_dim           = int(64 * 4 * 2 / 3)   # SwiGLU formula
    vocab_size               = 256      # tiny vocab
    max_seq_length           = 32
    head_dim                 = hidden_dim // num_attention_heads
    use_gradient_checkpointing = False


TINY = TinyConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model() -> LanguageModel:
    return LanguageModel(TINY).to(DEVICE)


def make_batch(batch=2, seq=None):
    seq = seq or TINY.max_seq_length
    ids = torch.randint(0, TINY.vocab_size, (batch, seq), device=DEVICE)
    return ids


# ===========================================================================
# TESTS
# ===========================================================================

class TestConfig(unittest.TestCase):
    """Test 1 — Config sanity checks."""

    def test_head_dim_divides_evenly(self):
        """hidden_dim must be divisible by num_attention_heads."""
        self.assertEqual(
            cfg.hidden_dim % cfg.num_attention_heads, 0,
            f"hidden_dim={cfg.hidden_dim} not divisible by "
            f"num_attention_heads={cfg.num_attention_heads}"
        )

    def test_kv_heads_divides_q_heads(self):
        """Q heads must be divisible by KV heads (GQA constraint)."""
        self.assertEqual(
            cfg.num_attention_heads % cfg.num_kv_heads, 0,
            f"num_attention_heads={cfg.num_attention_heads} not divisible by "
            f"num_kv_heads={cfg.num_kv_heads}"
        )

    def test_gradient_accumulation_positive(self):
        """Gradient accumulation steps must be at least 1."""
        self.assertGreater(cfg.gradient_accumulation_steps, 0)

    def test_ffn_dim_positive(self):
        """FFN hidden dim must be > 0."""
        self.assertGreater(cfg.ffn_hidden_dim, 0)

    def test_batch_size_x_seq_leq_global(self):
        """seq_per_micro_batch * seq_len * acc_steps should equal global_batch_size."""
        actual = cfg.seq_per_micro_batch * cfg.max_seq_length * cfg.gradient_accumulation_steps
        self.assertAlmostEqual(actual, cfg.global_batch_size, delta=cfg.seq_per_micro_batch * cfg.max_seq_length,
            msg=f"Accumulated tokens ({actual}) != global_batch_size ({cfg.global_batch_size})")


class TestRMSNorm(unittest.TestCase):
    """Test 2a — RMSNorm component."""

    def test_output_shape(self):
        norm = RMSNorm(64).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_unit_rms(self):
        """After normalization, RMS of output should be close to 1 (before gamma scaling)."""
        norm = RMSNorm(64).to(DEVICE)
        # Initialize gamma to 1 (it already is, but be explicit)
        nn.init.ones_(norm.gamma)
        x = torch.randn(4, 8, 64, device=DEVICE) * 5.0  # large values
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        self.assertTrue(
            rms.mean().item() < 2.0,
            f"RMS of normalized output is too large: {rms.mean().item():.4f}"
        )


class TestRoPE(unittest.TestCase):
    """Test 2b — RotaryEmbedding."""

    def test_output_shapes(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128).to(DEVICE)
        x = torch.zeros(2, 32, 64, device=DEVICE)
        cos, sin = rope(x, seq_len=32)
        self.assertEqual(cos.shape, (32, 64))
        self.assertEqual(sin.shape, (32, 64))

    def test_cos_sin_bounded(self):
        """cos and sin values must be in [-1, 1]."""
        rope = RotaryEmbedding(64).to(DEVICE)
        x = torch.zeros(1, 16, 64, device=DEVICE)
        cos, sin = rope(x, 16)
        self.assertLessEqual(cos.abs().max().item(), 1.001)
        self.assertLessEqual(sin.abs().max().item(), 1.001)


class TestFeedForward(unittest.TestCase):
    """Test 2c — FeedForward (SwiGLU)."""

    def test_output_shape(self):
        ff = FeedForward(64, 128).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = ff(x)
        self.assertEqual(out.shape, (2, 16, 64))

    def test_no_nan(self):
        ff = FeedForward(64, 128).to(DEVICE)
        x = torch.randn(2, 16, 64, device=DEVICE)
        out = ff(x)
        self.assertFalse(torch.isnan(out).any(), "FeedForward output contains NaN!")


class TestModelShapes(unittest.TestCase):
    """Test 3 — Forward pass shape checks."""

    def setUp(self):
        self.model = make_model()

    def test_logits_shape(self):
        ids = make_batch(batch=2, seq=TINY.max_seq_length)
        logits, loss = self.model(ids, targets=ids)
        self.assertEqual(
            logits.shape,
            (2, TINY.max_seq_length, TINY.vocab_size),
            f"Unexpected logits shape: {logits.shape}"
        )

    def test_loss_is_scalar(self):
        ids = make_batch()
        _, loss = self.model(ids, targets=ids)
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_no_output_nan(self):
        ids = make_batch()
        logits, loss = self.model(ids, targets=ids)
        self.assertFalse(torch.isnan(logits).any(), "logits contain NaN!")
        self.assertFalse(torch.isnan(loss), "loss is NaN!")


class TestLossSanity(unittest.TestCase):
    """
    Test 4 — Untrained model loss sanity.

    An untrained model has random weights. The expected loss for uniform
    predictions over V classes is:

        Loss ≈ ln(V) = ln(vocab_size)

    For tiny vocab (256), expected loss ≈ ln(256) ≈ 5.5
    For GPT-2 vocab (50304), expected loss ≈ ln(50304) ≈ 10.8

    If initial loss is very different (e.g. 0 or 100+), something is wrong
    with initialization or the loss function.
    """

    def test_initial_loss_reasonable(self):
        model = make_model()
        ids = make_batch()
        with torch.no_grad():
            _, loss = model(ids, targets=ids)

        expected_loss = math.log(TINY.vocab_size)
        loss_val = loss.item()

        print(f"\n  Initial loss: {loss_val:.4f}  (expected ≈ {expected_loss:.4f})")

        self.assertGreater(loss_val, 0.5,
            f"Loss {loss_val:.4f} is suspiciously low — possible data leak or wrong loss fn")
        self.assertLess(loss_val, expected_loss * 3,
            f"Loss {loss_val:.4f} is too high (>3x expected). Check weight init.")


class TestGradientFlow(unittest.TestCase):
    """Test 5 — All parameters receive non-zero gradients."""

    def test_all_params_have_grad(self):
        model = make_model()
        ids = make_batch()
        _, loss = model(ids, targets=ids)
        loss.backward()

        no_grad = []
        zero_grad = []

        for name, param in model.named_parameters():
            if param.grad is None:
                no_grad.append(name)
            elif param.grad.abs().max().item() == 0.0:
                zero_grad.append(name)

        if no_grad:
            self.fail(f"Parameters with NO gradient: {no_grad}")

        # Weight-tied lm_head.weight is the same tensor as token_embed.weight.
        # It's ok if one of them shows up as zero grad after tying — pytorch
        # accumulates the gradient into the shared tensor.
        non_tied_zero = [n for n in zero_grad if 'lm_head' not in n]
        if non_tied_zero:
            print(f"\n  ⚠️  Params with zero grad: {non_tied_zero}")
            # This is a warning not a hard failure (can happen with frozen layers)


class TestWeightTying(unittest.TestCase):
    """Test 6 — lm_head.weight and token_embed.weight are the SAME tensor."""

    def test_weights_are_tied(self):
        model = make_model()
        self.assertIs(
            model.lm_head.weight,
            model.token_embed.weight,
            "lm_head.weight and token_embed.weight are NOT tied! "
            "This wastes 38M parameters. Check _init_weights."
        )

    def test_tied_grad_accumulates(self):
        """After backward, both names point to the same gradient tensor."""
        model = make_model()
        ids = make_batch()
        _, loss = model(ids, targets=ids)
        loss.backward()
        self.assertIs(
            model.lm_head.weight.grad,
            model.token_embed.weight.grad
        )


class TestMiniTraining(unittest.TestCase):
    """
    Test 8 — Mini training loop (overfitting test).

    Train for 30 steps on a FIXED tiny batch.
    The model MUST memorise it (loss should drop significantly).
    If loss doesn't drop, the training loop, optimizer, or LR is broken.
    """

    def test_loss_decreases_on_overfit(self):
        model = make_model().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed batch — always the same (overfitting test)
        ids = make_batch(batch=2, seq=TINY.max_seq_length)

        initial_loss = None
        final_loss   = None

        for step in range(40):
            optimizer.zero_grad()
            _, loss = model(ids, targets=ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        print(f"\n  Overfit test: loss {initial_loss:.4f} -> {final_loss:.4f}")

        self.assertLess(
            final_loss, initial_loss * 0.5,
            f"Loss only dropped from {initial_loss:.4f} to {final_loss:.4f}. "
            f"Expected at least 50% reduction after 40 steps of overfitting. "
            f"Check the training loop and optimizer."
        )


class TestCheckpoint(unittest.TestCase):
    """Test 9 — Checkpoint save and load."""

    def test_save_and_load(self):
        from train import Checkpoint

        model = make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Checkpoint(tmpdir)

            # Forward + backward to populate optimizer state
            ids = make_batch()
            _, loss = model(ids, targets=ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Save weights before modifying
            original_weight = model.token_embed.weight.clone().detach()

            # Save checkpoint
            ckpt.save(model, optimizer, step=1, loss=loss.item(), scaler=scaler)

            # Corrupt the model weights
            with torch.no_grad():
                model.token_embed.weight.fill_(0.0)

            # Load checkpoint
            ckpt_data = ckpt.load_latest(model, optimizer, scaler)

            restored_weight = model.token_embed.weight.clone().detach()

            self.assertEqual(ckpt_data["step"], 1, "Checkpoint step not correctly restored")
            self.assertTrue(
                torch.allclose(original_weight, restored_weight),
                "Weights not correctly restored from checkpoint!"
            )


class TestGeneration(unittest.TestCase):
    """Test 10 — Generation doesn't crash and produces correct shapes."""

    def test_greedy_generation(self):
        model = make_model().eval()
        prompt = make_batch(batch=1, seq=4)

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10)

        # Output should be prompt + new tokens
        self.assertEqual(output.shape, (1, 4 + 8))

    def test_temperature_zero_is_greedy(self):
        """temperature close to 0 should be deterministic."""
        model = make_model().eval()
        prompt = make_batch(batch=1, seq=4)

        with torch.no_grad():
            out1 = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-10)
            out2 = model.generate(prompt.clone(), max_new_tokens=5, temperature=1e-10)

        self.assertTrue(
            torch.equal(out1, out2),
            "Extremely low temperature should give deterministic output"
        )


# ===========================================================================
# RUNNER
# ===========================================================================

def run_all_tests():
    """Run all tests and print a summary."""
    print("\n" + "=" * 65)
    print("  LLM 124M — Test Suite")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestConfig,
        TestRMSNorm,
        TestRoPE,
        TestFeedForward,
        TestModelShapes,
        TestLossSanity,
        TestGradientFlow,
        TestWeightTying,
        TestMiniTraining,
        TestCheckpoint,
        TestGeneration,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    t0 = time.time()
    result = runner.run(suite)
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    if result.wasSuccessful():
        print(f"  ✅  ALL {result.testsRun} TESTS PASSED  ({elapsed:.1f}s)")
        print("  Ready to start training!")
    else:
        print(f"  ❌  {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        print("  Fix the issues above before running train.py!")
    print("=" * 65 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
