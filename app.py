#!/usr/bin/env python3
"""
Inference Server — 124M Instruction-Tuned LLM
Launches a Gradio web UI for interactive text generation with the fine-tuned model.

Usage:
    python server.py

Then open http://localhost:7860 in your browser.
"""

import sys
import torch
import gradio as gr
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: this script lives in the project root alongside the llm_124m/
# sub-package and the POSTTRAINING_RESULT/ checkpoint directory.
# Allow it to be run from any working directory by setting sys.path early.
# ---------------------------------------------------------------------------
_TRAINING_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_TRAINING_ROOT / "llm_124m"))

import config          # llm_124m/config.py
from model import LanguageModel
from dataloader import get_tokenizer

# ============================================================================
# MODEL LOADING
# ============================================================================

FINE_TUNED_CHECKPOINT = _TRAINING_ROOT / "POSTTRAINING_RESULT" / "instruction_tuned_curated_epoch_3.pt"
BACKUP_CHECKPOINT     = _TRAINING_ROOT / "checkpoints"         / "instruction_tuned_epoch_3.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 60)
print("  Loading 124M Instruction-Tuned Model")
print("=" * 60)
print(f"  Device  : {device}")
print(f"  Dtype   : {config.dtype}")
print(f"  Context : {config.max_seq_length} tokens")

_ckpt_candidates = [FINE_TUNED_CHECKPOINT, BACKUP_CHECKPOINT]


def _resolve_checkpoint(candidates):
    """Return the first existing checkpoint path from the given candidates."""
    for path in candidates:
        if path.is_file():
            return path
    existence = {str(p): p.exists() for p in candidates}
    raise FileNotFoundError(
        "No checkpoint found. Checked:\n  "
        + "\n  ".join(
            f"{p}  {'(exists)' if ok else '(missing)'}"
            for p, ok in existence.items()
        )
    )


ckpt_path = _resolve_checkpoint(_ckpt_candidates)
print(f"  Checkpoint: {ckpt_path}")

model = LanguageModel(config)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

state_dict = checkpoint["model_state_dict"]
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("  Model loaded OK\n")

tokenizer = get_tokenizer()


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    """
    Autoregressive generation with:
      - Sliding context window (last max_seq_length tokens)
      - Temperature scaling
      - Top-K filtering
      - Nucleus (top-P) sampling
    """
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Keep at most config.max_seq_length tokens in context
        ctx = (
            input_tensor
            if input_tensor.shape[1] <= config.max_seq_length
            else input_tensor[:, -config.max_seq_length:]
        )

        logits, _ = model(ctx)
        next_logits = logits[:, -1, :] / temperature

        # Top-K filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, top_k)
            threshold = top_k_vals[:, -1:]
            next_logits[next_logits < threshold] = float("-inf")

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            nucleus_mask = cumulative > top_p
            nucleus_mask[:, 0] = False  # always keep the most likely token
            next_logits[sorted_idx[nucleus_mask]] = float("-inf")

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_tensor = torch.cat([input_tensor, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_tensor[0].tolist(), skip_special_tokens=True)


def _format_prompt(user_input: str) -> str:
    """Wrap raw user text in [INST] ... [/INST] tags if not already wrapped."""
    stripped = user_input.strip()
    if stripped.startswith("[INST]") and "[/INST]" in stripped:
        return stripped
    return f"[INST] {stripped} [/INST]"


def _extract_response(full_text: str) -> str:
    """Strip the prompt prefix; return only what the model replied."""
    marker = "[/INST]"
    if marker in full_text:
        return full_text.split(marker, 1)[1].strip()
    return full_text.strip()


# ============================================================================
# GRADIO WEB UI
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft(), title="124M Instruction-Tuned LLM") as demo:

    gr.Markdown(
        """
        # 🚀 124M Instruction-Tuned LLM

        Interactive text generation powered by a 124M-parameter instruction-tuned language model.
        Enter any question or instruction below to get a response from the model.
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="Instruction / Prompt",
                placeholder="Type your question or instruction here…",
                lines=4,
                autofocus=True,
            )
        with gr.Column(scale=1):
            response_output = gr.Textbox(
                label="Model Response",
                lines=6,
                interactive=False,
                show_copy_button=True,
            )

    with gr.Row():
        submit_btn = gr.Button("Generate", variant="primary", size="lg")
        clear_btn  = gr.Button("Clear", size="lg")

    with gr.Accordion("⚙️ Generation Settings", open=False):
        with gr.Row():
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, step=0.05, value=0.8,
                label="Temperature",
                info="Higher → more creative  ·  Lower → more deterministic",
            )
            max_tokens = gr.Slider(
                minimum=32, maximum=2048, step=8, value=300,
                label="Max new tokens",
                info="Maximum tokens the model will generate",
            )
        with gr.Row():
            top_k = gr.Slider(
                minimum=1, maximum=100, step=1, value=50,
                label="Top-K",
                info="Sample from only the K most likely next tokens",
            )
            top_p = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, value=0.95,
                label="Top-P (Nucleus)",
                info="Sample from smallest token set whose cumulative prob ≥ P",
            )

    gr.Examples(
        examples=[
            "What is the capital of France, and why is it historically significant?",
            "Write a Python function that checks whether a number is prime.",
            "Summarize the theory of general relativity in two paragraphs.",
            "Write a short haiku about the ocean.",
            "What are the main differences between C++ and Rust?",
            "Explain how a neural network learns backpropagation.",
        ],
        inputs=user_input,
        label="Example prompts",
    )

    # ------------------------------------------------------------------
    # Inference callback
    # ------------------------------------------------------------------

    def inference_fn(user_text, temp, tokens, k, p):
        try:
            if not user_text.strip():
                return ""
            prompt = _format_prompt(user_text)
            full   = _generate(
                model, tokenizer,
                prompt,
                max_new_tokens=tokens,
                temperature=temp,
                top_p=p,
                top_k=k,
            )
            return _extract_response(full)
        except Exception as exc:
            return f"[Error] {exc}"

    # Button / Enter triggers
    submit_btn.click(
        fn=inference_fn,
        inputs=[user_input, temperature, max_tokens, top_k, top_p],
        outputs=response_output,
    )

    user_input.submit(
        fn=inference_fn,
        inputs=[user_input, temperature, max_tokens, top_k, top_p],
        outputs=response_output,
    )

    # Clear button
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[user_input, response_output],
    )


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
