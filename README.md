# 124M Instruction-Tuned Language Model

A 124M-parameter transformer language model, trained from scratch and
instruction-tuned, served through an interactive web demo.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-blue)](https://huggingface.co/spaces/aishwarya/llm-124m-instruction-tuned)

## Overview

This project delivers a small language model built entirely from the ground up:

1. **Pretraining** — a transformer is trained from random initialization on ~4B
   tokens of text, acquiring language structure and broad knowledge through
   next-token prediction.
2. **Instruction tuning** — the base model is fine-tuned on a curated set of
   instruction/response pairs, enabling it to follow prompts such as "explain",
   "summarize", or "write code" rather than merely completing text.
3. **Deployment** — the model is served through a Gradio application on Hugging
   Face Spaces for interactive use.

The model architecture, training loop, dataset curation, and demo are implemented
from scratch, offering full visibility into every stage of the pipeline.

## Architecture

| Property | Value |
|----------|-------|
| Parameters | ~114M (weight-tied embeddings) |
| Layers | 12 transformer blocks |
| Hidden dim | 768 |
| Attention | 12 query / 4 key-value heads (GQA, 3:1) |
| Context | 2,048 tokens |
| Vocabulary | 50,304 (GPT-2 tokenizer) |
| Normalization | RMSNorm (pre-layer) |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Feed-forward | SwiGLU |

**Design choices.** RMSNorm with pre-layer normalization for stable deep networks;
RoPE for position-aware attention that generalizes beyond the training context;
Grouped-Query Attention to reduce the KV cache and accelerate generation; SwiGLU
for a more expressive feed-forward network; and weight tying between the embedding
and output head to improve parameter efficiency.

## Repository structure

```
llm_124m/                      # Model architecture + from-scratch pretraining
  config.py                    #   Hyperparameters
  model.py                     #   LanguageModel (RMSNorm / RoPE / GQA / SwiGLU)
  dataloader.py                #   GPT-2 tokenizer + packed training dataset
  train.py                     #   Pretraining script
  test_model.py                #   Sanity checks
posttraining/                  # Instruction-tuning pipeline
  instruction_tune_curated.py  #   Fine-tune script
  curate_dataset_final.py      #   Dataset curation
  DATASET_CURATION.md          #   How the dataset was built
continue_instruction_tuning.py # Continued tuning (Epochs 2-3)
app.py                         # Gradio demo (loads weights from the HF Space)
results/                       # Inference samples (before vs after)
  base_model_step4024.txt      #   Raw pretrained-model output
  original_vs_instruction_tuned.txt  # Base vs tuned, side by side
  before_after_comparison.txt  #   Additional before/after samples
requirements.txt               # Training dependencies
```

## Live demo

**[Hugging Face Space](https://huggingface.co/spaces/aishwarya/llm-124m-instruction-tuned)**

Model weights are hosted on the Space via Git LFS (GitHub caps files at 100 MB).
To run the demo locally:

```bash
pip install -r requirements.txt
# download model.pt from the HF Space, then:
python app.py
```

## Before vs. after instruction tuning

The `results/` folder contains representative outputs that illustrate the impact
of fine-tuning:

- `base_model_step4024.txt` — the pretrained model completes text without following
  instructions.
- `original_vs_instruction_tuned.txt` — identical prompts answered by the base and
  instruction-tuned models; the tuned model returns structured, on-topic responses
  in the `[INST] … [/INST]` format.
- `before_after_comparison.txt` — further before/after examples.

## Future work

- Expand the instruction dataset and add an automated evaluation harness.
- Apply 8-bit quantization for faster CPU inference.
- Extend continued training for further quality improvements.

## License

MIT (code). Dataset curation follows the licenses of its source corpora.
