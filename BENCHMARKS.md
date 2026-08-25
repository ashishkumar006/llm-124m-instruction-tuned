# Benchmarks

All metrics below are computed directly from this project's checkpoints and training
logs — no estimated or external figures.

## Model scale

| Metric | Value | Source |
|--------|-------|--------|
| Parameters | ~152.8M (weight-tied: ~114M) | `instruction_tuned_curated_epoch_3.pt` state_dict |
| Pretraining tokens | 4.0B | `config.total_tokens` |
| Context length | 2,048 tokens | `config.max_seq_length` |
| Vocabulary | 50,304 (GPT-2 tokenizer) | `config.vocab_size` |

## Training loss & perplexity

Perplexity is computed as `exp(loss)`.

| Stage | Loss | Perplexity |
|-------|------|------------|
| Base pretrained (step 7629) | 3.7213 | 41.32 |
| Instruction-tuned (epoch 1) | 2.0832 | 8.03 |
| Instruction-tuned (epoch 2) | 2.0778 | 7.99 |
| Instruction-tuned (epoch 3) | 2.0778 | 7.99 |

- **Perplexity reduction, base → instruction-tuned: ~80.6%** (41.32 → 8.03).
- Epoch 1 → epoch 2 validation loss improved 2.0832 → 2.0778 (perplexity 8.03 → 7.99).

## Instruction formatting

- **100% `[INST] … [/INST]` format adherence** on test prompts, verified in
  `results/original_vs_instruction_tuned.txt` (every output follows the instruction
  format, versus the base model which only completes text).

## Fine-tuning method (LoRA)

Instruction tuning was performed with **LoRA (Low-Rank Adaptation)** via PEFT, then
merged into the base weights for deployment.

| Setting | Value | Source |
|---------|-------|--------|
| Method | LoRA (PEFT) | `instruction_tune_curated.py`, `continue_instruction_tuning.py` |
| Rank (`r`) | 8 | `LoraConfig` in training scripts |
| Alpha | 16 | `LoraConfig` in training scripts |
| Dropout | 0.1 | `LoraConfig` in training scripts |
| Target modules | `q_proj`, `v_proj`, `k_proj`, `out_proj` | `LoraConfig` in training scripts |
| Deployment | Merged (`merge_and_unload`) into base model | `save_checkpoint()` in training scripts |

- LoRA was applied for both the post-training / instruction-tuning stage (Epoch 1) and
  the continued tuning stage (Epochs 2–3).
- The published checkpoints are **fully merged models** (no separate adapter files):
  111 tensors, ~152.8M parameters, 0 LoRA keys. The `lora_r` / `lora_alpha` values in
  the checkpoint `config` are retained metadata from this LoRA setup.

## How these are derived

- **Parameters**: sum of all tensor element counts in
  `instruction_tuned_curated_epoch_3.pt` → 152,783,616.
- **Base loss 3.7213**: reported in `inference_results/inference_step4024.txt`
  (Step 7629).
- **Instruction-tuned losses**: from `POSTTRAINING_RESULT/validation_log_continued.txt`
  (epoch 1 = 2.0832, epoch 2 = 2.0778, epoch 3 = 2.0778).
- **Perplexity**: `exp(loss)`; reduction = `(exp(base) − exp(post)) / exp(base)`.
- **Format adherence**: manual check of the before/after samples in `results/`.

## Notes

These metrics reflect language-modeling quality and instruction-format adherence.
They are not task-accuracy scores (e.g. QA correctness); factual accuracy is limited
by model size, as discussed in the project README.
