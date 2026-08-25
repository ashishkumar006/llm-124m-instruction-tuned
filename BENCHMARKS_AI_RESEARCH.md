# Benchmarks — AI Research Angle

This document frames the project through the lens of **machine-learning research
methodology**: parameter-efficient fine-tuning, training efficiency, and empirical
behavioral evaluation. All figures are computed from the project's own checkpoints
and logs.

## Parameter-efficient fine-tuning (LoRA)

Instruction tuning used LoRA (rank `r=8`, alpha `16`) on `q_proj`, `v_proj`,
`k_proj`, `out_proj` across 12 layers.

| Metric | Value |
|--------|-------|
| LoRA trainable parameters | **~0.49M** (491,520) |
| Full model parameters | ~152.8M |
| Fraction of weights updated | **0.322%** |
| Trainable-parameter reduction | **~311× fewer** than full fine-tuning |
| Trainable params per 4B pretrain token | 1.2 × 10⁻⁴ |

**Takeaway:** the model was adapted to instruction following while updating only
~0.3% of weights — a concrete demonstration of parameter-efficient transfer learning
rather than retraining the full network.

## Training & convergence

| Metric | Value | Source |
|--------|-------|-------|
| Pretraining tokens | 4.0B | `config.total_tokens` |
| Base pretrain loss → instruction-tuned val loss | 3.7213 → 2.0832 | logs |
| Perplexity (base → tuned) | 41.32 → 8.03 | `exp(loss)` |
| **Perplexity reduction** | **~80.6%** | computed |
| Epoch 1 → 2 val loss | 2.0832 → 2.0778 | `validation_log_continued.txt` |

**Takeaway:** a single epoch of LoRA instruction tuning on a curated dataset cut
language-model perplexity by ~80% relative to the base model, showing strong
task-adaptation signal from a small, high-quality dataset.

## Behavioral evaluation (instruction adherence)

| Metric | Value |
|--------|-------|
| `[INST] … [/INST]` format adherence | **100%** on test prompts |
| Base model behavior | text completion only (no instruction following) |

**Takeaway:** a manual before/after comparison (`results/`) shows the base model
merely completes text, while the tuned model consistently produces structured,
on-format responses — a qualitative behavioral shift attributable to fine-tuning.

## Research-relevant skills demonstrated

- From-scratch transformer implementation (RMSNorm, RoPE, GQA, SwiGLU).
- Pretraining at scale (4B tokens) with warmup + cosine LR schedule.
- Parameter-efficient adaptation (LoRA / PEFT) and weight merging for deployment.
- Empirical evaluation: loss/perplexity tracking, before/after behavioral analysis.
- Reproducible pipeline: dataset curation → pretrain → fine-tune → deploy.

## Notes

Metrics reflect language-modeling quality and instruction-format adherence, not
task accuracy (e.g. QA correctness). Figures are derived from project artifacts; see
`BENCHMARKS.md` for the full source list.
