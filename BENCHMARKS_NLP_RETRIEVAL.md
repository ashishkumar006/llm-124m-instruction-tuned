# Benchmarks — NLP & Retrieval Angle

This document frames the project for **NLP / retrieval** roles, focusing on dataset
engineering, curation, and evaluation — the parts of the work most relevant to
language-data and information-retrieval positions. All figures are computed from the
project's dataset artifacts.

## Dataset curation pipeline

The instruction-tuning dataset was built by curating a larger hybrid corpus.

| Metric | Value | Source |
|--------|-------|--------|
| Original hybrid corpus | 15,000 examples | `hybrid_dataset.json` |
| Curated (English-only) dataset | 7,522 examples | `curated_dataset_english_only.json` |
| **Curation removal rate** | **46.7%** | `curation_documentation.json` |
| Non-English examples removed | 7,478 (49.9% of hybrid) | `hybrid_dataset.json` vs English-only |
| Math examples removed | 6,257 (41.7% of original) | `curation_documentation.json` |

**Takeaway:** a deliberate data-cleaning pass that dropped ~47% of examples —
prioritizing English, removing tasks the model cannot learn (arithmetic) and
low-signal categories — to improve training efficiency and output quality.

## Curated dataset composition

Category mix of the final 8,000-example curated set:

| Category | Count | Share |
|----------|-------|-------|
| General Q&A | 4,280 | 53.5% |
| Code | 2,691 | 33.6% |
| Creative | 651 | 8.1% |
| Data visualization | 378 | 4.7% |

**Takeaway:** a balanced, task-focused mix skewed toward general instruction
following and code — the capabilities a small model can actually acquire.

## Instruction characteristics

| Metric | Value |
|--------|-------|
| Average instruction length | ~661 characters |
| Max instruction length | 59,157 characters |
| Format | `[INST] … [/INST]` instruction/response pairs |

**Takeaway:** prompts span short factual queries to long, complex multi-part
instructions — useful for evaluating instruction robustness and retrieval-style
long-context handling.

## Evaluation & behavioral signal

| Metric | Value |
|--------|-------|
| `[INST]` format adherence (tuned model) | 100% on test prompts |
| Base model behavior | text completion only |

**Takeaway:** before/after comparison (`results/`) demonstrates that curation +
fine-tuning shifted the model from passive completion to active instruction
following — the core NLP behavior retrieval-augmented and instruction-tuned systems
rely on.

## NLP / retrieval-relevant skills demonstrated

- Instruction dataset construction, cleaning, and English-language filtering.
- Category-balanced curation with explicit removal of low-signal / unlearnable tasks.
- Prompt-format design (`[INST] … [/INST]`) for structured instruction following.
- Before/after behavioral evaluation of language-model outputs.
- Reproducible data pipeline: scrape/hybrid → filter → categorize → curate → train.

## Notes

Dataset sizes and category shares are exact counts from the JSON artifacts. The
"expected 15–25% quality improvement" figure quoted in `curation_documentation.json`
is a project estimate, not a measured benchmark, and is not reported here as a result.
