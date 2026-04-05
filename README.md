# Latency-Constrained Sequence Labelling on CPU

This repository supports an NLP course case study on **sequence labelling under production-like CPU-only inference**. It benchmarks several taggers on **part-of-speech (POS)** and **named-entity recognition (NER)** tasks, measuring **quality** (F1), **latency** (percentiles), **throughput**, and **memory** (process RSS). The code compares classical and neural models and explores optional runtime optimizations such as batching, length-aware batch ordering, `torch.compile`, quantization (torchao), and ONNX Runtime for transformers.

## Learning goals

- Compare multiple model families on the same splits: **HMM** (two implementations), **linear-chain CRF**, **pretrained BiLSTM-CRF (Flair)**, and **pretrained transformer (Hugging Face)**.
- Report **micro/macro F1**, **latency** (p50/p90/p99), **throughput** (sequences per second), and **memory** deltas alongside the benchmark configuration (batch size, smart batching, quantization, ONNX).
- Relate findings to a **real-time tagging** scenario where both accuracy and speed matter. The full results, figures, and model-selection discussion are in [`poster.pdf`](poster.pdf).

## Repository layout

| Path | Role |
|------|------|
| [`sequence_tagging_benchmark/taggers.py`](sequence_tagging_benchmark/taggers.py) | Shared tagger implementations and `evaluate()` benchmark harness |
| [`notebooks/pos_tagging.ipynb`](notebooks/pos_tagging.ipynb) | POS pipeline (data load → train classical models → benchmark → CSV) |
| [`notebooks/ner.ipynb`](notebooks/ner.ipynb) | NER pipeline |
| [`notebooks/visualization.ipynb`](notebooks/visualization.ipynb) | Plots from the committed CSV artifacts |
| [`artifacts/pos_tagging_results.csv`](artifacts/pos_tagging_results.csv) | POS benchmark rows (one per configuration) |
| [`artifacts/ner_results.csv`](artifacts/ner_results.csv) | NER benchmark rows |
| [`pyproject.toml`](pyproject.toml) | Package metadata and dependencies (`sequence-tagging-benchmark`, Python ≥ 3.12) |
| [`poster.pdf`](poster.pdf) | **Primary deliverable:** all benchmark results, figures, and recommendations |

## Environment setup

- **Python** 3.12 or newer.
- From the repository root, install the package in editable mode (pulls dependencies from `pyproject.toml`):

  ```bash
  pip install -e .
  ```

- First runs **download** Hugging Face datasets and pretrained weights (Flair, transformers); ONNX export for the transformer baseline uses additional disk and time.
- The notebooks set `torch.set_num_threads(NUM_THREADS)` (e.g. 8) for reproducible CPU threading; adjust to match your machine.

## How to reproduce

1. Complete **Environment setup** above.
2. Run [`notebooks/pos_tagging.ipynb`](notebooks/pos_tagging.ipynb) and [`notebooks/ner.ipynb`](notebooks/ner.ipynb) end-to-end. They train classical taggers on the training split, run all configurations on the test split, and aggregate rows into the `artifacts/*.csv` files (you can overwrite them when re-benchmarking).
3. Optionally open [`notebooks/visualization.ipynb`](notebooks/visualization.ipynb) to regenerate figures from those CSVs.

A network connection is required for Hugging Face `datasets` and model downloads.

## Models and configurations benchmarked

**Tasks and data**

- **POS:** Universal Dependencies English EWT (`universal_dependencies` / `en_ewt`), **UPOS** tags.
- **NER:** OntoNotes-style data (`tner/ontonotes5`), token-level BIO labels (fixed `id2label` map in the NER notebook).

**Pretrained neural checkpoints (by task)**

- **POS — Flair BiLSTM-CRF:** `flair/upos-multi` (tag type `upos`).  
- **POS — Transformer:** `KoichiYasuoka/roberta-base-english-upos` (UPOS labels are normalized in the notebook for comparison with gold UPOS).
- **NER — Flair:** `flair/ner-english-ontonotes-fast` (tag type `ner`).  
- **NER — Transformer:** `pitangent-ds/roberta-base-ontonotes`.

**Classical models (trained in-notebook)**

- **HMM:** NLTK `HiddenMarkovModelTagger` and a **hmmlearn** categorical HMM with Viterbi decoding.
- **CRF:** `sklearn-crfsuite` with hand-crafted word features.

**Runtime variants (reflected in CSV row names)**

- Neural models: batch sizes **1** and **128**, optional **smart batching** (sort by sequence length before batching).
- BiLSTM and transformer: **dynamic** and **weight-only** INT8-style quantization via torchao (see notebooks for exact configs).
- Transformer: **ONNX Runtime** path via Optimum (`ONNXTransformerTagger`, batch 128 in the committed runs).

## Metrics

All metrics come from `BaseSequenceLabellingTagger.evaluate()` in [`sequence_tagging_benchmark/taggers.py`](sequence_tagging_benchmark/taggers.py):

| Metric | Meaning |
|--------|---------|
| `micro_f1`, `macro_f1` | Token-level F1 (`sklearn.metrics.f1_score` on flattened tags). The harness supports sequence-level F1 via seqeval if `evaluate_sequence=True`, but the notebooks use the default. |
| `p50_latency_ms`, `p90_latency_ms`, `p99_latency_ms` | Per-sentence latency in milliseconds: each batch’s wall time is divided equally across sentences in that batch. |
| `latency_variance_ms` | Variance of per-sentence latencies (ms²). |
| `throughput_seq_per_sec` | Test set size divided by total measured inference time. |
| `inference_memory_delta_mb` | Peak RSS during inference minus RSS at start (non-negative clamp). |
| `final_memory_retained_mb` | RSS after inference minus RSS at start (can be negative if memory is released). |

**Procedure notes:** By default, **5 warmup** forward passes run on the first test sentence (`warmup_iters=5`). Memory samples use the current process RSS (`psutil`).

## Results

**All benchmark results, figures, and the model-selection discussion are in [`poster.pdf`](poster.pdf)** at the repository root. Raw numeric tables for reproduction and plotting remain in [`artifacts/pos_tagging_results.csv`](artifacts/pos_tagging_results.csv) and [`artifacts/ner_results.csv`](artifacts/ner_results.csv).
