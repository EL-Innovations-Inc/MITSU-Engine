# Changelog

## 0.2.0 — language-model stage 1

Everything needed to define, train and sample from a sub-word language model in numpy.

- **`MitsuLM` and `LMConfig`.** Decoder-only transformer with RMS normalisation, rotary
  position encoding, MitsuGLU feed-forwards and tied input/output embeddings. The config is
  stored in every saved model; `MitsuLM.load(path)` rebuilds it.
- **Cached generation.** `generate` reuses each layer's keys and values (6× to 13× faster in
  the benchmark), with temperature, top-k, top-p and greedy decoding and an optional stop
  token. Cached and recomputed generation are tested to be identical.
- **`BPETokenizer`.** Byte-level byte-pair encoding trained from text, lossless for any
  input, with special tokens and JSON save/load.
- **Disk-backed corpora.** `prepare_corpus` trains the tokenizer on the training part,
  writes `train.bin` / `val.bin`, and encodes large files in chunks with a result identical
  to whole-text encoding. `open_tokens` memory-maps them for `TextBatcher`.
- **New layers.** `RMSNorm`, `TiedHead`; `MultiHeadSelfAttention(rope=True)` and its
  incremental `step`; `TransformerBlock(norm="rms", rope=True)`.
- **Trainer.** Gradient accumulation, `save_state` / `load_state` for bit-exact resume,
  a JSON-lines log with gradient-norm tracking, and `evaluate_lm` (loss and perplexity).
- **Command line.** `python -m mitsu_engine prepare | train | generate | info`, with
  `--resume` and `--stop-after`.
- **Module system.** A parameter shared by several modules is listed, counted and
  optimised once.
- 57 new tests (140 in total), a language-model example and a generation benchmark.

Changed: `TransformerBlock` now names its parts `norm1`, `attn`, `norm2`, `ffn`, so
parameter names inside `MitsuGPT` checkpoints from 0.1.0 differ. `TextBatcher.windows` takes
`max_windows` and returns int64 arrays.

## 0.1.0

First release: module system with hand-written backpropagation, MITSU and MitsuGLU blocks,
attention, `MLP`, `DeepMitsuNet`, `MitsuGPT`, optimisers, trainer, checkpoints, gradient
checker, 83 tests.
