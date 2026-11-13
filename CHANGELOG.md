# Changelog

## 0.3.0 — language-model stage 2: the PyTorch twin

`mitsu_engine.twin`: the same language model in PyTorch, tested to reproduce the numpy
engine. PyTorch is optional (`pip install ".[torch]"`); the core still needs numpy only and
never imports it.

- **`twin.TorchMitsuLM`.** `MitsuLM` with PyTorch modules: the same `LMConfig`, parameter
  names, generation procedure and checkpoint file. `from_engine`, `to_engine`, `save`, `load`.
  Its initial weights are drawn by the numpy engine, so the same seed gives both engines
  the same start.
- **Parity tests.** With identical weights, the logits, the loss, every gradient (numpy's
  hand-written backward against autograd) and the cached generation path agree to about
  1e-14 in float64 and 1e-5 in float32, and generated tokens are identical for greedy,
  temperature, top-k and top-p sampling.
- **The same training run.** `twin.EngineAdamW` and `twin.clip_grad_norm` reproduce the
  engine's update rules; `twin.TorchTrainer` is the engine's `Trainer` with the optimisation
  step replaced. In float64, 30 steps leave the two engines with the same weights to 7e-15.
- **One state format.** Models and training states written by either engine load in the
  other; a run can change engine midway.
- **Mixed precision.** `amp="bf16"` and `amp="fp16"` (with a gradient scaler). MITSU, RMSNorm
  and the position rotation compute in float32 under autocast.
- **Command line.** `train` and `generate` take `--backend torch --device ... --amp ...`;
  the new `parity` command measures the twin against the engine on a given device and exits
  with 0 or 1.
- **`write_checkpoint` / `read_checkpoint`.** The checkpoint layout now lives in these two
  functions, used by both engines.
- 50 new tests (190 with PyTorch installed; 140 without, the rest skipped), an example,
  two benchmarks, `docs/TWIN.md`, and a second CI job that installs PyTorch.

Not done: nothing in this release has been run on a GPU. `python -m mitsu_engine parity
--device cuda` is the check to run first on one.

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
