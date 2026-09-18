# Changelog

## 1.1.0 — 2026-09-18

The M288-12L mosaic is now reachable from this kit, and its weights are published.

- **New artifact path**: `./start-mosaic.sh` serves the M288-12L mosaic
  (mixed 2.05/3.05 bpw, codebook `mul1`) — download (96 GB, pinned revision,
  verified against the model repo's own sha256 manifest), census preflight,
  then SGLang with the flag set its confirming run was measured with.
- **Weights published**:
  `0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L` @
  `2642851741fc833764e77d03039117be559dc83e`, 12 shards, 96,105,136,306 B,
  every file hash-verifiable. Previously the mosaic was rebuild-only.
- **Stated limit, not a bug**: this kit's vLLM/MTP overlay rejects `mul1` at
  config validation (`exl3.py:568`), so the mosaic runs without MTP at
  ~9–10 tok/s against this kit's 18.7–19.2. Why, and what closing it would
  take: [docs/MOSAIC.md](docs/MOSAIC.md).
- Measured mosaic quality (existing receipts, not new claims): top-1 0.80842
  vs the base's 0.78902, KL 0.32522 vs 0.38378, 32/32 panel rows, and it
  regresses on neither MMLU nor GPQA.

## 1.0.0 — 2026-09-18

First public 1× DGX Spark cut. Forked from MiaAI-Lab's 2× EXL3 kit.

- Single-Spark serving of GLM-5.3-Flash EXL3 K2 2.0 bpw (`mcg`) with the
  model's native MTP layer as depth-2 speculator; CUDA graphs
  `FULL_DECODE_ONLY` on target and draft; 262,144-token context validated
  end to end.
- Measured: 18.7–19.2 tok/s single-stream decode (sampled, thinking on)
  across the full window; ~450 tok/s saturating prefill; KV pool 727,449
  tokens (2.77× full-length request).
- Single command: `./start.sh` (public GHCR image + public HF weights +
  CPU-only MTP draft derivation).
- Known prefill headroom vs the 2× kit documented in
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md) (custom fat-MoE prefill kernel
  not yet ported).
