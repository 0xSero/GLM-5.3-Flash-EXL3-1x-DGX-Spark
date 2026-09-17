# Changelog

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
