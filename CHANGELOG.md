# Changelog

## 1.2.0 — 2026-09-19

The `mul1` codebook gate is fixed and the mosaic's real blocker chain is measured, on a GB10 against
the published weights: [overlays/exl3-mul1/](overlays/exl3-mul1/), receipt
[docs/receipts/mul1-overlay-20260919T085502Z.txt](docs/receipts/mul1-overlay-20260919T085502Z.txt).

- **The MTP runtime's EXL3 overlay now accepts `mul1`.** The codebook selects the decode math end to
  end: `LinearEXL3` is built with `mul1=` and `exl3_moe` receives the real
  `(gate_mcg, gate_mul1, …)` flags instead of six hardcoded `True, False`. Marker parameters are named
  after the codebook (`w13_mul1` / `w2_mul1`), which is also what lets vLLM's generic expert mapping
  resolve `…gate_proj.mul1`, and marker validation checks `0x83DCD12D` rather than the mcg constant.
  The `mcg` path is unchanged.
- **Mixed-precision layers are allocated per layer.** A mosaic's `bits` is an average (2.05); the
  config now distils a per-layer rate from the artifact's `tensor_storage` ledger. On the published
  mosaic that recovers 30 layers at 2 bpw and 12 at 3 bpw — layers 3, 32, 33, 36–44, exactly the
  published composition — instead of under-allocating the upgraded layers' trellis.
- **Verified on hardware.** `LinearEXL3` decodes both mosaic rates (layer 4 `K=2`, layer 3 `K=3`), and
  the same bytes decoded as `mcg` differ by 1.6× — proof the codebook is math, not a label.
- **Correction: the earlier MTP diagnosis was wrong.** `KeyError: 'layers.0.mlp.down_proj.mul1'` was
  read in 1.1.2 as a naming-lineage mismatch between `experts.E.*` and `layers.N.mlp.down_proj.*`.
  With `first_k_dense_replace = 3`, layer 0 has no experts: it is a dense MLP, and the mosaic really
  does ship that tensor. No key remap is needed; the runtime simply has no loader for the artifact's
  non-expert quantized tensors.
- **The two remaining blockers, counted.** (1) The image's pinned exllamav3 0.0.43 fused MoE kernel
  raises `MoE kernel: Only mcg codebook is currently supported`; upstream already selects mul1
  (`cb_idx = gate_mul1 ? 1 : 0`), so this is an image rebuild. (2) Beyond its 37,152 routed-expert
  trellis tensors the mosaic also quantizes 128 attention projections, 129 shared experts, 9 dense-MLP
  tensors, 172 vision-tower tensors, `layers.45.eh_proj` and `lm_head`, all of which this runtime hands
  to `UnquantizedLinearMethod`. An EXL3 `LinearMethod` for plain linears is the bulk of the work left.

## 1.1.2 — 2026-09-18

A full validation pass over the mosaic, served from a directory fetched through the published path and
hash-checked: [docs/VALIDATION.md](docs/VALIDATION.md). No shipped behaviour changes.

- **Speed, measured cleanly.** Prefill **503.5 tok/s marginal** (r² 0.9999) and decode **10.79 tok/s
  mean** (10.61–10.94 across a 238× range of prompt length), from a streaming sweep with a unique nonce
  opening every prompt so radix caching cannot flatter it. The earlier ladder (`FILLER × N` prompts,
  each rung a prefix of the next) is kept as the record of that method error, with the server's
  `#cached-token` accounting as the explanation.
- **The cache demonstration failed, which is itself the finding.** A repeated 14 k prompt in a saturated
  pool was re-prefilled twice rather than served; the sweep before it had inserted 616,956 tokens into a
  595,200-token pool. Prefix reuse works normally when the pool has room, as the vision probes on the
  same server show (one step, 64 new tokens, 7,936 cached). See
  [docs/receipts/cache-demo-analysis.md](docs/receipts/cache-demo-analysis.md).
- **Vision at full resolution.** 4096×4096 accepted → 7,921 image tokens, 99 % of the artifact's declared
  8,000-per-image ceiling, with coordinate-accurate reading at 512² and 2048². A verified end-to-end word
  answer was not obtained: the probe's task does not fit its token budget. Stated as a gap.
- **MTP: the gate was the first blocker, not the only one.** With the `mul1` codebook gate relaxed, the
  MTP recipe passes validation and dies at weight loading —
  `KeyError: 'layers.0.mlp.down_proj.mul1'` at `glm5next/nvidia/model.py:904` — because the mosaic's keys
  are `model.language_model.layers.N.mlp.experts.E.*` (turboderp/SGLang lineage) while the vLLM loader
  expects `layers.N.mlp.down_proj.*`. Layer 0 is a stock layer, so this is a naming-lineage mismatch.
  MTP here needs a key remap, then the `mul1` decode path:
  [docs/receipts/m1-findings.md](docs/receipts/m1-findings.md).
- Context: `max_model_len 262144` declared, 236,510 prompt tokens exercised end to end, 128,000-token
  output budget accepted, 8,192 reasoning tokens in one response without server truncation,
  deterministic at `temperature 0`.

## 1.1.1 — 2026-09-18

Two honesty corrections to the experiment record. No shipped behaviour changes.

- **E9 did not measure the grouped kernels.** Its launcher exports
  `EXL3_FAT_GROUPED=1` and `EXL3_FUSED_TEMP_ROWS=4096` but mounts no overlay over
  the container's `exl3.py`, so both variables were read nowhere (the image's
  module has 0 occurrences of either and hardcodes `TEMP_ROWS_FUSED = 128`).
  The guard asserted the base image id, which a dropped overlay mount leaves
  unchanged, so it passed. E9's numbers are a valid re-measurement of the
  **baseline**, and the v3 K2/K3 grouped kernels remain performance-unmeasured.
  Full record and the fail-closed test design:
  `docs/receipts/quarantine-e9-invalid.md` and [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- **`docs/RECEIPTS.sha256` now lists only files a cloner has.** Four probe logs
  had been hashed but never published (`.gitignore` excludes `*.log` and they
  were not force-added); they are no longer present on either host. Their
  digests are preserved in `docs/receipts/UNAVAILABLE.txt` with what survives of
  each experiment. One cited log was recovered and is now published —
  `receipts/mtp-serve-probe-b12x-20260917T122134Z.log`, whose sha256 matches the
  digest recorded before it went missing. The manifest verifies 13/13, and
  `!docs/receipts/*.log` keeps evidence past the ignore rule.

## 1.1.0 — 2026-09-18

The M288-12L mosaic is now reachable from this kit, and its weights are published.

- **New artifact path**: `./start-mosaic.sh` serves the M288-12L mosaic
  (mixed 2.05/3.05 bpw, codebook `mul1`) — download (96 GB, pinned revision,
  verified against the model repo's own sha256 manifest), census preflight,
  then SGLang with the flag set its confirming run was measured with.
- **Weights published**:
  `0xSero/GLM-5.3-Flash-EXL3-Spark` @
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
