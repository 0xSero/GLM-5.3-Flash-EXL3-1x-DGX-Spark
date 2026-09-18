# Performance on 1× DGX Spark

All numbers measured on spark-557f (DGX Spark, GB10, 128 GB unified memory,
arm64) with the exact recipe this repository ships. UTC timestamps; every
number traces to a receipt file.

## Full-context-window ladder (2026-09-17)

Single stream, sampled (`temperature 1.0`, `top_p 0.95`, thinking on, no greedy
anywhere), native MTP depth 2, CUDA graphs `FULL_DECODE_ONLY` on target and
draft, `fp8_ds_mla` KV, block 256, 262,144-token window.

| Input tokens | Decode tok/s (aggregate) | Native decode tok/s | Prefill tok/s (from TTFT) | TTFT | Draft acceptance | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1,023 | 19.19 | 19.31 | 391.5 | 2.61 s | 0.98 | MATCHED_SUSTAINED |
| 4,095 | 19.22 | 19.35 | 427.0 | 9.59 s | 0.98 | MATCHED_SUSTAINED |
| 16,383 | 18.95 | 19.03 | 442.9 | 36.99 s | 0.98 | MATCHED_SUSTAINED |
| 65,535 | 18.82 | 18.92 | 454.3 | 144.26 s | 0.99 | MATCHED_SUSTAINED |
| 131,071 | 19.00 | 19.10 | 454.2 | 288.58 s | 0.99 | MATCHED_SUSTAINED |
| 199,999 | 18.70 | 18.79 | 452.7 | 441.83 s | 0.99 | MATCHED_SUSTAINED |
| 260,095 | 18.89 | 18.97 | 451.8 | 575.65 s | 1.00 | MATCHED_SUSTAINED |

- Decode varies 2.7 % end to end across the whole window.
- A 260,095-token prompt + 683 generated tokens completes without OOM; the KV
  pool is 727,449 tokens = 2.77× one full-length request.
- Prefill saturates at ~450 tok/s from ~64k input onward.

**Receipt.**
[receipts/mtp-ctx-sweep-full-20260917T160505Z.json](receipts/mtp-ctx-sweep-full-20260917T160505Z.json),
sha256 `a8023b2a9574…` (manifest: [RECEIPTS.sha256](RECEIPTS.sha256); the
admission receipt of the same recipe is
[receipts/mtp-serve-proof-20260917T122332Z.json](receipts/mtp-serve-proof-20260917T122332Z.json)).
Engine admission: weights 102.29 GiB, KV 727,449 tokens, six CUDA graph
capture receipts (target + draft), `/v1/models` reports `glm-5.3-flash`,
`max_model_len 262144`.

## Method (so the numbers mean something)

- Exact per-chunk `token_ids` returned by the server and compared client-side
  (`682 = 682` on measured cells); decode tok/s computed inside
  matched-sustained windows (≥ 30 s), not from wall clock.
- Cross-checked against vLLM's own counters
  (`vllm:request_decode_time_seconds`, `vllm:request_generation_tokens`);
  request-count deltas match, so vLLM-side and client-side totals agree.
- Speculative acceptance from `vllm:spec_decode_num_{draft,accepted}_tokens_total`.
- Prompt constructed by tokenization bisection to hit exact input lengths.
- Prefill tok/s = prompt tokens ÷ server-measured TTFT; both quoted so you can
  recompute.

## Squeeze log (prefill work, 2026-09-17/18)

Baseline recipe (`VLLM_EXL3_TRELLIS_MAX_M=32`, MNBT 2048) prefills at
~450 tok/s saturating. Knobs under test, one variable at a time, everything
else byte-identical to the proven recipe (sampled cells, exact token-ID
accounting):

| Exp | Change | Result (131,071-token cell) | Verdict |
|---|---|---|---|
| E1 | `TRELLIS_MAX_M` 32 → 128 | prefill 451.6 vs 454.2 tok/s; decode 18.75 vs 19.00; acceptance 0.976 vs 0.99 | **no gain — default stays 32** |
| E2 | MNBT 2048 → 7168 | (running; this table updates with the receipt) | pending |

The trellis tile cap is therefore not the prefill bottleneck. Receipts:
`receipts-557f/prefill-e1-trellisM128-20260917T232819Z/` (probe log with all
four cells; E1 container `glm53-k2-mtp-b12x-d2-e1` preserved, not removed).

For scale: the 2× upstream kit reaches ~1,500 tok/s prefill via a custom
grouped-MoE prefill kernel (`EXL3_FAT_GROUPED`, gather/gate-up/down fused into
three device-driven launches), 4 bpw weights, and two GPUs. Porting that
kernel into this image's vLLM fork is the known big lever for a future revision;
it is GPU-qualified multi-day work and deliberately out of scope for the first
public cut, which prefers a receipted recipe over a faster unproven one.

## Decode notes

- 19.2 tok/s aggregate decode with sampling on and thinking on. Depth-2 native
  MTP drafts 2 tokens per step; acceptance ≥ 0.95 at `temperature 1.0`
  end to end.
- Depth-1 fallback (`SPEC_DEPTH=1`) serves but measures materially slower
  (~13.3 tok/s class on the older image lineage); depth 2 with the B12X backend
  is the shipped default.
- Earlier probes that reported 7–13 tok/s used chunk-count accounting on
  coalescing SSE streams — an accounting bug, corrected by exact token-ID
  accounting; do not compare those numbers.
