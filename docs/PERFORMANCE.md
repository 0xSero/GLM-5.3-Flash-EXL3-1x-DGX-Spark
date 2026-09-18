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
~450 tok/s saturating. Knobs tested one variable at a time, everything else
byte-identical to the proven recipe (sampled cells, exact token-ID accounting):

| Exp | Change | 4,095-token cell | 131,071-token cell | Verdict |
|---|---|---|---|---|
| E1 | `TRELLIS_MAX_M` 32 → 128 | prefill 393.3 vs 427.0 | prefill 451.6 vs 454.2; decode 18.75 vs 19.00; acc 0.976 vs 0.99 | **no gain — stays 32** |
| E2 | MNBT 2048 → 7168 | **prefill 465.0 vs 427.0 (+8.9 %)** | **prefill 474.2 vs 454.2 (+4.4 %)**, decode 18.75 (unchanged), acc 0.985 | **shipped as default** |
| E3 | `kda_prefill_backend` b12x → flashkda (at MNBT 7168) | prefill 470.5 vs 465.0 | prefill 474.4 vs 474.2; decode 18.92; acc 0.993 | **neutral — stays b12x** |
| E4 | MNBT 7168 → 16384 | — | — | **fails: CUDA OOM in FP8 weight post-processing at util 0.93** |
| E5 | image v2: fused-MoE per-expert row cap 128 → 1024 (env) | **prefill 550.4 (+18.4 %)**, decode 19.13 | prefill 487.0 (+2.7 %); acc 0.987–0.996 | **root cause confirmed** |
| E6 | row cap 128 → 4096 | **prefill 570.5 (+22.6 %)**, decode 18.94 | **prefill 520.1 (+9.7 %)**, TTFT 252.4 s, decode 18.75–18.86 | **shipped as default (image v2)** |
| E7 | row cap → 7168 (hard bound) | — | — | **fails: engine init with 2.2 GB temps does not fit at util 0.93** |
| E8 | virtual-expert splitting (≤cap chunks, stock kernel) | prefill 435.9, decode 16.24 | prefill 352.1, decode 15.81 | **rejected: per-call bookkeeping + 8064-entry kernel walk outweigh load balancing at concurrency 6** |

### The fat-expert fallback: root cause and fix

`exllamav3_ext.exl3_moe` serves each expert from packed trellis weights inside
one fused launch, but it derives `max_tokens_per_expert` from
`temp_state_g.size(1)` and **silently skips any expert with more routed rows**.
The shipped overlay sized its temp buffers at 128 rows, so during prefill —
where ~57k routed slots spread over 288 experts puts most experts above 128 —
the fused kernel skipped nearly every expert and the overlay routed them into
`apply_exl3_python_loop`: a per-expert Python loop with `.tolist()` host syncs.
That loop was the ~475 tok/s prefill wall. The fix is an env-tunable row
capacity (`EXL3_FUSED_TEMP_ROWS`, default 4096 in image **v2**), allocated once
at load before CUDA graph capture. Dose–response across E5/E6 confirms the
mechanism: raising the cap monotonically lifts prefill (474 → 487 → 520 at
131k) while decode and acceptance stay flat.

Remaining gap to 600 at 131k (~13 %): what we know after E8, with evidence:

- Upstream's `EXL3_FAT_GROUPED` kernels tile fat experts across CTAs and are
  the right shape of fix, but the shipped extension is **K4-only** —
  `exl3_fat_moe.cu` hardcodes `FM_PACKED_WORDS = 64` (int16 words per 16×16
  **K4** tile) with no bit dispatch. This artifact is mixed **K2/K3**
  (per-layer uniform, 2 or 3 bits), so the extension cannot be adopted as-is;
  porting it means writing bit-templated dequant/pipeline kernels for K2/K3
  (the ext itself does compile against this image's exllamav3 0.0.43 in 25 s —
  verified — so only the K-generality is missing).
- A virtual-expert splitting alternative (replicate per-expert pointer tables
  S=⌈MNBT/cap⌉ times, split routed rows into ≤cap chunks with device-side
  arithmetic, keep the stock kernel) was implemented and measured (E8): it
  regresses — 435.9/352.1 tok/s prefill and 15.8–16.2 decode — because the
  per-call split bookkeeping and the kernel's per-group walk over 1152×7
  virtual experts cost more than the load balancing recovers at this image's
  concurrency (6 groups). Rejected on receipts
  (`receipts/e8b-split-regression-probe.log`).

Conclusion: with scheduler knobs exhausted and the split rejected on evidence,
600 tok/s prefill on one Spark requires the bit-templated K2/K3 grouped
kernels — a bounded, well-understood kernel project (compile path proven, call
path mapped: `layer._exl3_ptrs` tables already match the extension's expected
inputs), not further launcher tuning.

### Cold-start validation of the published recipe (2026-09-18)

`./start.sh` was executed on a **fresh `git clone` of this repository** on a
DGX Spark (image already local; weights staged; the MTP draft view derived
on-host by `tools/prepare_mtp_draft.py`; anonymous HF fetch of the public
weights repo verified separately). Result, measured against the server it
launched:

| Check | Result |
|---|---|
| Served model | `glm-5.3-flash`, `max_model_len 262144` |
| KV pool at MNBT 7168 | 498,073 tokens = 1.90× a full 262,144-token request |
| Decode (4,095-token cell, sampled) | **19.13 tok/s** (native counter 19.13) |
| Prefill (4,095-token cell) | **466.8 tok/s** |
| Draft acceptance | **0.996** cell-level; cumulative counters 909/915 = 99.3 % |

Receipt: `receipts/coldstart-1xspark-20260918T021243Z.json`. The tuned recipe
reproduces exactly through the public path.

### The path to 600 tok/s prefill (port plan)

The 2× upstream kit reaches ~1,500 tok/s prefill with a custom grouped-MoE
prefill kernel. Concrete port inventory from their tree: `overlay/exl3_fat_moe.cu`
(656 lines) + `overlay/exl3_fat_moe.cuh` + `overlay/exl3_fat_gemm.{cu,cuh}` +
`overlay/build_exl3_fat_moe_ext.py`, replacing the per-expert host loop with
three device-driven launches per MoE layer (gather, gate/up + SwiGLU,
down + scatter) built from device-side segment tables. Integration point in
this fork: the grouped-MoE prefill call path of the pinned vLLM fork's EXL3
overlay. Multi-day, GPU-qualified work — deliberately not attempted blind in
this release, which prefers a receipted recipe over a faster unproven one.

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
