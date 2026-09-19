# The M288-12L mosaic in this kit

This fork serves the **EXL3 K2 2.0 bpw `mcg`** staging quant with native MTP on vLLM/B12x. The
**M288-12L mosaic** is a different artifact from the same model family, and **this kit's runtime cannot
serve it**. This document exists so nobody spends an afternoon rediscovering that, and so the mosaic is
one command away — on its own runtime.

## Why this kit can't serve the mosaic

The mosaic is codebook **`mul1`** (it is built on the turboderp EXL3 line); this kit's overlay only
implemented **`mcg`**. The rejection happened at config validation, before a single weight was read:

```
ValueError: this overlay only implements codebook=mcg; got 'mul1'
```

Gate: `exl3.py:568` in both MTP-capable images (`glm53-reap-native-mtp:…-r4-expert-fp8` and
`glm53-b12x-exl3:jovian-3aada677-r3`). The codebook is not a label — it selects the decode math (the
trellis codebook tables consumed by `LinearEXL3`).

**That gate is the first blocker of three.** [`overlays/exl3-mul1/`](../overlays/exl3-mul1/) removes it:
the overlay now accepts `mul1`, names the marker parameter after the codebook, passes the real
`(gate_mcg, gate_mul1, …)` flags to `exl3_moe`, and — because a mosaic is mixed precision — sizes each
layer's trellis from the artifact's own `tensor_storage` ledger instead of one global average rate.
Measured on a GB10 against the published weights, `LinearEXL3` decodes both mosaic rates
(layer 4 `K=2`, layer 3 `K=3`) and the same bytes read as `mcg` differ by 1.6×. Receipt:
[`receipts/mul1-overlay-20260919T085502Z.txt`](receipts/mul1-overlay-20260919T085502Z.txt).

Two blockers remain, and both are outside that file:

1. **The image's fused MoE kernel has no mul1 instances.** exllamav3 0.0.43 (`c5d9c657`, the version
   pinned in the runtime image) raises `RuntimeError: MoE kernel: Only mcg codebook is currently
   supported`. Current upstream exllamav3 selects the codebook in `exl3_moe.cu`
   (`const int cb_idx = gate_mul1 ? 1 : 0;`) and instantiates its 32/64-row tiles for mul1 only, so the
   fix is an image rebuild against a newer `exllamav3_ext`, not new kernel work. Until then the routed
   experts fall back to the Python `LinearEXL3` loop.
2. **The mosaic quantizes much more than the routed experts.** Counted from its index: 37,152
   routed-expert trellis tensors, plus 128 attention projections, 129 shared experts, 9 dense-MLP
   tensors (layers 0–2), 172 vision-tower tensors, `layers.45.eh_proj` and `lm_head`. The overlay
   returns `UnquantizedLinearMethod` for every `LinearBase`, so none of those has a loader. Serving the
   mosaic here needs an EXL3 `LinearMethod` for plain linears, including splitting vLLM's merged
   `gate_up_proj` back into the two trellis sets the checkpoint stores separately. That is the bulk of
   the remaining work and it has not been done.

**Correction (2026-09-19).** This document previously read the M1 experiment's
`KeyError: 'layers.0.mlp.down_proj.mul1'` as a naming-lineage mismatch — the loader addressing experts as
`layers.N.mlp.down_proj.*` while the artifact names them `…experts.E.*`. That was wrong. With
`first_k_dense_replace = 3`, layer 0 has no experts at all: it is a dense MLP, and the mosaic really does
ship `model.language_model.layers.0.mlp.down_proj.mul1`. The loader was not misaddressing an expert, it
was meeting a quantized dense layer this runtime does not implement — which is blocker 2 above, not a
remap.

Consequence: **the mosaic still has no MTP here**, which is why its decode is ~10.8 tok/s against this
kit's 18.7–19.2. Details and the full argument:
[mosaic repo → `MTP-CODEBOOK-BLOCKER.md`](https://github.com/0xSero/glm-5.3-flash-spark-mosaic/blob/main/mosaic-gatea/MTP-CODEBOOK-BLOCKER.md).

## What the mosaic is

A mixed-precision artifact: 12 of the 45 MoE layers upgraded to 3.05 bpw (layers 3, 32, 33, 36–44 — all
288 experts of each), everything else including MTP layer 45 carried over byte-for-byte from the 2.05 bpw
base. 96.1 GB, codebook `mul1`.

| | |
|---|---|
| weights | [`0xSero/GLM-5.3-Flash-EXL3-Spark`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-Spark) @ `2642851741fc833764e77d03039117be559dc83e` |
| recipe repo | https://github.com/0xSero/glm-5.3-flash-spark-mosaic |
| runtime | SGLang `exl3-plain`, image `ghcr.io/0xsero/glm53-flash-exl3-plain:2p05-sglang-mul1-r1` (id `c65c840f1908…`) |
| launch | `./start-mosaic.sh` in this repo |

## Measured (receipted)

Quality — full G4 panel, 65,504 positions, 32 rows, both sides measured the same day on the same harness;
the mosaic wins 32/32 rows on top-1, KL and NLL:

| metric | base 2.05 bpw | M288-12L |
|---|---:|---:|
| top-1 agreement | 0.78902 | **0.80842** |
| KL lower-bound mean | 0.38378 | **0.32522** |
| candidate PPL | 4.28692 | **4.03563** |
| quick MMLU | 0.8342 ± 0.0105 | **0.8360 ± 0.0104** |
| GPQA diamond MC | 0.4343 ± 0.0353 | **0.4394 ± 0.0354** |

Serving: 262,144 context, vision on, fp8 KV, `max-running-requests 1`, `mem-fraction-static 0.95`
(KV 595,200), no CUDA graphs → **9–10 tok/s decode**. The 10-layer sibling reaches 11.2–11.5 with decode
graphs.

## Choosing between the two

| | this kit (K2 `mcg` + MTP) | mosaic (`mul1`) |
|---|---|---|
| decode @262k ctx | **18.7–19.2 tok/s** | 9–10 tok/s |
| quality vs base 2.05bpw | below the base on the panel | **+1.94 pp top-1, −15.3 % KL, 32/32 rows** |
| MTP | yes, 97.7–100 % acceptance | no |

Use this kit when throughput matters. Use the mosaic when fidelity matters, and pay ~2× in decode. The
open work that would collapse the trade-off is serving the mosaic on the MTP runtime. Its first step is
done and receipted ([`overlays/exl3-mul1/`](../overlays/exl3-mul1/): the `mul1` decode path and the
mixed per-layer rate); what is left is an image rebuild for the fused MoE kernel's mul1 instances and an
EXL3 `LinearMethod` for the artifact's quantized attention, dense MLP, shared-expert, vision and
`lm_head` tensors. Re-quantizing the mosaic to `mcg` remains the alternative.