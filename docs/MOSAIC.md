# The M288-12L mosaic in this kit

This fork serves the **EXL3 K2 2.0 bpw `mcg`** staging quant with native MTP on vLLM/B12x. The
**M288-12L mosaic** is a different artifact from the same model family, and **this kit's runtime cannot
serve it**. This document exists so nobody spends an afternoon rediscovering that, and so the mosaic is
one command away — on its own runtime.

## Why this kit can't serve the mosaic

The mosaic is codebook **`mul1`** (it is built on the turboderp EXL3 line); this kit's overlay only
implements **`mcg`**. The rejection happens at config validation, before a single weight is read:

```
ValueError: this overlay only implements codebook=mcg; got 'mul1'
```

Gate: `exl3.py:568` in both MTP-capable images (`glm53-reap-native-mtp:…-r4-expert-fp8` and
`glm53-b12x-exl3:jovian-3aada677-r3`). The codebook is not a label — it selects the decode math (the
trellis codebook tables consumed by `LinearEXL3`), so this is not a flag flip.

**That gate is the first blocker, not the only one.** The M1 experiment (2026-09-18) relaxed exactly that
gate and launched this kit's MTP recipe against the mosaic. Validation then passed and the engine
proceeded to weight loading, where it died at the first MoE layer:

```
File ".../vllm/models/glm5next/nvidia/model.py", line 904, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.down_proj.mul1'
```

The mosaic descends from the turboderp/SGLang artifact and its keys are
`model.language_model.layers.N.mlp.experts.E.*`; this kit's vLLM loader addresses experts as
`layers.N.mlp.down_proj.*`. Layer 0 is a stock (non-substituted) layer, so the mismatch is a property of
the artifact's naming lineage rather than of the mosaic's 12 upgraded layers. Bringing MTP to the mosaic
therefore needs **a checkpoint-key remap first, then the `mul1` decode path** — code work with a known
entry point (`model.py:904`), not a research problem, but more than the gate. Receipts:
[`VALIDATION.md`](VALIDATION.md) §7 and
[`receipts/m1-findings.md`](receipts/m1-findings.md).

Consequence: **the mosaic has no MTP here**, which is exactly why its decode is ~10.8 tok/s against this
kit's 18.7–19.2. Details and the full argument:
[mosaic repo → `MTP-CODEBOOK-BLOCKER.md`](https://github.com/0xSero/glm-5.3-flash-spark-mosaic/blob/main/mosaic-gatea/MTP-CODEBOOK-BLOCKER.md).

## What the mosaic is

A mixed-precision artifact: 12 of the 45 MoE layers upgraded to 3.05 bpw (layers 3, 32, 33, 36–44 — all
288 experts of each), everything else including MTP layer 45 carried over byte-for-byte from the 2.05 bpw
base. 96.1 GB, codebook `mul1`.

| | |
|---|---|
| weights | [`0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L`](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L) @ `2642851741fc833764e77d03039117be559dc83e` |
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
open work that would collapse the trade-off is implementing the `mul1` decode path in the MTP overlay, or
re-quantizing the mosaic to `mcg` — both multi-day, neither started.