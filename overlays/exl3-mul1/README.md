# `exl3-mul1` — the MTP runtime's EXL3 overlay, taught the mosaic's codebook

This directory holds the drop-in replacement for the MTP-capable image's
`vllm/model_executor/layers/quantization/exl3.py`, plus the probes that were
used to measure what it fixes and what it does not.

Mount it over the container's copy (and assert the sentinel in the file you
actually mounted, not the image id — a dropped bind-mount leaves the image
untouched and would otherwise pass):

```bash
docker run ... -v "$PWD/overlays/exl3-mul1/exl3.py":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro ...
docker exec <container> grep -c 'SUPPORTED_CODEBOOKS = ("mcg", "mul1")' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py   # must be 1
```

## What it changes

| # | Change | Why the stock overlay needed it |
|---|---|---|
| 1 | `codebook` may be `mcg` **or** `mul1` | the stock overlay raised `ValueError: this overlay only implements codebook=mcg` at config validation, before reading a weight |
| 2 | the codebook selects the decode math | `make_linear_exl3` passes `mul1=` (not `mcg=`) to `LinearEXL3`, and `apply_exl3_fused_moe` passes the real `(gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1)` flags instead of six hardcoded `True, False` |
| 3 | marker parameters are named by codebook | the checkpoint ships `…gate_proj.mul1`; the layer must register `w13_mul1` / `w2_mul1` or vLLM's generic expert mapping cannot resolve the tensor, which is what produced the `KeyError: 'layers.0.mlp.down_proj.mul1'` chain |
| 4 | marker validation is codebook-aware | `0x83DCD12D` for mul1, `0xCBAC1FED` for mcg; a checkpoint whose marker disagrees with its declared codebook now fails closed with that sentence |
| 5 | **per-layer** trellis width | a mosaic is mixed precision. `bits` in the artifact is the *average* (2.05); allocating `bits * 16` words for every layer under-allocates the upgraded ones. `Exl3Config` now distils a per-layer rate from the artifact's `tensor_storage` ledger and each `Exl3MoEMethod` sizes its own layer |

Everything else — rank-stacked TP1, the fat-expert fallback, the fused temp
cache, the pointer tables — is unchanged, and the `mcg` path is byte-for-byte
the same code it was.

## Measured on one DGX Spark (GB10, driver 580.173.02)

Receipt: [`../../docs/receipts/mul1-overlay-20260919T085502Z.txt`](../../docs/receipts/mul1-overlay-20260919T085502Z.txt).
Artifact: `0xSero/GLM-5.3-Flash-EXL3-Spark` @ `2642851741fc833764e77d03039117be559dc83e`.

| Probe | Result |
|---|---|
| `probe_mosaic_coverage.py` | config accepted (`codebook=mul1`, declared 2.05 bpw); ledger yields 42 expert layers, 30 at 2 bpw and **12 at 3 bpw — exactly layers 3, 32, 33, 36–44**, which is the mosaic's published composition |
| `probe_mul1_linear.py` | `LinearEXL3` decodes both rates: layer 4 `trellis (256,128,32) K=2`, layer 3 `trellis (256,128,48) K=3`, finite output, and `mcg` on the same bytes differs by 1.6× — the codebook is math, not a label |
| `probe_mul1_fused_moe.py` | the reference loop runs; the **fused kernel refuses**: `RuntimeError: MoE kernel: Only mcg codebook is currently supported` |

## Bring-up on the runtime (2026-09-19)

With `exl3.py` **and** `exl3_dequant.py` mounted, the kit's MTP recipe now gets the mosaic through
model construction and into checkpoint loading: four distinct failures were traced and fixed
(codebook gate, no loader for non-expert quantized tensors, the pre-fused KDA `qkv_proj`, and the
average-vs-per-layer rate), after which the engine logged
`EXL3: per-layer expert rate from quantization_config.json: {2: 30, 3: 12}` and loaded 7 of 12
shards before the rented box was stopped for an empty account. It has **not** yet answered a
request. Full log of each failure and what it actually was:
[`../../docs/receipts/mosaic-mtp-bringup-20260919T120326Z.md`](../../docs/receipts/mosaic-mtp-bringup-20260919T120326Z.md).
`serve-mosaic-mtp.sh` is the exact launch used (paths at the top).

## What still blocks MTP on the mosaic

The codebook gate was the first blocker, not the last. Two remain, and neither
is in this file:

1. **The fused MoE kernel in the shipped image has no mul1 instances.**
   exllamav3 0.0.43 (`c5d9c657`) raises at `exl3_moe.cu`; current upstream
   selects it (`const int cb_idx = gate_mul1 ? 1 : 0;`) and in fact instantiates
   its 32/64-row tiles for mul1 *only*. Fix = rebuild the image's
   `exllamav3_ext` against a version that has it. Until then routed experts can
   only run the Python `LinearEXL3` loop.

2. **The mosaic quantizes far more than the routed experts.** Counted from its
   own index: 37,152 routed-expert trellis tensors, but also 128 attention
   projections, 129 shared experts, 9 dense-MLP tensors (layers 0–2), 172
   vision-tower tensors, `layers.45.eh_proj` and `lm_head`. This overlay — like
   the stock one — returns `UnquantizedLinearMethod` for every `LinearBase`, so
   those tensors have no loader at all. Serving the mosaic on this runtime needs
   an EXL3 `LinearMethod` for plain linears, including splitting vLLM's merged
   `gate_up_proj` back into the two trellis sets the checkpoint stores
   separately. That is the bulk of the remaining work.

The earlier diagnosis in `docs/MOSAIC.md` — that `KeyError:
'layers.0.mlp.down_proj.mul1'` was a *naming-lineage* mismatch between
`experts.E.*` and `layers.N.mlp.down_proj.*` — was wrong. `layers.0.mlp` is a
**dense** MLP (`first_k_dense_replace = 3`), it has no experts, and the
checkpoint really does carry `model.language_model.layers.0.mlp.down_proj.mul1`.
The loader was not misaddressing an expert; it was meeting a quantized dense
layer the overlay does not implement.
