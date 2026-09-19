# Mosaic on the MTP runtime — bring-up log (2026-09-19)

Host: rented DGX Spark (vast.ai GB10, driver 580.173.02, 121 GB, aarch64). The runtime image's
rootfs was unpacked (OCI whiteouts applied) and its python/torch/`exllamav3_ext` run directly,
because that container has no docker daemon. Model: the verified 96.1 GB download of
`0xSero/GLM-5.3-Flash-EXL3-Spark @ 2642851741fc`. Draft view: `tools/prepare_mtp_draft_mosaic.py`
(3,513 keys, layer 45 + embeddings, 288 experts, quantized).

Launch = the kit's own MTP recipe (B12X, `fp8_ds_mla`, block 256, `FULL_DECODE_ONLY` graphs,
`--speculative-config method=mtp num_speculative_tokens=2`), `--max-model-len 32768` for bring-up,
`EXL3_FUSED_MOE=0` (the image's exllamav3 0.0.43 fused kernel has no mul1 instances).

## Each failure, and what it actually was

| # | Error | Cause | Fix |
|---|---|---|---|
| 1 | `ValueError: this overlay only implements codebook=mcg; got 'mul1'` | config gate | overlay accepts mul1; codebook selects the decode math and the marker parameter name |
| 2 | `KeyError: 'layers.0.mlp.down_proj.mul1'` | layer 0 is a **dense** MLP (`first_k_dense_replace=3`) and the mosaic quantizes it; the runtime has no `LinearMethod` for non-expert linears | `exl3_dequant.py` reconstructs every non-expert EXL3 group into one `.weight` as the weight stream is read (exllamav3's own `reconstruct`, codebook-aware) |
| 3 | `KeyError: 'layers.0.self_attn.qkv_proj.weight'` | the mosaic pre-fuses KDA q/k/v into one tensor (SGLang lineage); this loader wants `q_proj`/`k_proj`/`v_proj`, which it then re-fuses into `in_proj_qkvbfg_a` | split the reconstructed matrix by rows (3 × 8,192 = 64 heads × 128) before yielding |
| 4 | `EXL3 load shape mismatch layers.3…w2_trellis: dest (128,256,32) != loaded (128,256,48)` | layer 3 is one of the twelve 3.05 bpw layers; `config.json`'s embedded quant block carries only the **average** rate (2.05) and omits the `tensor_storage` ledger | resolve the ledger from the model directory at `create_weights` time |

After #4: `EXL3: per-layer expert rate from quantization_config.json: {2: 30, 3: 12}` and the engine
loaded checkpoint shards normally — **7 of 12 shards at 06:28 into the load**, with no further key,
shape or codebook error.

## Where it stopped

The rented instance was stopped mid-load when the account's credit reached zero
(`vastai show user → credit 0`), not by a defect in the run. The remaining unknowns are therefore
still unknowns, and are stated as such:

- whether the load completes and the engine reaches CUDA-graph capture (weights ≈96 GB packed plus
  ≈5 GB of reconstructed non-expert BF16, on a 128 GB unified-memory box — the throughput artifact
  already sits at 102 GB, so this is expected to be tight and may need a lower
  `--gpu-memory-utilization` or a shorter context than 262,144);
- whether MTP accepts drafts against a quantized draft block;
- decode and prefill rates, and therefore whether MTP closes the 9–10 → 18.7–19.2 tok/s gap.

Nothing above should be read as "it serves". It loads; it has not yet answered a request.

## Known follow-up, independent of budget

`EXL3_FUSED_MOE=0` forces the Python `LinearEXL3` loop for routed experts because the image pins
exllamav3 0.0.43, whose `exl3_moe` raises `MoE kernel: Only mcg codebook is currently supported`.
Upstream selects the codebook (`cb_idx = gate_mul1 ? 1 : 0`) and instantiates its 32/64-row tiles
for mul1 only, so rebuilding the image's `exllamav3_ext` is both the fix and, plausibly, a prefill
gain.
