# The serving recipe, flag by flag

Every flag in `start.sh` exists for a measured reason. Provenance: the
program's D2/C1 isolated-benchmark admission
(`d2-c1-attempt2/{launch.sh,plan.json,admission.json}`), the D1→D2 depth sweep
(D1 14.98 tok/s → D2 19.11/19.03 tok/s, +28 %), and the full-window ladder in
[PERFORMANCE.md](PERFORMANCE.md). Nothing here is set by taste.

## Model loading

| Flag / env | Why |
|---|---|
| `--quantization exl3 --load-format safetensors --dtype bfloat16` | EXL3 K2 checkpoint; bf16 activations |
| `SAFETENSORS_LOAD_DEVICE=cuda:0` | stream shards straight to device |
| `SAFETENSORS_DROP_PAGE_CACHE=1` | drop page cache after load; the 102 GiB model would otherwise evict everything else in UMA |
| `GLM53_MTP_EXPERT_FP8=1` | FP8 expert compute in the MTP draft (nvfp4 lm-head paths explicitly off: `VLLM_MXFP8_LM_HEAD=0`, `VLLM_MTP_NVFP4_LM_HEAD=0`) |
| `VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_V2_MODEL_RUNNER=1` | AOT compile + V2 runner, part of the pinned fork's validated path on SM121 |
| `VLLM_EXL3_TRELLIS_MIN_M=1 / MAX_M=128` | trellis dequant tile bounds: decode steps run M=1; prefill tiles up to 128 (E1; 32 was the decode-tuned default) |
| `VLLM_EXL3_PREFILL_TRELLIS=1`, `EXL3_FUSED_MOE=1` | trellis kernels in prefill; fused MoE path |

## Attention / KV

| Flag | Why |
|---|---|
| `--attention-backend B12X` + `kda_prefill_backend=b12x` | SM121 B12X sparse attention; the only validated fast path for this checkpoint family on GB10 |
| `--kv-cache-dtype fp8_ds_mla` | packed MLA KV (656 B/token record); doubles usable context |
| `--block-size 256` | large pages for the sparse-MLA indexer |
| `--no-enable-prefix-caching` | the benchmark recipe runs cold-cache per request; also avoids UMA pressure surprises at 262k |

## Speculation + graphs (never disabled)

| Flag | Why |
|---|---|
| `--speculative-config {"method":"mtp","model":"/mtp","num_speculative_tokens":2,"attention_backend":"B12X"}` | the model's own MTP layer (45), all 288 experts, unquantized, served from the derived view; depth 2 measured +28 % over depth 1 |
| `cudagraph_mode FULL_DECODE_ONLY`, sizes `[1,3]` | one target token + two draft tokens per decode step; decode-only graphs keep prefill memory bounded. Graph capture receipts exist for target and draft |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` | profiler reserves graph memory so 0.93 util doesn't OOM at capture |

## Serving shape

| Flag | Why |
|---|---|
| `--max-model-len 262144` | the full GLM-5.3-Flash window, validated end to end |
| `--max-num-seqs 1` | single-stream recipe; KV admits 2.77× full-length requests anyway |
| `--max-num-batched-tokens 2048` | chunked-prefill chunk size (upstream 2× kit uses 7168 with a custom fat-MoE prefill kernel; that kernel is not in this image) |
| `--reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice` | GLM 4.7 thinking + tool-call wire format |
| `--generation-config vllm` | generation defaults from the checkpoint's generation_config |

## The MTP draft is not a second model

`tools/prepare_mtp_draft.py` filters the checkpoint index to the 891 keys that
constitute the native MTP block (layer 45 + `lm_head.weight` + embeddings),
asserts dtypes (BF16/F32 only), expert count (864 = 288×3), and the gate shape
([288, 4096]), then symlinks shards and writes a filtered index. Byte-identity
to the main checkpoint is structural: there is nothing to drift.

## Codebook lineage (why these weights)

This kit serves the `mcg`-codebook K2 2.0 bpw line. The `mul1`-codebook EXL3
line (turboderp/a0 and its mosaics) is a different lineage that this runtime
rejects by design (`this overlay only implements codebook=mcg`). The public
weights repo `0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw` is the provenance-verified
twin of the measured staging artifact: `source_manifest_sha256 81785e4348…`
matches byte-for-byte.
