# GLM-5.3 Flash EXL3 on 1× DGX Spark — native MTP serving

One NVIDIA DGX Spark (GB10, 128 GB unified memory) serving
**GLM-5.3-Flash** (320B-A18B MoE, 45 layers, 288 routed experts) at
**EXL3 K2 2.0 bpw** (`codebook mcg`) with the model's **native MTP layer
(layer 45, all 288 experts, unquantized)** as the speculative decoder,
**CUDA graphs on** for both target and draft, and the **full 262,144-token
context window** usable end to end. OpenAI-compatible API on port 8888.

Forked from [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
and rebuilt for a **single** Spark: different quant tier (K2 2.0 bpw vs
TR3 4 bpw), different runtime image (B12X sparse-attention vLLM vs NoPE-MLA
overlay), different speculator (native MTP depth 2 vs DFlash2 k=7). See
[docs/FORK-NOTES.md](docs/FORK-NOTES.md).

## Measured on one DGX Spark (receipted)

Single stream, sampling **on** (`temperature 1.0`, `top_p 0.95`, thinking on),
MTP depth 2, CUDA graphs `FULL_DECODE_ONLY`. Accounting is exact per-chunk
`token_ids` cross-checked against vLLM's native counters; windows are
matched-sustained (≥ 30 s). Full ladder and method:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

| Input tokens | Decode tok/s | Prefill tok/s | TTFT | Acceptance |
|---:|---:|---:|---:|---:|
| 1,023 | 19.19 | 391.5 | 2.61 s | 0.98 |
| 4,095 | 19.22 | 427.0 | 9.59 s | 0.98 |
| 16,383 | 18.95 | 442.9 | 36.99 s | 0.98 |
| 65,535 | 18.82 | 454.3 | 144.26 s | 0.99 |
| 131,071 | 19.00 | 454.2 | 288.58 s | 0.99 |
| 199,999 | 18.70 | 452.7 | 441.83 s | 0.99 |
| 260,095 | 18.89 | 451.8 | 575.65 s | 1.00 |

Receipt: [docs/receipts/mtp-ctx-sweep-full-20260917T160505Z.json](docs/receipts/mtp-ctx-sweep-full-20260917T160505Z.json)
(sha256 `a8023b2a9574…`, manifest in [docs/RECEIPTS.sha256](docs/RECEIPTS.sha256)).
Decode spread across the whole window is 2.7 %. KV pool: **727,449 tokens** at `fp8_ds_mla` (2.77× a full
262,144-token request). Weights on disk: ~102 GiB; engine load 102.29 GiB.

## Requirements

- NVIDIA DGX Spark (GB10, SM121/a, arm64), driver ≥ 535, NVIDIA Container Toolkit
- Docker with `--gpus all` working
- ~135 GB free disk (111 GB weights download + extraction headroom)
- 128 GB unified memory (the engine reserves ~102 GiB for weights + activations at `MEM_FRACTION=0.93`)
- No other GPU job may run on the box while serving (the launcher checks and refuses)

## Quick start

```bash
git clone https://github.com/0xSero/glm-5.3-flash-1x-dgx-spark.git
cd glm-5.3-flash-1x-dgx-spark
./start.sh
```

`./start.sh` does everything: pulls the public container image, downloads the
public weights from Hugging Face (`0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw`,
133 shards, ~111 GB), derives the native MTP draft view from the downloaded
checkpoint (CPU-only, no tensor copies — symlinks + a filtered index), launches
vLLM, and waits for `/health`. Re-runs skip completed steps.

First token takes a few minutes (weight load + CUDA graph capture for target
and draft). Then:

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Give me three facts about DGX Spark."}],
  "max_tokens": 200
}'
```

The served model id is **`glm-5.3-flash`**. Thinking mode is on by default
(GLM 4.7 reasoning parser); pass `"chat_template_kwargs": {"enable_thinking": false}`
to disable.

## Lifecycle

| Command | Purpose |
|---|---|
| `./start.sh` | start (download if needed, launch, wait healthy) |
| `./start.sh stop` | stop the serving container (preserved, not deleted) |
| `./start.sh restart` | stop + start |
| `./start.sh status` | container state + API health |
| `./start.sh logs` | follow engine logs |
| `./download.sh` | weights download only (same as `./start.sh download`) |

## Configuration (`.env`, optional)

Copy `.env.example` to `.env` to override. Defaults are the measured recipe.

| Key | Default | Notes |
|---|---|---|
| `IMAGE` | `ghcr.io/0xsero/glm53-b12x-exl3:b12x-mtp-1xspark-v1` | public arm64 runtime image |
| `WEIGHTS_REPO` | `0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw` | public HF repo; provenance-verified twin of the staging K2 artifact (`source_manifest_sha256 81785e43…`) |
| `PORT` | `8888` | API port |
| `CONTEXT_LEN` | `262144` | full GLM-5.3-Flash window |
| `MEM_FRACTION` | `0.93` | engine memory fraction |
| `SPEC_DEPTH` | `2` | native MTP draft depth (1 is the fallback) |
| `HF_TOKEN` | unset | optional; only needed to dodge HF rate limits |

## How it works

- **Artifact.** K2 2.0 bpw EXL3 (`codebook mcg`), 45 layers, 288 routed experts
  per layer, tail 224-expert layers retained; shipped as flat safetensors with an
  EXL3 manifest. The native MTP block (layer 45 + embeddings + lm_head, 889
  tensors BF16/F32 + 2 shared) stays **unquantized**: `tools/prepare_mtp_draft.py`
  filters the checkpoint index to those 891 keys and symlinks the shards — the
  draft is literally part of the main checkpoint, byte-identical, nothing regenerated.
- **Runtime.** vLLM (pinned fork `local-inference-lab/vllm@3aada677`) with the
  B12X attention backend and `kda_prefill_backend=b12x`, packed `fp8_ds_mla` KV
  cache, block size 256, chunked prefill, EXL3 trellis dequant tuned for
  decode-M1/prefill-M128, fused MoE.
- **Speculation.** `--speculative-config method=mtp, num_speculative_tokens=2,
  attention_backend=B12X`, draft served from the derived native view.
- **Graphs.** `cudagraph_mode FULL_DECODE_ONLY`, capture sizes `[1,3]` (1 target
  token + 2 draft tokens per step). Graphs are never disabled.
- Flag-by-flag rationale and provenance: [docs/RECIPE.md](docs/RECIPE.md).
- Image build (public sources, pinned): [image/](image/).

## Honest limits

- ~450 tok/s single-stream prefill: the upstream 2× recipe reaches ~1,500 tok/s
  with a custom grouped-MoE prefill kernel (`EXL3_FAT_GROUPED`), 4 bpw weights,
  and two GPUs. Porting that kernel to this 1× image is the known path to a big
  prefill gain and is tracked in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- Decode is tuned for single stream (`MAX_NUM_SEQS=1`); concurrent clients share it.
- The B12X path is a pinned fork, not upstream vLLM; the image pins exact digests.

## Credits & licenses

- Upstream recipe and inspiration: [MiaAI-Lab](https://github.com/MiaAI-Lab)'s
  2× DGX Spark EXL3 kit.
- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash);
  weights repo license applies (see the HF repo card).
- Code in this repository: MIT ([LICENSE](LICENSE)).
- Quantization lineage: EXL3/TurboDERP toolchain, `mcg` codebook.
