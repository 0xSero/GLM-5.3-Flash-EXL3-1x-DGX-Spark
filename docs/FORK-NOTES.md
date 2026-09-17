# Fork notes — from the 2× kit to 1× Spark

Upstream: [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(GLM-5.3-Flash EXL3 TR3 4 bpw, TP=2 over CX7, DFlash2 k=7 speculator,
NoPE-MLA overlay image, ~1,500 tok/s prefill / 36–63 tok/s decode at ×1).

This fork keeps the same product shape — clone, `./start.sh`, OpenAI API on a
fixed port — and changes the entire engine layer to fit and stay fast on
**one** Spark:

| | Upstream (2×) | This fork (1×) |
|---|---|---|
| Parallelism | TP=2, 2 nodes, mp executor | TP=1, single node |
| Weights | TR3 4 bpw (~164 GiB) | K2 2.0 bpw, `mcg` codebook (~102 GiB) |
| Runtime | NoPE-MLA overlay (FLASHINFER_MLA_SPARSE_SM120 geometry) | B12X attention backend + `kda_prefill_backend=b12x` |
| KV | `fp8_ds_mla` packed | `fp8_ds_mla` packed, block 256 |
| Speculator | DFlash2 k=7 (external draft model) | **native MTP layer 45**, depth 2, unquantized 288-expert MoE |
| Context | 850k–1M default | 262,144 (full model window), validated end to end |
| Prefill kernel | `EXL3_FAT_GROUPED` custom CUDA overlay | stock EXL3 trellis + fused MoE (port tracked as future work) |
| Decode (×1 stream) | 36–63 tok/s (greedy, thinking off) | ~19 tok/s (sampled temp 1.0, thinking on) |
| Prefill | ~1,500 tok/s | ~450 tok/s |

What was deliberately kept: one-command UX, public image + public weights,
`.env`-overridable knobs, lifecycle subcommands, and the refusal to launch
next to another GPU job.

Why decode/prefill differ so much: upstream's numbers are greedy-sampled,
thinking-off, structured prompts on 4 bpw weights with two GPUs and a custom
prefill kernel; ours are sampled non-greedy with thinking on, which is a
harder operating point — and still meets the 15–20 tok/s single-Spark decode
target with 100 % CUDA-graph coverage of decode.
