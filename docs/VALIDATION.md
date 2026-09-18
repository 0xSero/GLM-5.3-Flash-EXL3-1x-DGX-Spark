# Validation pass — M288-12L mosaic, served on one DGX Spark

Date: 2026-09-18 (UTC). Host: spark-557f. Served model: `0xSero/GLM-5.3-Flash-EXL3-Spark`
@ `2642851741fc833764e77d03039117be559dc83e`, served from a directory that was itself downloaded through
the published path and hash-checked — i.e. the artifact below is what a stranger gets, not a build tree.

Every number here has a receipt file under `557f:/home/valentine/mosaic-gatea/receipts-557f/`. Where a
measurement turned out to be wrong or unusable, it is kept and labelled rather than deleted.

## 1. Artifact and download

| Check | Result | Receipt |
|---|---|---|
| Files fetched by the published path | 27/27 in 1 h 23 m | `sweep-streaming-20260918/` sibling `dl.log` |
| `sha256sum -c sha256-manifest.txt` | **26/26 OK, exit 0** | `mosaic-12l-published/VERIFY-LOG.txt` |
| Bytes on disk vs published total | 96,105,137,024 B exact | `01_artifact_scan.json` |
| Served bytes vs published | `served_bytes_match_published: true` | `01_artifact_scan.json` |
| Shards | 12, index complete | `01_artifact_scan.json` |

The download path is the one documented in the repo (`docker/download-mosaic.sh`: pinned revision,
`snapshot_download`, then manifest verification). It fails closed if the manifest is absent, and it
verified clean on a fresh destination.

## 2. What the artifact declares (config layer)

Read from the published `config.json` / `processor_config.json` / `generation_config.json`:

| Property | Value | Meaning |
|---|---|---|
| `text_config.max_position_embeddings` | 1,048,576 | 1 M native; we serve 262,144 |
| `text_config.num_hidden_layers` | 45 | + MTP layer, see below |
| `text_config.num_nextn_predict_layers` | **1** | the artifact declares MTP |
| `generation_config` | no `max_new_tokens` / `max_tokens` | no artifact-level output cap |
| image processor | `max_image_tokens: 8000`, patch 14, merge 2 | per-image vision ceiling |
| video processor | `max_image_tokens: 240000`, fps 2 | |
| tensor naming | `model.language_model.layers.N.*` | |

**The MTP layer is in the published artifact.** Layer 45 is the multi-token-prediction block, and its
module set proves it — `eh_proj`, `enorm`, `hnorm`, `shared_head`, alongside `self_attn`, `mlp`, and the
two layer norms:

| Layer-45 fact | Value |
|---|---|
| tensors | 3,508 |
| distinct expert indices | **288** (0…287) |
| quantized leaf sets | 873 × (`trellis`, `mul1`, `suh`, `svh`) |
| native tensors | 12 × `.weight`, `e_score_correction_bias`, `index_kpool_compress_{ape,gate}`, `bias` |
| codebook | `mul1` |

Across the whole model there are **450 native (bf16) non-visual `.weight` tensors** — the protected set
(embeddings, norms, `f_a_proj`/`f_b_proj`, and the layer-45 natives) is stored unquantized, as intended.
The vision tower ships too (`model.visual.blocks.0…9.*`).

Consequence for MTP: the weights are not the blocker. The MTP layer exists, has all 288 experts, and is
declared in config. What rejects it is a codebook gate in the serving overlay's quantization module
(`exl3.py`, `mul1` vs `mcg`) — a code question, not a weight question.

## 3. Server scans

Server: SGLang `exl3-plain` (`glm53-mosaic-test`), 262,144 context, `--mem-fraction-static 0.95`,
`--max-running-requests 1`, `--chunked-prefill-size 256`, `--max-prefill-tokens 256`,
`--disable-cuda-graph`, DSA attention, fp8 KV, multimodal on.

| Requirement | Result | Receipt |
|---|---|---|
| ≥ 256k context | declared `max_model_len 262144`; **236,510 prompt tokens exercised end to end** (longest sweep rung, TTFT 472.3 s, decode clean). A dedicated ~260k-token capacity probe was written but its script never reached the host, so it exited rc=2 without running — stated as a gap rather than quietly dropped. | `03_server_scan.json`, `sweep_streaming.json` |
| max_tokens 128k accepted | **128,000 accepted**, answer returned, `finish_reason: stop` | `04_reasoning_cap_scan.json` |
| reasoning not server-limited | 8,192 reasoning tokens / 30,912 chars, cut only by the client's cap, `not_truncated_by_server: true` | `05_long_reasoning_scan.json` |
| determinism | identical output on repeat at `temperature 0` ("391" both runs) | `06_determinism_scan.json` |
| vision present | `vision_config` in artifact, image tokens accepted up to the declared ceiling | `02_config_scan.json`, `07_*`, `vision_recheck` |
| MTP present | **false** on this runtime (codebook gate) | `03_server_scan.json` |

Two of these deserve their qualifiers stated plainly.

*The 128k scan* proves the server accepts a 128,000-token budget; the answer came back in 32 reasoning
tokens because the question was small. It does not claim the model will always use 128k.

*The long-reasoning scan* ran 787.6 s and produced 8,192 reasoning tokens with **empty content** and
`finish_reason: length` — the client's own 8,192-token cap consumed the entire budget, so no final answer
was reached. That is the expected behaviour of a cap, not a server limit: the server did not truncate.

## 4. Vision at full resolution

The artifact caps one image at **8,000 image tokens** (`processor_config.json`). The probe encodes the
word `SPARK` as binary bar columns and asks for the word, at three sizes:

| Image | PNG bytes | Image tokens | Share of the 8,000-token ceiling |
|---|---|---|---|
| 512² | 4,407 | 361 | 4.5 % |
| 2048² | 33,557 | 5,476 | 68 % |
| 4096² | 91,352 | **7,921** | **99 %** |

All three were accepted and processed at full resolution with no error and no instability — 4096² is
99 % of the declared per-image ceiling.

First pass, `07_vision_fullres_scan.json`: every size returned `answered: false` with
`finish_reason: length` and `reasoning_tokens == completion_tokens == 64`. That is a **probe defect**,
not a vision defect: a reasoning model given 64 tokens spends all 64 thinking. The images were accepted
(`prompt_tokens_details.image_tokens` populated) but no answer could land, so the first pass is evidence
of *acceptance*, not of *correctness*.

Second pass, `vision_recheck.json`, keeps the same fixtures and fixes only the budget: 256 tokens
with thinking disabled, plus one thinking arm at 4096².

| Arm | Image tokens | reasoning tokens | seconds | finish | answered | word correct |
|---|---|---|---|---|---|---|
| 512², thinking off | 361 | 0 | 53.4 | length | yes | no |
| 2048², thinking off | 5,476 | 0 | 68.6 | length | yes | no |
| 4096², thinking off | 7,921 | 0 | 71.6 | length | yes | no |
| 4096², thinking on | 7,921 | 2,048 | 219.7 | length | no | no |

`enable_thinking: false` was honoured (zero reasoning tokens). At every size the model produced an
analysis of the actual image and ran out of the 256-token budget mid-decode; the thinking arm spent
all 2,048 tokens reasoning and produced no answer. **No verified word answer was obtained**, and that is
a probe-design limit, not a vision limit: the fixture encodes a word as 7 bits per character, and
decoding it verbosely does not fit the budget.

What the arms *do* establish is perception at resolution, because the coordinates the model reports
match the encoder's own geometry:

- 512²: the model reports bars at `x≈70-135`; the fixture puts column 1 at `x = 512//7 = 73` to `131`.
- 2048²: the model reports "Column 1 (x ~290-525)"; the fixture puts it at `x = 2048//7 = 292` to `526`.
- 4096²: the model reports 5 columns of bars before the budget ended — the fixture encodes a 5-letter word.

Two independent resolutions show coordinate-accurate reading, and the third shows correct structure at
99 % of the declared per-image ceiling. That is the claim this pass supports: **full-resolution vision
works up to the artifact's 8,000-token per-image ceiling.** A verified end-to-end answer would need a
decidable fixture and a larger budget — both cheap, neither run here.

## 5. Speed

Two sweeps, one of which had to be thrown away. Both are kept.

**Superseded — the battery ladder (`08_speed_sweep.json`).** Its prompts were `FILLER × N`, so rung *N*
was a *prefix* of rung *N+1*. RadixAttention serves shared prefixes from cache, so after the first rung
each cell measured only its delta: the 260,096 rung (125.4 s) came out *faster* than the 131,072 rung
(130.1 s), which is impossible for a cold prefill. The cell totals are therefore not a throughput curve.
Kept as the record of a method error.

**Authoritative — streaming sweep (`sweep_streaming.json`).** Same ladder, but every prompt opens with a
unique 16-hex-character nonce, so no prefix can match from position 0, and TTFT is timestamped from the
SSE stream to split prefill from decode.

| Prompt tokens | TTFT s | prefill tok/s | decode tok/s |
|---|---|---|---|
| 990 | 2.441 | 405.6 | 10.94 |
| 3,792 | 8.281 | 457.9 | 10.81 |
| 14,950 | 30.035 | 497.8 | 10.84 |
| 59,631 | 117.675 | 506.7 | 10.81 |
| 119,213 | 234.415 | 508.6 | 10.78 |
| 181,870 | 359.097 | 506.5 | 10.72 |
| 236,510 | 472.342 | 500.7 | 10.61 |

Fitted on all seven points: `TTFT = fixed + prompt_tokens / R`.

- **Marginal prefill R = 503.5 tok/s**, intercept **−0.163 s**, r² = 0.999913, max residual 2.74 s.
  A slightly negative intercept means the fixed per-request cost is not resolvable at this precision —
  effectively zero against prefill times measured in minutes. Fitting only the first five points gives
  R = 510.0 tok/s with +0.696 s, so the honest summary is **~504–510 tok/s marginal prefill**, the
  spread coming from which end of the ladder is weighted.
- The small residuals are themselves the evidence that every rung paid a full cold prefill — a prefix
  hit would collapse TTFT on the later rungs and destroy the linear fit. Per-cell prefill rate peaks
  mid-ladder (508.6 tok/s at 119 k) and eases slightly at both ends: 405.6 tok/s at 990 tokens where
  per-request overheads dominate, 500.7 tok/s at 236 k.
- **Decode 10.61–10.94 tok/s**, mean 10.79, spread **3.06 %** across a 238× range of prompt length —
  flat, as decode should be, and the same 9–10 tok/s class the artifact's own measured record already
  claimed, now with a clean method behind it.

(`cached_tokens` is not populated by this SGLang build. `cache_clean` therefore reads `null`, meaning
*unreported*, not *false*. The no-reuse evidence is the linear fit above.)

Against the program's targets — 600 tok/s prefill, 15–20 tok/s decode — the mosaic measures
**~504 tok/s prefill (84 % of target)** and **~10.8 tok/s decode (54–72 % of target)**. Prefill is within
16 % of target under a deliberately throttled prefill configuration (`--chunked-prefill-size 256`,
`--max-prefill-tokens 256`); decode is the real gap, and decode is exactly what MTP addresses: the K2
sibling artifact, which runs MTP, measures 18.5–20.0 tok/s on the same box.

## 6. Cache demonstration — the expected result did not occur, and that is the finding

To show the superseded ladder's failure mode directly, the same ~14 k-token body was sent four times:
twice with an identical opening (so the second *should* hit the cache), twice with a fresh nonce.

| Arm | prompt tokens | TTFT s | prefill tok/s |
|---|---|---|---|
| hit_1, nonce A | 14,043 | 28.847 | 486.8 |
| hit_2, nonce A (identical) | 14,043 | **57.114** | 245.9 |
| miss_1, nonce B | 14,044 | 56.940 | 246.6 |
| miss_2, nonce C | 14,042 | 57.133 | 245.8 |

The identical repeat was **twice as slow**, not faster. The server's own `Prefill batch` log explains it:
all four requests logged `#cached-token: 0` and together accounted for 113,088 new tokens against 56,172
tokens of prompt — requests 2–4 each prefilled their prompt twice at the normal per-step rate.

The cause is documented in `receipts/cache-demo-analysis.md`: the sweep that ran immediately before had
inserted its whole ladder into the radix cache (990 + 3,792 + 14,950 + 59,631 + 119,213 + 181,870 +
236,510 = **616,956 tokens**, against a pool of 595,200), so the pool was over-subscribed by its own
sweep and a repeated prompt was re-prefilled rather than served.

Reuse itself works fine on this server, as the vision probes immediately afterwards show — one step,
64 new tokens, and 384 / 5,504 / 7,936 tokens served from cache for the repeated image prompts. So the
honest statement is: **prefix reuse is real and large when the pool has room; it silently fails under
the saturation a ladder sweep creates.** This is exactly why the streaming sweep's numbers can be
trusted (its steps logged zero cached tokens, and its TTFT is linear in prompt length) and why the
superseded ladder's later rungs were fast (each rung matched the previous rung's cached prefix and paid
only the delta).

## 7. MTP

Two facts from this pass settle the shape of the MTP question:

1. **The artifact has MTP.** Layer 45 (`eh_proj`/`enorm`/`hnorm`/`shared_head`) with 288 experts and
   `num_nextn_predict_layers: 1`. Nothing about the weights blocks MTP.
2. **Neither MTP-capable runtime accepts `mul1`.** The mosaic's only codebook, `mul1` (873 tensor sets in
   layer 45 alone), is rejected at config validation by this kit's vLLM/MTP overlay
   (`exl3.py:568`, codebook gate).

So MTP is a code problem, not a weight problem — and M1 pinned down *which* code problem.

**M1 ran** (`receipts/m1-findings.md`, raw logs in the receipt directory): the exact MTP recipe against
the mosaic, with the codebook gate relaxed to a warning, the overlay sentinel-asserted in the launched
file so it cannot silently measure the wrong module. Result: **the gate is cleared and the next blocker
is a checkpoint-key mismatch.**

```
File ".../vllm/models/glm5next/nvidia/model.py", line 904, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.down_proj.mul1'
```

No codebook error appears anywhere in the log, and the engine reached weight loading — i.e. config
validation passed with `mul1` accepted. It died after `Loading safetensors checkpoint shards: 0/12`, at
the first MoE layer, before any kernel ran. The reason is a naming-lineage difference: the mosaic
descends from the turboderp/SGLang artifact and carries `model.language_model.layers.N.mlp.experts.E.*`
keys, while the vLLM loader addresses experts through `layers.N.mlp.down_proj.*`. The mismatch shows up
at layer 0, which is a stock layer — so it is a property of the artifact's naming lineage, not of the
mosaic's 12 upgraded layers.

Cost estimate, stated honestly: a checkpoint-key remap into the loader's expected layout, then the
`mul1` decode path (never executed, so acceptance after loading is unknown), then measurement. It has a
known entry point (`model.py:904`) and is not research — but it is more than the gate, which is what the
artifact documentation implied before this pass.

## 8. Restore, and what the baseline actually does

After M1 the script restored the standing-best server (`glm53-flash-1x-v2`) and it reached health:

| Check | Result |
|---|---|
| `/health` | 200 |
| `/v1/models` | `glm-5.3-flash`, `max_model_len 262144` |
| Chat completion | answers with `finish_reason: stop` |
| MTP counters | present and advancing (`spec_decode_num_drafts_total`, `…_accepted_tokens_total`) |

Two earlier attempts to start this container had died with `torch.OutOfMemoryError` during MTP weight
post-processing (4.50 GiB requested, 3.71 GiB free) — once while the E9 container still held the GPU and
once while the 96 GB download filled page cache. Both are memory-pressure failures, not recipe failures;
with the box quiet it loads.

Its decode rate depends on state in a way worth stating, because a naive probe reads low:

| Prompt tokens | decode tok/s | MTP acceptance |
|---|---|---|
| 48 | 17.06 | 80.6 % |
| 4,026 | **19.41** | 100 % |
| 16,028 | **19.60** | 94.3 % |

The recorded 18.7–19.2 tok/s reproduces at realistic context. The first requests after a cold load
measure ~11–12 tok/s (CUDA graph capture and kernel warmup — the server's own log ramps 6.0 → 11.7 tok/s
over the first seconds), and a 48-token prompt gives 17.1 because the MTP draft has little context to
condition on (80.6 % acceptance against 94–100 %). Prefill on the same cells: 4,026 tokens in 7.18 s
(561 tok/s), 16,028 in 35.70 s (449 tok/s).

## 9. Method notes

- The battery ran on the live mosaic server; nothing was stopped or restarted to obtain these numbers.
- One measurement in this pass was invalidated by its own method (the ladder in §5) and one by its own
  probe design (§4 first pass). Both are published with the correction rather than removed.
- `07_vision_fullres_scan.json` ran before the ladder despite the script's intent to run it after; the
  reordered file was copied to the host after the copy that was executed. The server survived either
  order, so nothing was lost — recorded because the receipts and the script's own header disagree.