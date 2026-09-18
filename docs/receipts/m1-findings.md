# M1 — the MTP blocker, isolated

Date: 2026-09-18. Host: spark-557f. Receipt: `receipts-557f/m1-mul1-mtp-20260918T173913Z/`.

## What was run

The vLLM MTP recipe that this kit ships, pointed at the M288-12L mosaic, with exactly one change:
the codebook gate in `exl3.py` (which rejects `mul1` at config validation) relaxed to a warning — the
patched overlay sentinel-asserted in the launched file, so the run cannot silently measure the wrong
module the way E9 did.

## Outcome: FAIL, one layer deeper than the gate

The gate is cleared. Config validation completed and the engine moved on to weight loading, where the
GLM-5-Next model implementation raised:

```
File "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py", line 904, in load_weights
    param = params_dict[name]
KeyError: 'layers.0.mlp.down_proj.mul1'
```

No codebook error and no gate error appears anywhere in the log. The run died after
`Loading safetensors checkpoint shards: 0% | 0/12` — at the first MoE layer, before any quantized
kernel executed.

## Why

A naming-lineage mismatch. The mosaic descends from the turboderp/SGLang artifact, whose keys carry a
transformers-5.x style prefix and are enumerated per expert:

```
model.language_model.layers.0.mlp.experts.<e>.down_proj.mul1        <- mosaic (what exists)
layers.0.mlp.down_proj.mul1                                          <- vLLM loader (what it looks for)
```

The vLLM/MTP runtime's loader (`glm5next/nvidia/model.py`) addresses experts through a mapped,
non-prefixed key. It cannot find any tensor it is looking for, so this is not a quantization problem,
not an MTP-weight problem, and not the codebook gate — it is a **checkpoint-key mapping problem**,
reported at layer 0, which is a stock (non-substituted) layer. That detail matters: the mismatch is a
property of the artifact's naming lineage, not of the mosaic's 12 upgraded layers.

## What this means for "it needs MTP"

Evidence now stands in a chain, each link receipted:

1. **The artifact carries MTP.** Layer 45 is the MTP block (`eh_proj`, `enorm`, `hnorm`, `shared_head`)
   with all 288 experts, 873 `mul1` tensor sets and 12 native tensors; `num_nextn_predict_layers: 1`.
   The weights are not the obstacle.
2. **The codebook gate was a real blocker, and it is cleared.** With `mul1` accepted, the recipe
   proceeds past validation (previous gate error at `exl3.py:568` is gone from this log).
3. **The next blocker is key mapping.** `KeyError: 'layers.0.mlp.down_proj.mul1'` at
   `glm5next/nvidia/model.py:904`, before any kernel runs.
4. **After remapping there will likely be more.** The `mul1` decode path in the EXL3 overlay has never
   been executed, so acceptance is unknown even once the checkpoint loads. This run does not bound that.

So the honest cost estimate for MTP on the mosaic is: a checkpoint-key remap from the
`model.language_model.layers.N.mlp.experts.E.*` layout to the loader's expected layout, then the
`mul1` codebook decode path, then measurement. It is a code port with a known entry point
(`model.py:904`), not a research problem — but it is more than the gate, which is what the mosaic
documentation previously implied.

## Restore

`glm53-flash-1x-v2` was started by the M1 script at 17:41:26Z and reached `running=true`. Health is
recorded in `RUN.txt`. The mosaic's SGLang container was stopped (not removed) at 17:40:15Z with its
`docker inspect` preserved in the receipt directory, so the mosaic can be brought back with
`docker start glm53-mosaic-test`.