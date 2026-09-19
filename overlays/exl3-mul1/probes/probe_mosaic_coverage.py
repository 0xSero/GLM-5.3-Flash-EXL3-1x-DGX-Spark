#!/usr/bin/env python3
"""What the overlay accepts from the mosaic, and what it still cannot serve.

    MOSAIC_DIR=/model python3 probe_mosaic_coverage.py

Reads the artifact's own `quantization_config.json` and safetensors index:
  * the patched `Exl3Config` accepts codebook `mul1` and recovers the
    per-layer expert rate from the 46 MiB `tensor_storage` ledger;
  * every EXL3-quantized tensor is bucketed by scope, which is how you see
    that routed experts are a large majority of the tensors but not the whole
    artifact — attention, shared experts, the dense MLPs, the vision tower and
    `lm_head` are quantized too, and the overlay hands those to
    `UnquantizedLinearMethod`.
"""

import collections
import importlib.metadata as md
import json
import os

import exl3 as OV

SRC = os.environ.get("MOSAIC_DIR", "/model")

try:
    installed = md.version("exllamav3")
except Exception:  # noqa: BLE001 - diagnostics only
    installed = "?"
print("exllamav3 in image:", installed, "overlay pin:", OV.EXLLAMAV3_VERSION, OV.EXLLAMAV3_COMMIT[:12])

config = OV.Exl3Config.from_config(json.load(open(f"{SRC}/quantization_config.json")))
rates = config.layer_bits
print(
    "config accepted: codebook=%s declared_bits=%s fallback_bits=%s"
    % (config.codebook, config.declared_bits, config.bits)
)
print("per-layer expert rate: %d layers, %s" % (len(rates), collections.Counter(rates.values())))
print("upgraded (3 bpw) layers:", sorted(k for k, v in rates.items() if v == 3))

index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
groups: collections.Counter = collections.Counter()
for key in (k for k in index if k.endswith(".trellis")):
    if ".mlp.experts." in key:
        groups["routed experts (overlay implements)"] += 1
    elif key.startswith("model.visual."):
        groups["vision tower"] += 1
    elif ".self_attn." in key:
        groups["attention projections"] += 1
    elif ".mlp.shared_experts." in key:
        groups["shared experts"] += 1
    elif ".mlp." in key:
        groups["dense MLP (layers 0-2)"] += 1
    else:
        groups[key] += 1
print("quantized (trellis) tensors by scope:")
for scope, count in groups.most_common():
    print("   %6d  %s" % (count, scope))
