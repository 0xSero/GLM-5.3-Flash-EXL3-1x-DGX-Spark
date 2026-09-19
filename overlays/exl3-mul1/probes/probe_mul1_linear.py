#!/usr/bin/env python3
"""LinearEXL3 must decode the mosaic's `mul1` trellis, at both of its rates.

Run inside (or against) the MTP runtime image, with the overlay directory on
`PYTHONPATH`:

    MOSAIC_DIR=/model python3 probe_mul1_linear.py

Proves three things per layer:
  * the artifact's scalar marker is the mul1 multiplier 0x83DCD12D;
  * `make_linear_exl3(..., codebook="mul1")` builds a LinearEXL3 whose decode
    path is mul1 (not mcg) and whose K comes from the trellis itself;
  * decoding the same bytes as `mcg` gives a materially different result, so
    the codebook selects real math rather than a label.
"""

import json
import os

import torch
from safetensors import safe_open

import exl3 as OV

SRC = os.environ.get("MOSAIC_DIR", "/model")
DEV = os.environ.get("PROBE_DEVICE", "cuda:0")
# layer 4 is a stock 2.05 bpw layer, layer 3 one of the twelve 3.05 bpw upgrades
LAYERS = [int(x) for x in os.environ.get("PROBE_LAYERS", "4,3").split(",")]

index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]


def fetch(base: str) -> dict:
    out = {}
    for suffix in ("trellis", "suh", "svh", "mul1"):
        key = f"{base}.{suffix}"
        with safe_open(f"{SRC}/{index[key]}", "pt") as f:
            out[suffix] = f.get_tensor(key)
    return out


print("codebook support:", OV.SUPPORTED_CODEBOOKS, "mul1 marker", OV.MUL1_MARKER_SIGNED_INT32)
for layer in LAYERS:
    tensors = fetch(f"model.language_model.layers.{layer}.mlp.experts.0.gate_proj")
    marker = int(tensors["mul1"].reshape(-1)[0])
    assert marker == OV.MUL1_MARKER_SIGNED_INT32, f"layer {layer}: marker {marker}"
    print(
        f"layer {layer}: trellis {tuple(tensors['trellis'].shape)} "
        f"K={tensors['trellis'].shape[-1] // 16} marker={marker}"
    )

    linear = OV.make_linear_exl3(
        tensors["trellis"].to(DEV),
        tensors["suh"].to(DEV),
        tensors["svh"].to(DEV),
        tensors["mul1"].reshape(1).to(DEV),
        codebook="mul1",
    )
    assert linear.mul1 and not linear.mcg
    print("   LinearEXL3 mcg?", linear.mcg, "mul1?", linear.mul1, "K", linear.K)

    x = torch.randn(4, tensors["suh"].numel(), device=DEV, dtype=torch.half)
    y = linear.forward(x, {}, out_dtype=torch.float32)
    assert torch.isfinite(y).all()
    print("   forward", tuple(y.shape), "finite True", "std %.4f" % y.std().item())

    wrong = OV.make_linear_exl3(
        tensors["trellis"].to(DEV),
        tensors["suh"].to(DEV),
        tensors["svh"].to(DEV),
        torch.tensor([OV.MCG_MARKER_SIGNED_INT32], dtype=torch.int32, device=DEV),
        codebook="mcg",
    )
    y_mcg = wrong.forward(x, {}, out_dtype=torch.float32)
    rel = ((y - y_mcg).norm() / y.norm()).item()
    assert rel > 0.5, "mcg and mul1 decoded the same bytes identically"
    print("   mcg-vs-mul1 relative difference: %.3f (must be large)" % rel)

print("MUL1_LINEAR_OK")
