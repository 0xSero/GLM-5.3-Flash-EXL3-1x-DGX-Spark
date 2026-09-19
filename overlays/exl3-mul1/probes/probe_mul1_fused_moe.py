#!/usr/bin/env python3
"""The fused `exl3_moe` launch with mul1 flags vs the LinearEXL3 reference loop.

    MOSAIC_DIR=/model python3 probe_mul1_fused_moe.py 4

On the shipped runtime image (exllamav3 0.0.43) this probe FAILS on purpose:
the compiled kernel refuses the codebook with

    RuntimeError: MoE kernel: Only mcg codebook is currently supported

which is the receipt for blocker 4 in ../README.md. Upstream exllamav3 has
since instantiated the mul1 MoE path (`cb_idx = gate_mul1 ? 1 : 0` in
`exl3_moe.cu`), so the probe is expected to pass once the image is rebuilt
against it; until then the routed experts must run the Python loop.
"""

import json
import os
import sys
import types

import torch
from safetensors import safe_open

import exl3 as OV

SRC = os.environ.get("MOSAIC_DIR", "/model")
DEV = os.environ.get("PROBE_DEVICE", "cuda:0")
NE = int(os.environ.get("PROBE_EXPERTS", "8"))
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4

index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]


def get(key: str) -> torch.Tensor:
    with safe_open(f"{SRC}/{index[key]}", "pt") as f:
        return f.get_tensor(key)


base = f"model.language_model.layers.{LAYER}.mlp.experts"
gate0 = get(f"{base}.0.gate_proj.trellis")
down0 = get(f"{base}.0.down_proj.trellis")
hidden = get(f"{base}.0.gate_proj.suh").numel()
inter = get(f"{base}.0.gate_proj.svh").numel()
K = gate0.shape[-1] // 16
print(f"layer {LAYER}: hidden={hidden} intermediate={inter} K={K} experts={NE}")

w13_tr = torch.empty(NE, 2, *gate0.shape, dtype=torch.int16)
w13_suh = torch.empty(NE, 2, hidden, dtype=torch.float16)
w13_svh = torch.empty(NE, 2, inter, dtype=torch.float16)
w13_marker = torch.empty(NE, 2, 1, dtype=torch.int32)
w2_tr = torch.empty(NE, *down0.shape, dtype=torch.int16)
w2_suh = torch.empty(NE, inter, dtype=torch.float16)
w2_svh = torch.empty(NE, hidden, dtype=torch.float16)
w2_marker = torch.empty(NE, 1, dtype=torch.int32)
for e in range(NE):
    for j, proj in enumerate(("gate_proj", "up_proj")):
        w13_tr[e, j] = get(f"{base}.{e}.{proj}.trellis")
        w13_suh[e, j] = get(f"{base}.{e}.{proj}.suh")
        w13_svh[e, j] = get(f"{base}.{e}.{proj}.svh")
        w13_marker[e, j] = get(f"{base}.{e}.{proj}.mul1").reshape(1)
    w2_tr[e] = get(f"{base}.{e}.down_proj.trellis")
    w2_suh[e] = get(f"{base}.{e}.down_proj.suh")
    w2_svh[e] = get(f"{base}.{e}.down_proj.svh")
    w2_marker[e] = get(f"{base}.{e}.down_proj.mul1").reshape(1)

layer = types.SimpleNamespace(
    _exl3_hidden_size=hidden,
    _exl3_intermediate_local=inter,
    _exl3_bits=K,
    _exl3_k=K,
    _exl3_codebook="mul1",
    _exl3_rank_stacked_tp=1,
    _exl3_physical_experts=NE,
    expert_map=None,
)
inners = []
for e in range(NE):
    inners.append(
        {
            "gate": OV.make_linear_exl3(
                w13_tr[e, 0].to(DEV), w13_suh[e, 0].to(DEV), w13_svh[e, 0].to(DEV),
                w13_marker[e, 0].to(DEV), codebook="mul1",
            ),
            "up": OV.make_linear_exl3(
                w13_tr[e, 1].to(DEV), w13_suh[e, 1].to(DEV), w13_svh[e, 1].to(DEV),
                w13_marker[e, 1].to(DEV), codebook="mul1",
            ),
            "down": OV.make_linear_exl3(
                w2_tr[e].to(DEV), w2_suh[e].to(DEV), w2_svh[e].to(DEV),
                w2_marker[e].to(DEV), codebook="mul1",
            ),
        }
    )
layer._exl3_inners = inners
OV.build_exl3_fused_state(layer, inners)

torch.manual_seed(0)
tokens, topk = 16, 4
x = torch.randn(tokens, hidden, device=DEV, dtype=torch.bfloat16) / 4
ids = torch.randint(0, NE, (tokens, topk), device=DEV)
weights = torch.rand(tokens, topk, device=DEV, dtype=torch.float32)
weights = weights / weights.sum(-1, keepdim=True)

reference = OV.apply_exl3_experts(x, ids, weights, layer, fused=False)
print("loop std %.4f (LinearEXL3 mul1 reference)" % reference.float().std().item())

fused = OV.apply_exl3_experts(x, ids, weights, layer, fused=True)
rel = ((reference - fused).norm() / reference.norm()).item()
print("fused std %.4f relative diff %.3e" % (fused.float().std().item(), rel))
assert rel < 2e-2, "fused mul1 path disagrees with the LinearEXL3 loop"

# Negative control: the same launch with the mcg flag must be wrong.
layer._exl3_codebook = "mcg"
bad = OV.apply_exl3_experts(x, ids, weights, layer, fused=True)
bad_rel = ((reference - bad).norm() / reference.norm()).item()
layer._exl3_codebook = "mul1"
assert bad_rel > 0.5, "the mcg flag decoded mul1 weights identically"
print("fused-with-mcg-flag relative diff %.3f (must be large)" % bad_rel)
print("FUSED_MUL1_OK")
