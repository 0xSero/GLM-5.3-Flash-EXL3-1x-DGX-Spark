# SPDX-License-Identifier: Apache-2.0
"""Dequantize an EXL3 artifact's *non-expert* tensors as the weight stream is read.

The MTP runtime implements EXL3 for routed experts only; every other
`LinearBase` gets `UnquantizedLinearMethod`, which wants a plain `.weight`.
A fully quantized artifact (the M288-12L mosaic: attention, dense MLP, shared
experts, vision tower, `eh_proj`, `lm_head`) therefore has no loader for those
tensors at all.

This module closes that gap without a new `LinearMethod`: it wraps the model
loader's weight iterator, groups each non-expert module's
`trellis / suh / svh / <codebook>` set, reconstructs the dense matrix with
exllamav3's own `reconstruct` kernel (`LinearEXL3.get_weight_tensor`, which is
codebook-aware), and yields a single `<module>.weight` in the model's dtype.
Routed-expert tensors pass through untouched and are still loaded packed by
`Exl3MoEMethod` — they are 37,152 of the artifact's trellis tensors and the
only ones that matter for memory.

Cost on the mosaic: ~5 GB of extra resident weights, in exchange for the
artifact loading at all. Attention, shared experts and the LM head then run
the runtime's normal BF16 kernels.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

CODEBOOK_SUFFIXES = ("mcg", "mul1")
PACKED_SUFFIXES = ("trellis", "suh", "svh") + CODEBOOK_SUFFIXES
EXPERT_MARKER = ".mlp.experts."
# The mosaic descends from the SGLang line and pre-fuses the KDA q/k/v
# projection into one tensor; this runtime's loader wants the three separate
# checkpoint names (its stacked mapping re-fuses them into in_proj_qkvbfg_a).
# The split is only possible once the trellis is reconstructed.
ROW_SPLITS = {".self_attn.qkv_proj": ("q_proj", "k_proj", "v_proj")}


def _split_rows(prefix: str, weight: torch.Tensor):
    for suffix, parts in ROW_SPLITS.items():
        if not prefix.endswith(suffix):
            continue
        rows = weight.shape[0]
        if rows % len(parts):
            raise RuntimeError(
                f"EXL3 cannot split {prefix}: {rows} rows is not divisible by {len(parts)}"
            )
        chunk = rows // len(parts)
        base = prefix[: -len(suffix.rsplit(".", 1)[-1])]
        return [
            (f"{base}{part}.weight", weight[i * chunk : (i + 1) * chunk].contiguous())
            for i, part in enumerate(parts)
        ]
    return [(f"{prefix}.weight", weight)]


def dequant_enabled() -> bool:
    return os.environ.get("EXL3_DEQUANT_NON_EXPERT", "1") != "0"


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.weight_transfer import (
        get_file_tensor_source,
        materialize_weight,
    )

    if get_file_tensor_source(tensor) is not None:
        return materialize_weight(tensor)
    return tensor


def _dequantize(group: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.exl3 import make_linear_exl3

    codebook = "mul1" if "mul1" in group else "mcg"
    device = torch.device("cuda", torch.cuda.current_device())
    trellis = _materialize(group["trellis"]).to(device)
    suh = _materialize(group["suh"]).to(device).half()
    svh = _materialize(group["svh"]).to(device).half()
    marker = _materialize(group[codebook]).reshape(1).to(device)
    linear = make_linear_exl3(trellis, suh, svh, marker, codebook=codebook)
    # get_weight_tensor() is [in_features, out_features]; a torch Linear wants
    # [out_features, in_features].
    weight = linear.get_weight_tensor().t().contiguous().to(dtype)
    del linear, trellis, suh, svh, marker
    return weight


def dequantize_non_expert_stream(
    weights: Iterable[tuple[str, torch.Tensor]],
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[str, torch.Tensor]]:
    pending: dict[str, dict[str, torch.Tensor]] = {}
    converted = 0
    for name, tensor in weights:
        tail = name.rsplit(".", 1)[-1]
        if tail not in PACKED_SUFFIXES or EXPERT_MARKER in name:
            yield name, tensor
            continue
        prefix = name[: -(len(tail) + 1)]
        group = pending.setdefault(prefix, {})
        group[tail] = tensor
        if "trellis" in group and "suh" in group and "svh" in group:
            codebooks = [s for s in CODEBOOK_SUFFIXES if s in group]
            if not codebooks:
                continue
            del pending[prefix]
            converted += 1
            if converted == 1:
                logger.info(
                    "EXL3: reconstructing non-expert quantized tensors to %s "
                    "(first: %s, codebook=%s)",
                    dtype,
                    prefix,
                    codebooks[0],
                )
            yield from _split_rows(prefix, _dequantize(group, dtype))
    if pending:
        raise RuntimeError(
            "EXL3 non-expert dequantization saw incomplete tensor groups: "
            f"{sorted(pending)[:4]} (+{max(0, len(pending) - 4)} more)"
        )
    if converted:
        logger.info("EXL3: reconstructed %d non-expert tensors", converted)


def install() -> None:
    """Wrap `DefaultModelLoader.get_all_weights` once."""
    if not dequant_enabled():
        return
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if getattr(DefaultModelLoader, "_exl3_dequant_patched", False):
        return
    original = DefaultModelLoader.get_all_weights

    def get_all_weights(self, model_config, model):  # type: ignore[no-untyped-def]
        dtype = getattr(model_config, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        yield from dequantize_non_expert_stream(
            original(self, model_config, model), dtype=dtype
        )

    DefaultModelLoader.get_all_weights = get_all_weights
    DefaultModelLoader._exl3_dequant_patched = True
    logger.info("EXL3: non-expert dequantization hook installed")
