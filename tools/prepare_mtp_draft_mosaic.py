#!/usr/bin/env python3
"""Derive the native MTP draft view from a *quantized* EXL3 checkpoint (the mosaic).

`prepare_mtp_draft.py` assumes the MTP block is native BF16, which is true of the
K2/TR3 throughput artifact and false of the mosaic: its layer 45 (and `lm_head`)
are EXL3 `mul1` like everything else. This tool builds the same kind of view —
filtered index plus symlinks, no tensor copied or modified — but keeps the
artifact's `quantization_config.json` so the draft loads through the EXL3 path.
"""

import argparse
import copy
import json
from pathlib import Path

MTP_LAYER_PREFIX_TEMPLATE = "model.language_model.layers.{layer}."
SHARED_PREFIXES = ("model.language_model.embed_tokens.", "lm_head.")


def prepare(source: Path, output: Path, container_source: str) -> dict:
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    assert int(text["num_nextn_predict_layers"]) == 1, "expected 1 native MTP layer"
    mtp_prefix = MTP_LAYER_PREFIX_TEMPLATE.format(layer=layers)

    index = json.loads((source / "model.safetensors.index.json").read_text())
    keys = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith(mtp_prefix) or name.startswith(SHARED_PREFIXES)
    }
    assert keys, f"no tensors found under {mtp_prefix}"
    experts = len({k.split(".mlp.experts.")[1].split(".")[0] for k in keys if ".mlp.experts." in k})
    assert experts == int(text["n_routed_experts"]), (
        f"MTP block carries {experts} experts, config declares {text['n_routed_experts']}"
    )

    out_config = copy.deepcopy(config)
    out_config["architectures"] = ["Glm5NextForConditionalGeneration"]
    if "image_token_id" in out_config:
        out_config["image_token_index"] = out_config["image_token_id"]

    output.mkdir(parents=True, exist_ok=True)
    for shard in sorted(set(keys.values())):
        link = output / shard
        if not link.exists():
            link.symlink_to(str(Path(container_source) / shard))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "quantization_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
    ):
        if (source / name).is_file() and not (output / name).exists():
            (output / name).symlink_to(str(Path(container_source) / name))
    (output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": keys}, indent=2) + "\n"
    )
    receipt = {
        "source": str(source),
        "mtp_layer": layers,
        "keys": len(keys),
        "experts": experts,
        "shards": sorted(set(keys.values())),
        "quantized_draft": True,
    }
    (output / "native-mtp-view.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--container-source", default=None)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.container_source or str(a.source)), indent=2))


if __name__ == "__main__":
    main()
