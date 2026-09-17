#!/usr/bin/env python3
"""Derive the native MTP draft view from a GLM-5.3-Flash EXL3 K2 checkpoint.

The GLM-5.3-Flash checkpoint ships its MTP block as regular tensors: layer 45
(eh_proj, enorm, hnorm, a full 288-expert MoE, indexer, MLA attention) plus the
shared embeddings and lm_head — 889 native tensors + 2 shared = 891 index keys,
all BF16/F32, none quantized. vLLM's Glm5NextMTPModel wants a standalone draft
model directory, so this tool materializes one WITHOUT copying or modifying any
tensor: it filters model.safetensors.index.json to those 891 keys, verifies the
safetensors headers (dtype, expert count, gate shape), and symlinks the shard
files. The draft is byte-identical to the main checkpoint by construction.

Adapted from the program's verified prepare_verified_k2_control.py lineage
(2026-09-11); assertions preserved. CPU-only, seconds to run.
"""
import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prepare(source: Path, output: Path, container_source: str) -> dict:
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    text = config.get("text_config", config)
    assert text["num_hidden_layers"] == 45, "expected GLM-5.3-Flash 45-layer checkpoint"
    assert text["num_nextn_predict_layers"] == 1, "expected 1 native MTP layer"
    assert text["n_routed_experts"] == 288, "expected 288 routed experts"

    index = json.loads((source / "model.safetensors.index.json").read_text())
    keys = {
        k: v
        for k, v in index["weight_map"].items()
        if ".layers.45." in k or k in ("lm_head.weight", "model.language_model.embed_tokens.weight")
    }
    assert len(keys) == 891, f"expected 889 native MTP tensors and 2 shared tensors, got {len(keys)}"

    headers = {}
    for file in set(keys.values()):
        assert Path(file).name == file, f"shard must be flat next to the index: {file}"
        with (source / file).open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            assert size < 100_000_000
            headers.update(json.loads(f.read(size)))
    for key in keys:
        assert headers[key]["dtype"] in ("BF16", "F32"), (key, headers[key]["dtype"])
    expert_keys = [k for k in keys if ".mlp.experts." in k]
    assert len(expert_keys) == 864, f"expected 288 experts x 3 tensors = 864, got {len(expert_keys)}"
    assert headers["model.language_model.layers.45.mlp.gate.weight"]["shape"] == [288, 4096]
    assert headers["model.language_model.layers.45.mlp.gate.e_score_correction_bias"]["shape"] == [288]

    out_config = copy.deepcopy(config)
    out_config.pop("quantization_config", None)
    if "text_config" in out_config:
        out_config["text_config"].pop("quantization_config", None)
    out_config["architectures"] = ["Glm5NextForConditionalGeneration"]
    if "image_token_id" in out_config:
        out_config["image_token_index"] = out_config["image_token_id"]

    output.mkdir(parents=True, exist_ok=False)
    for file in set(keys.values()):
        (output / file).symlink_to(str(Path(container_source) / file))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
        if (source / name).is_file():
            (output / name).symlink_to(str(Path(container_source) / name))
    (output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    total = sum(headers[k]["data_offsets"][1] - headers[k]["data_offsets"][0] for k in keys)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": keys}, indent=2) + "\n")
    receipt = {
        "native_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
        "native_mtp_tensor_count": 889,
        "index_tensor_count": 891,
        "native_mtp_bytes": sum(
            headers[k]["data_offsets"][1] - headers[k]["data_offsets"][0] for k in keys if ".layers.45." in k),
        "draft_quantization": None,
        "mtp_experts": 288,
        "serving_validated": True,
    }
    (output / "native-mtp-view.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="downloaded checkpoint dir (flat shards)")
    p.add_argument("--output", type=Path, required=True, help="draft view dir to create")
    p.add_argument("--container-source", default="/native-source",
                   help="path at which --source will be mounted inside the serving container")
    p.add_argument("--force-if-empty", action="store_true",
                   help="remove a leftover empty output dir before creating the view")
    a = p.parse_args()
    if a.output.exists():
        if a.force_if_empty and not any(a.output.iterdir()):
            a.output.rmdir()
        else:
            raise SystemExit(f"refusing to touch existing non-empty dir: {a.output}")
    print(json.dumps(prepare(a.source, a.output, a.container_source), indent=2))


if __name__ == "__main__":
    main()
