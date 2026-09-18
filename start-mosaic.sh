#!/usr/bin/env bash
# start-mosaic.sh — serve the M288-12L mosaic from this kit, on its own runtime.
#
#   ./start-mosaic.sh              download (if needed) + census + serve
#   ./start-mosaic.sh download     weights only
#   ./start-mosaic.sh serve        census + serve (assumes weights present)
#
# Why a separate script instead of start.sh: the mosaic is codebook `mul1` and
# this kit's vLLM/MTP overlay only implements `mcg` — it refuses the model at
# config validation ("this overlay only implements codebook=mcg; got 'mul1'").
# The mosaic therefore runs on SGLang, without MTP, at ~9–10 tok/s decode in
# exchange for the best quality in the family. Read docs/MOSAIC.md first.
#
# Nothing here is guessed: the flags are the ones the mosaic's confirming run
# (full G4 panel + MMLU/GPQA) was measured with.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_REPO="${GLM53_MOSAIC_REPO:-0xSero/GLM-5.3-Flash-EXL3-M288-Mosaic-12L}"
HF_PIN="${GLM53_MOSAIC_PIN:-2642851741fc833764e77d03039117be559dc83e}"
MODEL="${GLM53_MOSAIC_DIR:-$HERE/models/mosaic-12l}"
IMG="${GLM53_MOSAIC_IMG:-ghcr.io/0xsero/glm53-flash-exl3-plain:2p05-sglang-mul1-r1}"
NAME="${GLM53_MOSAIC_CONTAINER:-glm53-mosaic-12l}"
PORT="${GLM53_PORT:-8888}"
RECEIPTS="${GLM53_RECEIPTS:-$HERE/receipts/mosaic}"
MEM="${GLM53_MEM_FRACTION:-0.95}"
MAXRUN="${GLM53_MAX_RUNNING:-1}"
CMD="${1:-up}"

say() { echo "$@" | tee -a "$RECEIPTS/RUN.txt"; }
mkdir -p "$RECEIPTS"

download() {
  if [ -f "$MODEL/sha256-manifest.txt" ] && [ -f "$MODEL/model-00012-of-00012.safetensors" ]; then
    echo "### weights already present at $MODEL (re-run of start.sh-style resume skips)"
  fi
  echo "### downloading $HF_REPO @ $HF_PIN -> $MODEL"
  python3 - "$HF_REPO" "$HF_PIN" "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, pin, dest = sys.argv[1], sys.argv[2], sys.argv[3]
print("downloaded:", snapshot_download(repo, revision=pin, local_dir=dest,
                                       ignore_patterns=[".cache/*"]))
EOF
  echo "### verifying against the repo's own sha256-manifest.txt"
  ( cd "$MODEL" && sha256sum -c sha256-manifest.txt ) || {
    echo "VERIFY_FAILED — refusing to serve a download that does not match the manifest" >&2
    exit 1
  }
}

serve() {
  [ -d "$MODEL" ] || { echo "no weights at $MODEL — run: $0 download" >&2; exit 1; }
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    echo "GPU already has a compute process; refusing to launch (stop your own container first)" >&2
    exit 1
  fi
  docker rm -f "$NAME" 2>/dev/null || true

  say "### census $(date -u +%FT%TZ)"
  docker run --rm --gpus all -v "$MODEL":/model:ro -v "$RECEIPTS":/receipts \
    --entrypoint python3 "$IMG" \
    -m exl3_plain_sglang_overlay.census /model --out /receipts/census.json >> "$RECEIPTS/census.log" 2>&1
  python3 - "$RECEIPTS/census.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["schema"] == "exl3-plain-census-v1" and r["index_only"] is False \
    and r["verdict"]["contract_ok"] is True, "census contract check FAILED"
print("census_ok tensors=%s codebook=%s" % (r["tensors"], r["codebook"]))
EOF

  say "### serve $(date -u +%FT%TZ) port=$PORT mem=$MEM maxrun=$MAXRUN"
  docker run -d --name "$NAME" \
    -e EXL3_PLAIN_CENSUS=/receipts/census.json \
    -e EXL3_PLAIN_DECODER=exl3_plain_sglang_overlay.exl3_reference:decode \
    -e SGLANG_EXL3_MAX_BATCH_TOKENS=256 \
    -v "$MODEL":/model:ro -v "$RECEIPTS":/receipts:ro \
    -p "$PORT":8000 --gpus all --shm-size=16g \
    --entrypoint python3 "$IMG" \
    -m sglang.launch_server --model-path /model --host 0.0.0.0 --port 8000 \
    --quantization exl3 --tp-size 1 --ep-size 1 --context-length 262144 \
    --kv-cache-dtype fp8_e4m3 --attention-backend dsa \
    --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
    --linear-attn-backend triton --disable-shared-experts-fusion \
    --chunked-prefill-size 256 --max-prefill-tokens 256 --max-running-requests "$MAXRUN" \
    --mem-fraction-static "$MEM" --enable-multimodal \
    --chat-template /opt/glm53/chat-template-mm.jinja --reasoning-parser glm45 \
    --tool-call-parser glm47 --disable-cuda-graph

  echo "### waiting for /health (first load takes several minutes for 96 GB)"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      say "HEALTH_OK $(date -u +%FT%TZ) — serving on http://127.0.0.1:$PORT/v1  (no MTP: expect ~9-10 tok/s)"
      exit 0
    fi
    if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
      echo "CONTAINER_DIED — logs:"; docker logs --tail 60 "$NAME"; exit 1
    fi
    sleep 10
  done
  echo "HEALTH_TIMEOUT — logs:"; docker logs --tail 60 "$NAME"; exit 1
}

case "$CMD" in
  up)       download; serve ;;
  download) download ;;
  serve)    serve ;;
  *) echo "usage: $0 [up|download|serve]" >&2; exit 2 ;;
esac