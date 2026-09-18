#!/usr/bin/env bash
# ============================================================================
# start.sh — 1× DGX Spark runtime for GLM-5.3-Flash EXL3 K2 2.0bpw + native MTP
# ============================================================================
#
#   ./start.sh            start (pull image, download weights, derive MTP
#                         draft view, launch, wait healthy) — default
#   ./start.sh download   weights download only
#   ./start.sh stop       stop the serving container (preserved, not deleted)
#   ./start.sh restart    stop + start
#   ./start.sh status     container state + API health
#   ./start.sh logs       follow engine logs
#
# Env wins over .env. Knobs: IMAGE, WEIGHTS_REPO, WEIGHTS_DIR, PORT,
# CONTEXT_LEN, MEM_FRACTION, SPEC_DEPTH, HF_TOKEN, SKIP_PULL, SKIP_DOWNLOAD.
#
# Safety rules this script follows:
#   - refuses to start if a glm53 serving container already exists or another
#     GPU compute job is visible — one GPU job per node;
#   - never deletes containers, weights, or the derived MTP view;
#   - never prints or stores HF_TOKEN (passed through to the downloader only).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-1x}"
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[glm53-1x %s] %s\n' "$(TS)" "$*"; }
die() { printf '[glm53-1x %s] ERROR: %s\n' "$(TS)" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- .env
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    log "wrote .env from .env.example (defaults are the measured recipe)"
fi
if [ -f .env ]; then
    # Caller-provided exports win over .env.
    while IFS='=' read -r k v; do
        case "$k" in ''|\#*) continue ;; esac
        var="${k%% }"; [ -n "${!var+x}" ] || export "$var=$v"
    done < <(grep -v '^\s*$' .env)
fi

IMAGE="${IMAGE:-ghcr.io/0xsero/glm53-b12x-exl3:b12x-mtp-1xspark-v2}"
WEIGHTS_REPO="${WEIGHTS_REPO:-0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$SCRIPT_DIR/models/$(basename "$WEIGHTS_REPO")}"
MTP_DIR="${MTP_DIR:-$SCRIPT_DIR/mtp}"
PORT="${PORT:-8888}"
CONTEXT_LEN="${CONTEXT_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.93}"
SPEC_DEPTH="${SPEC_DEPTH:-2}"
TRELLIS_MAX_M="${TRELLIS_MAX_M:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"

CMD="${1:-start}"

# ---------------------------------------------------------------- preflight
preflight() {
    command -v docker >/dev/null || die "docker not found"
    docker info >/dev/null 2>&1 || die "docker daemon not responding"
    if [ "$CMD" = "start" ]; then
        local need_kb=140000000  # ~140 GB for weights + headroom
        local avail_kb
        avail_kb=$(df -Pk "$(dirname "$WEIGHTS_DIR")" | awk 'NR==2 {print $4}')
        [ "${avail_kb:-0}" -ge "$need_kb" ] || die "only ${avail_kb} kB free under $WEIGHTS_DIR; need ~$need_kb kB"
        log "disk ok ($((avail_kb / 1048576)) GB free)"
    fi
}

# ---------------------------------------------------------------- image
ensure_image() {
    if [ "${SKIP_PULL:-0}" = "1" ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
        log "using local image $IMAGE"
        return
    fi
    if docker image inspect "$IMAGE" >/dev/null 2>&1 && [ "${PULL:-0}" != "1" ]; then
        log "image $IMAGE already present (PULL=1 to refresh)"
        return
    fi
    log "pulling $IMAGE (arm64, ~26 GB)…"
    docker pull --platform linux/arm64 "$IMAGE"
}

# ---------------------------------------------------------------- weights
weights_complete() {
    [ -f "$WEIGHTS_DIR/config.json" ] && [ -f "$WEIGHTS_DIR/model.safetensors.index.json" ] || return 1
    local want got
    want=$(python3 - "$WEIGHTS_DIR/model.safetensors.index.json" <<'PY' 2>/dev/null || echo 0
import json,sys
print(len({f for f in json.load(open(sys.argv[1]))["weight_map"].values()}))
PY
)
    got=$(find "$WEIGHTS_DIR" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l)
    [ "${want:-0}" -gt 0 ] && [ "$want" = "$got" ]
}

download_weights() {
    if weights_complete; then
        log "weights already complete in $WEIGHTS_DIR"
        return
    fi
    mkdir -p "$WEIGHTS_DIR"
    log "downloading $WEIGHTS_REPO into $WEIGHTS_DIR (~111 GB; resumes on re-run)"
    if command -v hf >/dev/null 2>&1; then
        HF_TOKEN="${HF_TOKEN:-}" hf download "$WEIGHTS_REPO" --local-dir "$WEIGHTS_DIR"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_TOKEN="${HF_TOKEN:-}" huggingface-cli download "$WEIGHTS_REPO" --local-dir "$WEIGHTS_DIR"
    else
        log "installing huggingface_hub (user scope) for the download"
        python3 -m pip install --quiet --user -U "huggingface_hub[hf_transfer]"
        HF_TOKEN="${HF_TOKEN:-}" python3 -m huggingface_hub.commands.huggingface_cli download \
            "$WEIGHTS_REPO" --local-dir "$WEIGHTS_DIR"
    fi
    weights_complete || die "weights incomplete after download (re-run to resume)"
    log "weights verified against index"
}

# ---------------------------------------------------------------- MTP draft
derive_mtp() {
    if [ -f "$MTP_DIR/model.safetensors.index.json" ] && [ -f "$MTP_DIR/native-mtp-view.json" ]; then
        log "MTP draft view already present in $MTP_DIR"
        return
    fi
    log "deriving native MTP draft view (CPU-only; symlinks + filtered index)"
    python3 tools/prepare_mtp_draft.py \
        --source "$WEIGHTS_DIR" \
        --output "$MTP_DIR" \
        --container-source /native-source \
        --force-if-empty
    log "MTP draft view ready (891 keys expected: layer 45 + embeddings + lm_head)"
}

# ---------------------------------------------------------------- launch
gpu_busy() {
    docker ps --format '{{.Names}}' | grep -v "^$CONTAINER_NAME$" | grep -qi 'glm53' && return 0
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]' && return 0
    return 1
}

launch() {
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && \
        die "container $CONTAINER_NAME exists (it is preserved, never deleted). Use: ./start.sh restart"
    if gpu_busy; then
        die "a GPU compute job or another glm53 container is running — one GPU job per node"
    fi
    docker run -d --pull never --gpus all --ipc host --network host \
        --name "$CONTAINER_NAME" \
        -v "$WEIGHTS_DIR":/model:ro \
        -v "$WEIGHTS_DIR":/native-source:ro \
        -v "$MTP_DIR":/mtp:ro \
        -e HF_HUB_OFFLINE=1 \
        -e SAFETENSORS_DROP_PAGE_CACHE=1 \
        -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
        -e VLLM_USE_AOT_COMPILE=1 \
        -e VLLM_USE_V2_MODEL_RUNNER=1 \
        -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
        -e VLLM_MXFP8_LM_HEAD=0 \
        -e VLLM_MTP_NVFP4_LM_HEAD=0 \
        -e GLM53_MTP_EXPERT_FP8=1 \
        -e VLLM_EXL3_TRELLIS_MIN_M=1 \
        -e VLLM_EXL3_TRELLIS_MAX_M="$TRELLIS_MAX_M" \
        -e VLLM_EXL3_PREFILL_TRELLIS=1 \
        -e EXL3_FUSED_MOE=1 \
        -e EXL3_FUSED_TEMP_ROWS="${EXL3_FUSED_TEMP_ROWS:-4096}" \
        "$IMAGE" \
        --model /model --served-model-name glm-5.3-flash \
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 \
        --decode-context-parallel-size 1 --no-enable-expert-parallel \
        --quantization exl3 --load-format safetensors --dtype bfloat16 \
        --kv-cache-dtype fp8_ds_mla --block-size 256 \
        --gpu-memory-utilization "$MEM_FRACTION" \
        --max-model-len "$CONTEXT_LEN" --max-num-seqs 1 \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --attention-backend B12X \
        --additional-config '{"kda_prefill_backend":"b12x"}' \
        --enable-chunked-prefill --no-enable-prefix-caching \
        --mm-processor-cache-gb 0.1 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}' \
        --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
        --enable-auto-tool-choice --trust-remote-code \
        --speculative-config "{\"method\":\"mtp\",\"model\":\"/mtp\",\"num_speculative_tokens\":$SPEC_DEPTH,\"attention_backend\":\"B12X\"}"
    log "container $CONTAINER_NAME launched; waiting for /health (load + graph capture takes minutes)…"
    wait_healthy
}

wait_healthy() {
    local i
    for i in $(seq 1 360); do
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            log "API healthy on http://127.0.0.1:$PORT (served id: glm-5.3-flash)"
            log "measured on 1× DGX Spark: ~19 tok/s decode, ~450+ tok/s prefill, 262,144 ctx (docs/PERFORMANCE.md)"
            return 0
        fi
        sleep 5
    done
    log "not healthy after 30 min — recent log:"
    docker logs --tail 40 "$CONTAINER_NAME" >&2 || true
    die "server did not become healthy"
}

case "$CMD" in
    start)
        preflight; ensure_image; download_weights; derive_mtp; launch ;;
    download)
        preflight; download_weights ;;
    stop)
        docker stop -t 120 "$CONTAINER_NAME" && log "stopped $CONTAINER_NAME (container preserved)" ;;
    restart)
        docker stop -t 120 "$CONTAINER_NAME" >/dev/null 2>&1 || true
        sleep 3
        preflight; launch ;;
    status)
        docker ps -a --filter "name=^/$CONTAINER_NAME$" --format 'table {{.Names}}\t{{.Status}}'
        if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            log "API healthy on :$PORT"
        else
            log "API not responding on :$PORT"
        fi ;;
    logs)
        docker logs -f "$CONTAINER_NAME" ;;
    *)
        die "unknown command: $CMD (start|download|stop|restart|status|logs)" ;;
esac
