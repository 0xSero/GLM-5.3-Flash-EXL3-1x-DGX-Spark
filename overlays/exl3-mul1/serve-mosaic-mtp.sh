RT=/root/rt
export PYTHONHOME=$RT/usr
export PYTHONPATH=$RT/usr/lib/python3/dist-packages:$RT/usr/local/lib/python3.12/dist-packages:$RT/usr/lib/python3.12:$RT/usr/lib/python3.12/lib-dynload
export LD_LIBRARY_PATH=$RT/usr/local/lib/python3.12/dist-packages/torch/lib:$RT/usr/local/cuda/lib64:$RT/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu
export CUDA_HOME=$RT/usr/local/cuda HF_HUB_OFFLINE=1
export SAFETENSORS_DROP_PAGE_CACHE=1 SAFETENSORS_LOAD_DEVICE=cuda:0
export VLLM_USE_AOT_COMPILE=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_MXFP8_LM_HEAD=0 VLLM_MTP_NVFP4_LM_HEAD=0
export VLLM_EXL3_TRELLIS_MIN_M=1 VLLM_EXL3_TRELLIS_MAX_M=128 VLLM_EXL3_PREFILL_TRELLIS=1
export EXL3_FUSED_MOE=${EXL3_FUSED_MOE:-0} EXL3_FUSED_TEMP_ROWS=4096
CTX=${CTX:-32768}
exec $RT/usr/bin/python3.12 -m vllm.entrypoints.openai.api_server \
  --model /root/mosaic --served-model-name glm-5.3-flash \
  --host 127.0.0.1 --port 8888 --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 --no-enable-expert-parallel \
  --quantization exl3 --load-format safetensors --dtype bfloat16 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --gpu-memory-utilization ${MEMF:-0.92} \
  --max-model-len $CTX --max-num-seqs 1 --max-num-batched-tokens 7168 \
  --attention-backend B12X --additional-config '{"kda_prefill_backend":"b12x"}' \
  --enable-chunked-prefill --no-enable-prefix-caching --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}' \
  --generation-config vllm --reasoning-parser glm47 --tool-call-parser glm47 \
  --enable-auto-tool-choice --trust-remote-code \
  --speculative-config '{"method":"mtp","model":"/root/mtp","num_speculative_tokens":2,"attention_backend":"B12X"}'
