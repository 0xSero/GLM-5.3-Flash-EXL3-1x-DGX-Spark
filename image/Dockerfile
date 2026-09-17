# syntax=docker/dockerfile:1.7
# Build only on linux/arm64. The launch script verifies the local base image ID.
ARG BASE_IMAGE=ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:5ea6d04d65d1a2f30f632815776fb765359529ab117818ba2b5e5c5480e34963
ARG COMPILED_IMAGE=compiled
FROM ${BASE_IMAGE} AS toolchain
ARG BUILD_JOBS=4
ENV UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy \
    MAX_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    CARGO_BUILD_JOBS=${BUILD_JOBS} TORCH_CUDA_ARCH_LIST=12.1a \
    VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0 \
    VLLM_REQUIRE_RUST_FRONTEND=1 CUDA_HOME=/usr/local/cuda \
    RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH}
USER root
WORKDIR /opt/src
RUN python3 -c 'import platform,torch; assert platform.machine()=="aarch64"; assert torch.__version__=="2.13.0+cu130"; assert not torch.cuda.is_initialized()'
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates pkg-config protobuf-compiler libprotobuf-dev libssl-dev \
      build-essential liburing-dev && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh && \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.95.0 --no-modify-path && \
    rm /tmp/rustup.sh
RUN git init vllm && git -C vllm remote add origin https://github.com/local-inference-lab/vllm.git && \
    git -C vllm fetch --depth 1 origin 3aada67721bfdb8b98355207ca3780bb32e6f434 && \
    git -C vllm checkout --detach FETCH_HEAD && git -C vllm submodule update --init --recursive
COPY build-patches /patches/build-patches
RUN VLLM_PRESERVE_SM12X_TARGET=1 python3 /patches/build-patches/patch_vllm_preserve_sm12x_target.py /opt/src/vllm && \
    python3 /patches/build-patches/patch_vllm_mrv2_speculator_cudagraph_pool.py /opt/src/vllm
RUN uv pip install -r /opt/src/vllm/requirements/build/cuda.txt
RUN git init b12x && git -C b12x remote add origin https://github.com/local-inference-lab/b12x.git && \
    git -C b12x fetch --depth 1 origin 3b862805d2b7fd52e6fe507fd28038bd48797cf5 && \
    git -C b12x checkout --detach FETCH_HEAD
COPY dependency-overrides.txt /opt/dependency-overrides.txt
COPY compile.sh verify_bootstrap_exllama.py exllama-bootstrap-pin.json /opt/
ENTRYPOINT ["bash"]
CMD ["/opt/compile.sh"]

# Public all-in-one build compiles here. Fast local builds can run this same
# script in a memory/CPU-limited container and use its committed result instead.
FROM toolchain AS compiled
RUN bash /opt/compile.sh

FROM ${COMPILED_IMAGE} AS compiled_source

FROM scratch AS runtime_files
COPY exl3-port/apply_port.py exl3-port/apply_config_override.py exl3-port/exl3.py exl3-port/PORT_MANIFEST.json exl3-port/cpu_import_smoke.py /patches/exl3-port/
COPY apply_native_mtp_port.py mtp_expert_fp8.py /patches/
COPY smoke_build.py test_mtp_external_embeddings.py test_mtp_fp8_scope.py test_b12x_cache_contract.py finalize_runtime.sh verify_runtime.sh verify_bootstrap_exllama.py exllama-bootstrap-pin.json /opt/

# Fast experimental image: retained privately; includes build workspace.
FROM compiled_source AS runtime-fast
ARG COMPILED_IMAGE
ENV VLLM_MXFP8_LM_HEAD=0 VLLM_MTP_NVFP4_LM_HEAD=0
ENV GLM53_RUNTIME_BASE_IMAGE=${COMPILED_IMAGE}
COPY --from=runtime_files / /
COPY source-pins.json /opt/source-pins.json
WORKDIR /
RUN bash /opt/finalize_runtime.sh
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD []

# Publication image starts from the verified bootstrap, not the compiled
# filesystem. BuildKit exposes wheels only during installation, so source,
# object files, Rust targets and build caches never enter release layers.
FROM ${BASE_IMAGE} AS runtime
ARG BASE_IMAGE
ENV VLLM_MXFP8_LM_HEAD=0 VLLM_MTP_NVFP4_LM_HEAD=0
ENV GLM53_RUNTIME_BASE_IMAGE=${BASE_IMAGE}
USER root
ENV UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=4
COPY --from=runtime_files / /
COPY dependency-overrides.txt /opt/dependency-overrides.txt
COPY source-pins.public.json /opt/source-pins.json
RUN apt-get update && apt-get install -y --no-install-recommends liburing2 libssl3t64 && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=compiled_source,source=/opt/wheels,target=/opt/wheels,readonly \
    uv pip uninstall vllm && \
    uv pip install /opt/wheels/vllm-*.whl --override /opt/dependency-overrides.txt && \
    uv pip install /opt/wheels/b12x-*.whl --override /opt/dependency-overrides.txt && \
    rm -rf /root/.cache/uv
WORKDIR /
RUN bash /opt/finalize_runtime.sh
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD []
