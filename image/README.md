# Image build (public sources, pinned)

`Dockerfile`, `compile.sh`, and `build.sh` reproduce the runtime image
`ghcr.io/0xsero/glm53-b12x-exl3:b12x-mtp-1xspark-v1` from public sources:

- base: `ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:5ea6d04d…` (public, digest-pinned)
- vLLM fork: `github.com/local-inference-lab/vllm` pinned to
  `3aada67721bfdb8b98355207ca3780bb32e6f434` (fetched by digest in-Dockerfile)
- EXL3 kernels: trellis reference implementation, pinned in `dependency-overrides.txt`
- torch `2.13.0+cu130`, CUDA 13, `TORCH_CUDA_ARCH_LIST=12.1a`, arm64 only

Build stages: `toolchain` → in-container `compile.sh` (vLLM + EXL3 kernels,
needs ~80 GB free host memory for the 4-job build) → `runtime-fast` → `runtime`.
`build.sh` ends with driver-backed CPU import gates (no visible CUDA device)
and refuses to publish a runtime that fails them.

Most users should just `docker pull` the published image — `./start.sh` does.
Building yourself takes hours (vLLM + kernels from source).

Public build provenance receipts (image ID pins, dependency audits, CPU-gate
receipts) live in the private program archive next to the measurements.
