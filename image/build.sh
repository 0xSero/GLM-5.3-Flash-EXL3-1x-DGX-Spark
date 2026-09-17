#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
expected=sha256:ddc1b6cf8d89f1f0c0294a9a7a3c86d63cb7ad486d236013d75fb46914e2c576
actual=$(docker image inspect "$expected" --format '{{.Id}}')
[[ "$actual" == "$expected" ]] || { echo 'Wrong observer base image' >&2; exit 1; }
docker tag "$expected" glm53-observer-build-base:ddc1b6cf
docker build --target toolchain --progress plain --build-arg BUILD_JOBS=4 \
  --build-arg BASE_IMAGE=glm53-observer-build-base:ddc1b6cf \
  --tag glm53-b12x-toolchain:3aada677 .
# Do not reuse/overwrite a previous experiment container; it retains failure logs.
name="glm53-b12x-compile-${GLM53_BUILD_ATTEMPT:-attempt1}"
awk '/MemAvailable:/ { if ($2 < 83886080) exit 1 }' /proc/meminfo || {
  echo 'Four-job build needs at least80GiB host memory available' >&2; exit 1;
}
docker run --name "$name" --cpus=4 --memory=64g --memory-swap=64g \
  --shm-size=1g glm53-b12x-toolchain:3aada677 /opt/compile.sh
docker cp "$name:/opt/compile-complete.json" "${name}-complete.json"
docker commit "$name" glm53-b12x-compiled:3aada677
docker build --target runtime-fast --progress plain \
  --build-arg COMPILED_IMAGE=glm53-b12x-compiled:3aada677 \
  --tag glm53-b12x-exl3:jovian-3aada677-test .
# Publish only this slim runtime after all GPU/model acceptance gates pass.
docker build --target runtime --progress plain \
  --build-arg COMPILED_IMAGE=glm53-b12x-compiled:3aada677 \
  --tag glm53-b12x-exl3:jovian-3aada677 .

# Real driver-backed CPU import gates: expose driver libraries but no visible CUDA device.
gate="glm53-b12x-cpu-gate-${GLM53_BUILD_ATTEMPT:-attempt1}"
docker run --name "$gate" --gpus all -e CUDA_VISIBLE_DEVICES= \
  --entrypoint bash glm53-b12x-exl3:jovian-3aada677 /opt/verify_runtime.sh
for receipt in build-identity mtp-embedding-cpu-evidence mtp-fp8-scope-cpu-evidence b12x-cache-contract; do
  docker cp "$gate:/opt/$receipt.json" "$gate-$receipt.json"
done
