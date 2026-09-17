#!/usr/bin/env bash
set -euo pipefail
# Runtime-derived bases retain NVRTC13 but may omit the development linker
# symlink. Point CMake at that same existing library, never another CUDA build.
if [[ ! -e "$CUDA_HOME/lib64/libnvrtc.so" ]]; then
  test -f "$CUDA_HOME/lib64/libnvrtc.so.13"
  ln -s libnvrtc.so.13 "$CUDA_HOME/lib64/libnvrtc.so"
fi
# A failed configure can cache the old NOTFOUND value in CUDA_NVRTC_LIB even
# after FindCUDA locates the repaired linker entry. Invalidate only that cache.
for cache in /opt/src/vllm/build/*/CMakeCache.txt; do
  if [[ -f "$cache" ]] && grep -q '^CUDA_NVRTC_LIB:.*=CUDA_nvrtc_LIBRARY-NOTFOUND$' "$cache"; then
    rm "$cache"
  fi
done
python3 /patches/build-patches/repair_build_paths.py --root /opt/src/vllm
mkdir -p /opt/wheels
python3 /opt/verify_bootstrap_exllama.py
cd /opt/src/vllm
if [[ "${GLM53_REUSE_BUILT_VLLM:-0}" != 1 ]]; then
  uv build --no-build-isolation --wheel --out-dir /opt/wheels
fi
python3 - <<'PYWHEEL'
import hashlib,json,pathlib,subprocess,zipfile
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == '3aada67721bfdb8b98355207ca3780bb32e6f434'
files=list(pathlib.Path('/opt/wheels').glob('vllm-*.whl'));assert len(files)==1
path=files[0]
with zipfile.ZipFile(path) as z:
    assert z.testzip() is None
    meta=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
    assert '3aada677' in z.read(meta).decode()
    assert any('_C_stable_libtorch' in n and n.endswith('.so') for n in z.namelist())
pathlib.Path('/opt/vllm-wheel-build.json').write_text(json.dumps({'state':'BUILT_WHEEL_INTEGRITY_VERIFIED','file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_revision':'3aada67721bfdb8b98355207ca3780bb32e6f434'},indent=2)+'\n')
PYWHEEL
cd /opt/src/b12x
uv build --no-build-isolation --wheel --out-dir /opt/wheels
uv pip uninstall vllm
uv pip install /opt/wheels/vllm-*.whl --override /opt/dependency-overrides.txt
# Retain the exact verified public-bootstrap ExLlama GPU extension and package.
uv pip install /opt/wheels/b12x-*.whl --override /opt/dependency-overrides.txt
cd /
python3 /opt/verify_bootstrap_exllama.py
python3 - <<'PY'
import importlib.metadata as m,json,pathlib,torch
assert torch.__version__=='2.13.0+cu130'
assert not torch.cuda.is_initialized()
pathlib.Path('/opt/compile-complete.json').write_text(json.dumps({
    'state':'SOURCE_WHEELS_BUILT_AND_INSTALLED_FINAL_PATCHES_PENDING',
    'packages':{n:m.version(n) for n in ['torch','vllm','exllamav3','b12x']}
},indent=2)+'\n')
PY
