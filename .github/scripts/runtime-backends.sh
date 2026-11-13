#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inside the pinned Triton SDK image, from packages/leapp-runtime-rust.
set -euo pipefail
backend=$1
engines=(native triton)
if [[ "$backend" == tensorrt ]]; then
  engines=(triton)
fi
work=target/backend-ci
unset PYTHONPATH
git config --global --add safe.directory "$(realpath ../..)"

apt-get update
apt-get install -y --no-install-recommends nlohmann-json3-dev
pixi install --locked
pixi run --locked python -m venv "$work/sdk"
python="$PWD/$work/sdk/bin/python"
"$python" -m pip install --report "$work/fixtures-sdk.json" onnx==1.19.0

export TRITON_ROOT=/opt/tritonserver CUDA_ROOT=/usr/local/cuda TENSORRT_ROOT=/usr
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:/opt/tritonserver/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
case "$backend" in
  onnx)
    export LD_LIBRARY_PATH="/opt/tritonserver/backends/onnxruntime:$LD_LIBRARY_PATH"
    "$python" tests/generators/generate_models.py --output "$work/fixtures"
    ;;
  torchscript)
    "$python" -m pip install --report "$work/torch-sdk.json" \
      --index-url https://download.pytorch.org/whl/cu132 'torch==2.12.1+cu132'
    export TORCH_ROOT
    TORCH_ROOT=$("$python" -c 'import torch; print(torch.__path__[0])')
    export LD_LIBRARY_PATH="$TORCH_ROOT/lib:$LD_LIBRARY_PATH"
    "$python" tests/generators/generate_torchscript_models.py --output "$work/fixtures"
    ;;
  tensorrt)
    "$python" tests/generators/generate_tensorrt_models.py --output "$work/fixtures" \
      --trtexec /usr/src/tensorrt/bin/trtexec
    ;;
  *) exit 2 ;;
esac

# Keep immutable origins for the SDK image and additional wheels in provenance.
sdk_origin=$("$python" - <<'PY'
import json, os
from pathlib import Path

origins = [os.environ["SDK_IMAGE"]]
for report in Path("target/backend-ci").glob("*-sdk.json"):
    for item in json.loads(report.read_text())["install"]:
        artifact = item["download_info"]
        origins.append(artifact["url"] + "#sha256=" + artifact["archive_info"]["hashes"]["sha256"])
print(";".join(origins))
PY
)
pixi run --locked build-runtime --output "$work/packages" --glibc 2.39 \
  --backends "$backend" --engines "${engines[@]}" --cuda --sdk-origin "$sdk_origin"
pixi run --locked package-runtime --input "$work/packages"

# Each consumer installs one engine and one backend. The core wheel is shared
# with the SDK-free packaging job and is never rebuilt in this SDK environment.
for engine in "${engines[@]}"; do
  pixi run --locked consumer --channel "$work/packages/channel" --core-source channel \
    --binding none --output "$work/$engine" --backends "$backend" --engines "$engine" --cuda
done
pixi run --locked python - "${engines[@]}" <<'PY'
from pathlib import Path
import sys

root = Path("target/backend-ci")
wheel = next((root / "core").glob("*.whl"))
for engine in sys.argv[1:]:
    manifest = root / engine / "pixi.toml"
    text = manifest.read_text().replace("[dependencies]\n", '[dependencies]\npython = "3.12.*"\n')
    text += 'CUDA_PATH = "$CONDA_PREFIX/lib/leapp/cuda"\n'
    text += '\n[pypi-dependencies]\nleapp-runtime = { path = "/wheels/' + wheel.name + '" }\n'
    text += 'cupy-cuda13x = "==14.2.0"\n'
    manifest.write_text(text)
PY
