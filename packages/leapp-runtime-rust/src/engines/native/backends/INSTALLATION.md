<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native backend installation

Run these commands from `packages/leapp-runtime-rust` on Linux. The
[core runtime](../../../../README.md#installation) and backend plugins build
separately. Builds use existing
SDKs without installing them; the C++ plugins require CMake 3.24+ and a C++20
compiler. Build the runtime and LEAPP plugins from the same source revision.

```sh
export LEAPP_NATIVE_BACKENDS="$PWD/target/native-backends"
```

The default backend directory is `/opt/leapp/backends`. Make each backend's
dependencies available through `LD_LIBRARY_PATH` or the system loader.

| Backend | Installed path relative to `LEAPP_NATIVE_BACKENDS` |
| --- | --- |
| ONNX | `onnxruntime/libonnxruntime.so` |
| TorchScript | `torchscript/libleapp_torchscript.so` |
| NativeRT | `nativert/libleapp_nativert_backend.so` |
| TensorRT | `tensorrt/libleapp_tensorrt.so` |

## ONNX

Install an ONNX Runtime library supporting API 24 at the path above, or set
`ORT_DYLIB_PATH` to its absolute path. This adapter is included in the Rust
runtime and has no separate plugin build.

For GPU execution, install the CUDA provider libraries beside ONNX Runtime and
make their CUDA/cuDNN dependencies discoverable. CUDA provider initialization
errors fail session loading. One ONNX Runtime library is retained per process;
changing its path after loading is unsupported.

## TorchScript

Supply `TORCH_ROOT` with the `include/` and `lib/` directories from a CUDA-enabled
LibTorch SDK, even for CPU execution. `TORCH_CXX11_ABI` defaults to `1`; override
it to match the SDK. CMake uses `CUDAToolkit_ROOT` to locate a nonstandard CUDA
installation. TorchScript does not require NativeRT sources or libraries.

```sh
cmake -S src/engines/native/backends/torchscript -B target/native-torchscript \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/native-torchscript -j4
cmake --install target/native-torchscript
```

## NativeRT / PT2

Supply matching PyTorch v2.12.1 sources with submodules, LibTorch 2.12.1, and a
NativeRT library built against that LibTorch distribution. The existing
[NativeRT source-build recipe](../../triton/backends/README.md#pt2--nativert)
can produce this library without Triton. The TorchScript SDK and ABI requirements
above also apply.

```sh
cmake -S src/engines/native/backends/nativert -B target/native-nativert \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DNATIVERT_LIBRARY=/path/to/libleapp_nativert.so \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/native-nativert -j4
cmake --install target/native-nativert
```

Make the LibTorch, NativeRT, and CUDA library directories available to the loader.

## TensorRT

Use TensorRT 10.16 or newer 10.x headers/libraries and CUDA.

```sh
cmake -S src/engines/native/backends/tensorrt -B target/tensorrt-backend \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS" \
  -DTENSORRT_ROOT=/path/to/TensorRT -DCUDA_ROOT=/usr/local/cuda
cmake --build target/tensorrt-backend -j
cmake --install target/tensorrt-backend
```

Build engine artifacts ahead of time for the deployed TensorRT version and GPU.
See [TensorRT model support](../../../../README.md#tensorrt-engines) for I/O limits.

## Warp operators

Build LEAPP's shared adapters using the
[Warp build recipe](../../triton/backends/README.md#warp-operators), with ONNX
Runtime headers and LibTorch matching the selected native backends. Install and
link the required adapters beside their backends:

```sh
cmake --install /path/to/warp-build --prefix "$PWD/target/warp-adapters"
mkdir -p "$LEAPP_NATIVE_BACKENDS/onnxruntime" "$LEAPP_NATIVE_BACKENDS/nativert"
ln -s "$PWD/target/warp-adapters/lib/libleapp_wrp_onnx_custom_op.so" \
  "$LEAPP_NATIVE_BACKENDS/onnxruntime/libleapp_wrp_onnx_custom_op.so"
ln -s "$PWD/target/warp-adapters/lib/libleapp_wrp_torch_custom_op.so" \
  "$LEAPP_NATIVE_BACKENDS/nativert/libleapp_wrp_torch_custom_op.so"
```

Keep Warp's export and runtime releases aligned. Make its `bin/`, LibTorch, and
CUDA library directories available to the loader. Ordinary models do not need
these adapters; an invalid installed adapter causes a load error for models
using that backend. See [Warp model support](../../../../README.md#native-warp-support)
for execution requirements.
