<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native backend plugins

See the [runtime guide](../../../../README.md) for model support and public APIs.

## Installation

Build the core runtime and plugins from the same source revision. The commands
below run from `packages/leapp-runtime-rust` and require CMake 3.24+ and a C++20
compiler. For the core build and library-loader setup, see
[core installation](../../../../README.md#installation).
SDKs and sources are build dependencies; deployment needs the installed libraries
and their runtime dependencies.

Set `LEAPP_NATIVE_BACKENDS` to the plugin installation directory (default:
`/opt/leapp/backends`):

```sh
export LEAPP_NATIVE_BACKENDS="$PWD/target/native-backends"
```

Install the libraries needed by your bundle at these relative paths:

| Backend | Path under `LEAPP_NATIVE_BACKENDS` |
| --- | --- |
| ONNX | `onnxruntime/libonnxruntime.so` |
| TorchScript | `torchscript/libleapp_torchscript.so` |
| NativeRT | `nativert/libleapp_nativert_backend.so` |
| TensorRT | `tensorrt/libleapp_tensorrt.so` |

### ONNX

Use ONNX Runtime supporting API 24, installed at the path above or selected with
`ORT_DYLIB_PATH`. The adapter is included in the Rust runtime and has no separate
plugin build. For GPU execution, install the CUDA provider libraries beside ONNX
Runtime and make their CUDA/cuDNN dependencies discoverable.

### TorchScript

Set `TORCH_ROOT` to a CUDA-enabled LibTorch SDK containing `include/` and `lib/`,
even for CPU execution. `TORCH_CXX11_ABI` defaults to `1`; match it to the SDK.
Use `CUDAToolkit_ROOT` for a nonstandard CUDA installation.

```sh
cmake -S src/engines/native/backends/torchscript -B target/native-torchscript \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/native-torchscript -j4
cmake --install target/native-torchscript
```

### NativeRT / PT2

Supply PyTorch v2.12.1 sources with submodules, LibTorch 2.12.1, and a matching
NativeRT library from the
[shared source-build recipe](../../triton/backends/README.md#pt2--nativert).
The [TorchScript SDK and ABI settings](#torchscript) also apply.

```sh
cmake -S src/engines/native/backends/nativert -B target/native-nativert \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DNATIVERT_LIBRARY=/path/to/libleapp_nativert.so \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/native-nativert -j4
cmake --install target/native-nativert
```

### TensorRT

Use TensorRT 10.16 or newer 10.x headers/libraries and CUDA.

```sh
cmake -S src/engines/native/backends/tensorrt -B target/tensorrt-backend \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS" \
  -DTENSORRT_ROOT=/path/to/TensorRT -DCUDA_ROOT=/usr/local/cuda
cmake --build target/tensorrt-backend -j
cmake --install target/tensorrt-backend
```

See [TensorRT model support](../../../../README.md#tensorrt-engines) for engine
compatibility and I/O limits.

### Warp operators

Build the [shared Warp adapters](../../triton/backends/README.md#warp-operators)
with ONNX Runtime headers and LibTorch matching the selected native backends.
Install and link the required adapters beside their backends:

```sh
cmake --install /path/to/warp-build --prefix "$PWD/target/warp-adapters"
mkdir -p "$LEAPP_NATIVE_BACKENDS/onnxruntime" "$LEAPP_NATIVE_BACKENDS/nativert"
ln -s "$PWD/target/warp-adapters/lib/libleapp_wrp_onnx_custom_op.so" \
  "$LEAPP_NATIVE_BACKENDS/onnxruntime/libleapp_wrp_onnx_custom_op.so"
ln -s "$PWD/target/warp-adapters/lib/libleapp_wrp_torch_custom_op.so" \
  "$LEAPP_NATIVE_BACKENDS/nativert/libleapp_wrp_torch_custom_op.so"
```

Use matching Warp export/runtime releases and add Warp's `bin/` directory to the
library search path. See [Warp model support](../../../../README.md#native-warp-support)
for execution requirements.

## Adding a backend

Implement `BackendPlugin` and register it in [`backends.rs`](../backends.rs).
The trait documents execution, ownership, and safety requirements. C++ plugins
can use the [shared bridge](bridge.rs).
