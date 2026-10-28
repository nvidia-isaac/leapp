<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native backend installation

Run these commands from `packages/leapp-runtime-rust` on Linux. The
[core runtime](../../../../README.md#installation) and backend plugins build
separately. Builds use existing
SDKs without installing them; the C++ plugins require CMake 3.24+ and a C++20
compiler. Rebuild all native backend libraries for the private v2 bridge ABI.

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

## ONNX

Install an ONNX Runtime library supporting API 24 at the path above, or set
`ORT_DYLIB_PATH` to its absolute path. This adapter is included in the Rust
runtime and has no separate plugin build.

For GPU execution, install the CUDA provider libraries beside ONNX Runtime and
make their CUDA/cuDNN dependencies discoverable. CUDA provider initialization
errors fail session loading. One ONNX Runtime library is retained per process;
changing its path after loading is unsupported.

An optional `libleapp_wrp_onnx_custom_op.so` beside the default ONNX library is
registered when present. See the
[shared adapter recipe](../../triton/backends/README.md#warp-operators)
to build it.

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
