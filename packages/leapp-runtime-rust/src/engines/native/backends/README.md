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

### ONNX

Use ONNX Runtime supporting API 24, installed at the path above or selected with
`ORT_DYLIB_PATH`. The adapter is included in the Rust runtime and has no separate
plugin build. For GPU execution, install the CUDA provider libraries beside ONNX
Runtime and make their CUDA/cuDNN dependencies discoverable.

For optional TensorRT acceleration, also install
`libonnxruntime_providers_tensorrt.so` beside ONNX Runtime, with matching TensorRT
runtime, parser, plugin, and builder-resource libraries. Configure
[`SessionOptions.onnx_tensorrt`](../../../../README.md#onnx-tensorrt-acceleration).

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

Register the backend in [`backends.rs`](../backends.rs). Rust implementations
implement its `BackendPlugin` trait; C++ implementations derive from
[`lrt::native::BackendPlugin`](include/leapp/native/backend_plugin.h).
The header documents each method's defaults, lifecycle, ownership, and threading
requirements. Only `new_session` and `run_dynamic` are required.

Public C++ headers are in [`include/leapp/native/`](include/leapp/native/).
Implement a `LoadBackend` factory and export it with `LEAPP_NATIVE_BACKEND(my_backend, Load)` from
`<leapp/native/export.h>`. Register the same name with the Rust bridge loader;
adding a backend name requires rebuilding the runtime registration table.

The built-in C++ implementation is in [`torch/plugin.cc`](torch/plugin.cc),
shared by TorchScript and NativeRT. Its adjacent `executor.h` / `executor.cc`
files own SDK execution resources.

### Build a C++ plugin

The standalone SDK requires CMake 3.24+ and C++20. Run from
`packages/leapp-runtime-rust`:

```sh
cmake -S src/engines/native/backends/cxx -B target/native-backend-sdk \
  -DCMAKE_INSTALL_PREFIX=/your/backend-sdk -DLEAPP_NATIVE_BACKEND_TESTS=OFF
cmake --install target/native-backend-sdk
```

Set `CMAKE_PREFIX_PATH` to that prefix and consume the SDK:

```cmake
find_package(leapp_native_backend CONFIG REQUIRED)
add_library(my_backend SHARED plugin.cc)
target_link_libraries(my_backend PRIVATE lrt::native_backend)
```

Source builds can use `add_subdirectory` on `src/engines/native/backends/cxx`
and link the same target. Add the implementation's model SDK dependencies.
The adapter compiles with the plugin's C++ ABI settings, including LibTorch's
`_GLIBCXX_USE_CXX11_ABI`. Build plugins from the matching runtime revision;
the loader rejects incompatible or incomplete libraries with rebuild instructions.

### Validate a C++ backend

The single CPU fixture under `cxx/tests/` exercises the Rust/C++ boundary.
Run from `packages/leapp-runtime-rust`:

```sh
cmake -S src/engines/native/backends/cxx -B target/native-backend-tests \
  -DCMAKE_INSTALL_PREFIX=/your/backend-sdk
cmake --build target/native-backend-tests -j4
ctest --test-dir target/native-backend-tests --output-on-failure
LEAPP_NATIVE_TEST_PLUGINS="$PWD/target/native-backend-tests/plugins" \
  cargo test --locked --lib engines::native::backends::bridge -- --test-threads=1
```

Reuse the same fixture against the installed SDK:

```sh
cmake --install target/native-backend-tests
cmake -S src/engines/native/backends/cxx/tests -B target/installed-backend-tests \
  -DCMAKE_PREFIX_PATH=/your/backend-sdk
cmake --build target/installed-backend-tests -j4
ctest --test-dir target/installed-backend-tests --output-on-failure
LEAPP_NATIVE_TEST_PLUGINS="$PWD/target/installed-backend-tests/plugins" \
  cargo test --locked --lib engines::native::backends::bridge -- --test-threads=1
```

Use the [runtime contributor guide](../../../../AGENTS.md) for the existing
model SDK and CPU/CUDA suites. Format C++ with the repository's Google C++20
`.clang-format` configuration.
