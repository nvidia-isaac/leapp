<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native backend installation

Run these commands from `packages/leapp-runtime-rust` on Linux. The
[core runtime](../../../../README.md#installation) and ONNX Runtime are installed
separately. Builds use existing SDKs without installing them.

```sh
export LEAPP_NATIVE_BACKENDS="$PWD/target/native-backends"
```

The default backend directory is `/opt/leapp/backends`. Make each backend's
dependencies available through `LD_LIBRARY_PATH` or the system loader.

| Backend | Installed path relative to `LEAPP_NATIVE_BACKENDS` |
| --- | --- |
| ONNX | `onnxruntime/libonnxruntime.so` |

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
