<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pixi binary packages

Pixi creates a locked environment containing the runtime and only the backend
packages your application selects. Both engines are included; continue using
`SessionOptions.engine` to select native or Triton. Adding a backend does not
rebuild the runtime. These recipes currently produce a **local channel**;
there is no public package service or upload step.

## Package selection

| Package | Contents |
|---|---|
| `leapp-runtime-python` | `lrt` extension for the build's Python version and NumPy dependency |
| `leapp-runtime-cxx` | Shared library, public headers, `lrt::runtime` CMake target |
| `leapp-backend-onnx` | ONNX Runtime and its Triton adapter |
| `leapp-backend-torchscript` | Native and Triton TorchScript plugins |
| `leapp-backend-nativert` | Custom NativeRT library and both plugins |
| `leapp-backend-tensorrt` | TensorRT libraries and both plugins |
| `leapp-warp-onnx`, `leapp-warp-nativert` | Optional Warp adapters for the selected backend |

Shared dependencies are installed once. The `leapp-libs-*` packages are resolved
automatically; applications do not need to list them. The current Pixi packages
install Triton and CUDA runtime libraries with the common runtime; the runtime
loads them only when needed. Backend packages support
CPU and GPU execution where the backend does; TensorRT requires a GPU. Runtime
installations contain no compiler, exporter, model SDK headers, or build tools, except
the public LEAPP/DLPack headers in the C++ package.

Builds target the native Linux architecture (`linux-64` or `linux-aarch64`). SDK
versions are inspected and recorded; the runtime and adapters enforce their
compatibility requirements. Python and build tools are pinned in `pixi.toml` and
`pixi.lock`. The Python package requires the interpreter's major/minor version;
it is not an `abi3` wheel.

The producer declares a minimum glibc version with `--glibc`; packaging rejects
binaries for another architecture or requiring newer glibc symbols. This check
does not replace testing on the intended deployment OS. Builds on Jetson also
record the detected L4T release. Create their consumer environments on that release;
the generator and deployment audit reject mismatches. Pixi does not detect L4T
by default, so moving a manifest and running `pixi install` alone does not validate
Jetson compatibility. ARM server and Jetson packages are not interchangeable merely
because both use aarch64.

## Install an application environment

Given a channel built as described below, create a consumer manifest:

```sh
pixi run consumer --channel target/packaging/channel \
  --output target/my-app --backends onnx --cuda
pixi install --manifest-path target/my-app/pixi.toml
pixi run --manifest-path target/my-app/pixi.toml python -c 'import lrt'
```

Omit `--cuda` for CPU deployment. Use `--binding cxx` or `--binding both` when
needed, list multiple `--backends`, or omit backends for bundle inspection.
`--warp onnx` adds Warp to an ONNX environment. Warp execution requires CUDA,
including for CPU model instances. A deployment machine needs a compatible host
OS and NVIDIA driver for GPU execution, but no build container or CUDA SDK.

The generated manifest pins the local channel's exact build. Commit the consumer's
`pixi.toml` and generated `pixi.lock` with your application. To add another backend,
use the same version and build string shown in that manifest:

```sh
pixi add --manifest-path target/my-app/pixi.toml \
  'leapp-backend-torchscript ==0.1.0 <build-from-manifest>'
pixi remove --manifest-path target/my-app/pixi.toml leapp-backend-torchscript
```

An upgrade must update the binding and selected plugins together. Exact package
dependencies reject incompatible combinations instead of mixing plugin ABIs.
The package build identifier comes from the source and SDK library checksums;
it is package metadata, not a new versioned runtime interface.

Activation selects the installed backend directories. Explicit
`LEAPP_NATIVE_BACKENDS`, `LEAPP_TRITON_BACKENDS`, and `ORT_DYLIB_PATH` overrides
retain their existing behavior. Libraries use relative ELF search paths, so
installation does not need `LD_LIBRARY_PATH`. Do not inherit an SDK's
`LD_LIBRARY_PATH` when testing or deploying an isolated environment.

## Build a local channel

Run on the target architecture in a trusted, compatible SDK environment. Install
[Pixi 0.84 or newer](https://pixi.prefix.dev/latest/installation/), then run from
`packages/leapp-runtime-rust`. The producer environment in `pixi.toml` is separate
from every generated consumer environment.

```sh
pixi install --locked
export TRITON_ROOT=/opt/tritonserver
export CUDA_ROOT=/usr/local/cuda
export TORCH_ROOT=/path/to/torch
export PYTORCH_SOURCE=/path/to/pytorch-v2.12.1-with-submodules
export TENSORRT_ROOT=/path/to/TensorRT
# Make these SDK libraries resolvable for builds and dependency inspection.
export LD_LIBRARY_PATH=/path/to/sdk/libraries

pixi run build-runtime --glibc 2.39 \
  --sdk-origin 'vendor-image@sha256:<verified-digest>'
pixi run package-runtime
```

Use the same command on each architecture with its compatible SDK. The glibc
value declares the deployment baseline, not an exact build-host version. Build
natively; this workflow does not cross-compile CUDA, PyTorch or TensorRT. NativeRT
is built from matching PyTorch sources against prebuilt LibTorch, using the existing
CMake target; it does not rebuild all of PyTorch. An incompatible or missing
NativeRT SDK is a build failure, not an omitted backend.

The producer needs the SDK's C++ compiler and compatible CUDA toolchain. Pixi
locks Rust, Python, Maturin, CMake, Ninja, LIEF and rattler-build for both
architectures. SDK acquisition remains explicit: supply an immutable vendor
artifact origin, preserve its licenses, and use matching headers and libraries.
`build.json` and package metadata record source and tool versions; every copied
SDK library has its origin and SHA-256 recorded.

LIEF sets the packaged libraries' relative lookup paths. Conda-forge's
rattler-build still installs patchelf transitively, but this workflow does not
invoke it. Neither relocation tool is included in consumer environments.

`--backends onnx tensorrt` can restrict producer work. With no option, all four
backends build. `--backends` with no values builds only the core. To package Warp,
build the existing [Warp adapters](../src/engines/triton/backends/README.md#warp-operators)
against the same SDK, then pass `--warp /path/to/adapter-libraries`. Each adapter
is packaged separately and only depends on its own model backend.

Outputs are under ignored `target/packaging/`:

- `stage/`: original compiled outputs, preserved across packaging attempts.
- `payloads/`: relocated package files, licenses and provenance.
- `channel/`: `.conda` packages, repository metadata and `packages.json` with
  compressed sizes, installed sizes and exact dependency specifications.
- `recipes/`: generated recipes for the shared recipe template.

Use a new `--output` directory for another source build and pass that directory
to `package-runtime --input`. Packaging inspects trusted SDK binaries with `ldd`,
resolves their transitive dependencies, and includes known dynamically loaded
ORT/cuDNN/NVRTC libraries. Missing libraries or licenses fail packaging. Driver
libraries and glibc stay on the host. Vendor libraries live under
`lib/leapp/vendor`; they cannot overwrite Conda's Python dependencies.

No public redistribution is implied by producing a local package. Vendor license
notices travel with each payload; public channel hosting and release policy are
separate work.

## Validate deployment

Generate the existing ONNX, TorchScript, NativeRT and TensorRT fixtures in the SDK
build environment first. TensorRT plans must be generated on the target GPU.
Copy the fixtures and `tests/deployment/` into a clean runtime container, without
mounting the source SDK or producer environment. Set the existing fixture variables
from [AGENTS.md](../AGENTS.md#minimal-backend-installations).

Install one consumer at a time and run:

```sh
pixi run --manifest-path /app/pixi.toml python /tests/check_packages.py \
  --backends onnx --cuda --report /results/onnx.json
```

Repeat for core only (omit `--backends`), each individual backend,
ONNX + TensorRT, and TorchScript + NativeRT. The audit executes both engines,
checks missing-backend errors, rejects build tools and unselected runtimes,
verifies package file hashes, and records loaded library paths and package sizes.
Compare the core hashes between reports: they must remain identical.

Also run the existing C++ installed consumer, mixed ONNX/TensorRT graph, and Warp
checks where selected. Test a second installation prefix, backend add/remove,
and rejection of mismatched builds. A clean consumer container, not merely a
restricted backend directory in a full SDK container, is required for isolation
claims. The tasks can run on native CI workers once those workers exist; no
unconfigured runner or publishing workflow is added here.
