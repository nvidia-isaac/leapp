<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime distribution and backend environments

Cargo and PyPI distribute the core runtime. C++ applications integrate the source
through CMake. Pixi creates a locked environment containing the selected backend
libraries and, for Python consumers, the PyPI core. Both orchestration engines
remain in the core; `SessionOptions.engine` selects native or Triton at execution.
Adding a backend does not rebuild it. Backend recipes currently produce a
**local channel**; public backend hosting is separate work.

## Core release artifacts

Run with Python 3.12+, Rust, Maturin, and the workspace dependencies available:

```sh
cargo package --locked -p leapp-runtime
python packaging/build_sdist.py
```

`build_sdist.py` uses Maturin to include the core and shared binding sources. It
prunes the unrelated C++ workspace entries from the source archive's Cargo lock
file without changing any retained dependency version. The resulting archive
builds with `--locked`. The planned release wheel matrix is manylinux_2_28
for x86-64 and aarch64 and CPython 3.10–3.14, built from that archive.
The local SDK producer's `--auditwheel skip` path is only for local Conda packages.

The runtime version is independent of the exporter version. Keep the runtime's
Cargo manifests, Python metadata, CMake version, and lockfile aligned when
preparing a release, and keep the packaged license copies synchronized with the
repository license.

## Package selection

| Package | Contents |
|---|---|
| `leapp-runtime-environment` | Backend discovery configuration; no SDK libraries |
| `leapp-backend-onnx` | Native ONNX Runtime installation |
| `leapp-backend-torchscript` | Native TorchScript plugin |
| `leapp-backend-nativert` | NativeRT library and native plugin |
| `leapp-triton-{backend}` | Selected backend's Triton adapter and server dependencies |
| `leapp-onnx-tensorrt` | Optional ONNX Runtime TensorRT provider and builder dependencies |
| `leapp-warp-{engine}-{backend}` | Optional Warp adapter for the selected engine and backend |
| `leapp-runtime-python`, `leapp-runtime-cxx` | Optional local bindings, built with `--bindings`, for development before publication |

Shared dependencies are installed once. The `leapp-libs-*` packages are resolved
automatically. A core-only environment installs no model SDKs. A native CPU ONNX
channel needs neither Triton nor CUDA. Triton packages depend on the server;
CUDA libraries follow the selected SDK payloads. The current Torch plugins need
CUDA libraries even for CPU inference. Backend packages support
CPU and GPU execution where the backend does. Precompiled TensorRT engines require
Triton and a GPU. Runtime
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

For a published Python core, a core-only environment needs no backend channel:

```sh
pixi run consumer --output target/inspect-only
```

Given a channel built from the matching runtime release, create a consumer manifest:

```sh
pixi run consumer --channel target/packaging/channel \
  --output target/my-app --backends onnx --cuda
pixi install --manifest-path target/my-app/pixi.toml
pixi run --manifest-path target/my-app/pixi.toml python -c 'import lrt'
```

Omit `--cuda` for CPU deployment and use a CPU-only ONNX channel to avoid CUDA
dependencies. Use `--binding none` for Rust/C++ applications building their core
from source. Select `--engines triton` or `--engines native triton` when those
adapters are needed; native is the default. List multiple `--backends`, or omit
the channel and backends for bundle inspection.
`--warp onnx` adds Warp to an ONNX environment. Warp execution requires CUDA,
including for CPU model instances. A deployment machine needs a compatible host
OS and NVIDIA driver for GPU execution, but no build container or CUDA SDK.

Add `--onnx-tensorrt` to both the producer build and an ONNX CUDA consumer command
to include TensorRT acceleration. Configure selected nodes with
[`SessionOptions.onnx_tensorrt`](../README.md#onnx-tensorrt-acceleration).
Other consumers keep their existing backend dependencies.

The generated manifest pins the Python core version and local channel's exact build. Commit the consumer's
`pixi.toml` and generated `pixi.lock` with your application. To add another backend,
use the same version and build string shown in that manifest:

```sh
pixi add --manifest-path target/my-app/pixi.toml \
  'leapp-backend-torchscript ==0.1.0 <build-from-manifest>'
pixi remove --manifest-path target/my-app/pixi.toml leapp-backend-torchscript
```

An upgrade must update the core and selected plugins together. A PyPI consumer
requires its backend channel to have been built from the clean matching
`runtime-vVERSION` tag. Local development bindings and plugins share an exact
package build. Dependency-only Rust/C++ consumers must build the core from the
same revision recorded in `packages.json`; Cargo/CMake dependencies are managed
by the application.
The package build identifier comes from the source and SDK library checksums;
it is package metadata, not a new versioned runtime interface.

Activation selects the installed backend directories. Explicit
`LEAPP_NATIVE_BACKENDS`, `LEAPP_TRITON_BACKENDS`, `TRITON_ROOT`, `CUDA_ROOT`, and `ORT_DYLIB_PATH` overrides
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
  --backends onnx torchscript nativert tensorrt --engines native triton --cuda \
  --sdk-origin 'vendor-image@sha256:<verified-digest>'
pixi run package-runtime
```

For native CPU ONNX, supply only `ORT_DYLIB_PATH` pointing to its shared library
and the matching license files. This can be an independent ONNX Runtime SDK:

```sh
export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so
pixi run build-runtime --backends onnx --glibc 2.28 \
  --sdk-origin 'onnxruntime-artifact@sha256:<verified-digest>'
pixi run package-runtime
```

With no backends or explicit engines, the producer creates only discovery
configuration and requires no SDK. Before the first public release, or for
uncommitted development builds, add `--bindings python` to the producer command
and `--core-source channel` to the consumer command. Local C++ validation can use
`--bindings cxx` and `--core-source channel --binding cxx`; public C++ integration
uses CMake source builds. The local binding packages are not public release
artifacts.

The producer's `--onnx-tensorrt` option requires ONNX Runtime's TensorRT provider,
a compatible `TENSORRT_ROOT`, and `--cuda`. With Triton selected, the producer
packages the SDK's stock ONNX Runtime adapter.

Use the same command on each architecture with its compatible SDK. The glibc
value declares the deployment baseline, not an exact build-host version. Build
natively; this workflow does not cross-compile CUDA, PyTorch or TensorRT. NativeRT
is built from matching PyTorch sources against prebuilt LibTorch, using the existing
CMake target; it does not rebuild all of PyTorch. An incompatible or missing
NativeRT SDK is a build failure, not an omitted backend.

Plugin builds need the SDK's C++ compiler and compatible CUDA toolchain. Pixi
locks Rust, Python, Maturin, CMake, Ninja, LIEF and rattler-build for both
architectures. SDK acquisition remains explicit: supply an immutable vendor
artifact origin, preserve its licenses, and use matching headers and libraries.
`build.json` and package metadata record source and tool versions; every copied
SDK library has its origin and SHA-256 recorded.

LIEF sets the packaged libraries' relative lookup paths. Conda-forge's
rattler-build still installs patchelf transitively, but this workflow does not
invoke it. Neither relocation tool is included in consumer environments.

`--backends onnx tensorrt --engines native triton` selects producer work.
The `tensorrt` backend is packaged only for Triton. Backends and local bindings are
opt-in; `--engines native` is the default. To package Warp,
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

For a package smoke check, generate fixtures only for the selected backends in
the SDK build environment. Core-only checks need no model fixtures. TensorRT
plans must be generated on the target GPU.
Copy the fixtures and `tests/deployment/` into a clean runtime container, without
mounting the source SDK or producer environment. Set the existing fixture variables
from [AGENTS.md](../AGENTS.md#minimal-backend-installations).

Install one consumer at a time and run:

```sh
pixi run --manifest-path /app/pixi.toml python /tests/check_packages.py \
  --backends onnx --engines native --cuda --report /results/onnx.json
```

The audit executes the selected backends with the selected engines,
rejects build tools and unselected runtimes,
verifies package file hashes, and records loaded library paths and package sizes.
It retains loaded-library isolation checks and checks optional Warp
adapters where applicable. Missing-backend execution checks belong to the full
minimal-installation matrix in [AGENTS.md](../AGENTS.md#minimal-backend-installations);
the package smoke check uses its `--installed-only` mode.

For release validation, repeat for core only (omit `--backends`), each individual
backend, ONNX + TensorRT, and TorchScript + NativeRT. Compare the core hashes
between reports: they must remain identical. Also run the existing C++ installed
consumer and test a second installation prefix, backend add/remove, and rejection
of mismatched builds. The audit already exercises mixed ONNX/TensorRT graphs
and installed Warp adapters where selected. A clean consumer container, not merely a
restricted backend directory in a full SDK container, is required for isolation
claims. Run the SDK/GPU matrix on configured workers before releasing those
backend packages.
