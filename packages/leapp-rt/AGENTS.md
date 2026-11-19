<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime contributor guide

Keep consumer installation, public APIs, and ownership contracts in
`../../docs/source/runtime/`. Keep internal build and validation recipes here,
and put implementation-specific contracts beside the code.
Use `name.rs` beside `name/` directories, rather than `mod.rs`.

## Layout

- `src/bundle/`: immutable bundle inspection and validation.
- `src/session.rs` and `src/session/`: public execution API, graph routing,
  initial values, feedback, and caller-provided outputs.
- `src/engines.rs`: private engine contract; native is selected by default.
- `src/engines/memory*`: reusable CPU, pinned-host, and CUDA allocations, copies,
  DLPack validation, and output allocation. Keep engine SDK types out of this code.
- `src/engines/triton/`: conversion, model repositories, native requests, and
  Triton-specific datatype and memory tags.
- `src/dlpack/`: ABI, leases, and ownership guards.
- `bindings/common/`: shared lifecycle and locking; Python and C++ adapters
  retain input owners and translate public types.

Tests follow `tests/unit_tests/` and `tests/functional_tests/`. Private unit tests
are included by their implementation modules. Binding tests stay with each binding.

## Updating Triton

Keep the SDK headers, supported server release, and bundled model configuration
schema aligned. The C API major/minor check alone does not identify a Triton
release.

1. Obtain the target Triton SDK and compatible backend libraries. Replace
   `src/engines/triton/convert/model_config.proto` with the schema from the
   matching `triton-inference-server/common` release tag. Rebuild any custom
   backend plugins against the target SDK.
2. Update `TRITON_VERSION` and the release-specific comments and error message in
   `src/engines/triton/native.rs`, plus the generated-release label in
   `src/engines/triton/native/generate_bindings.py`. Update the supported-release
   references in this file and the runtime documentation, and check build/CI
   configuration for old release pins.
3. Set `TRITON_ROOT` to the target SDK and regenerate the C API bindings from
   this package directory:

   ```sh
   python src/engines/triton/native/generate_bindings.py \
     "$TRITON_ROOT/include/triton/core/tritonserver.h"
   rustfmt --edition 2021 src/engines/triton/native/sys.rs
   ```

   Review the generated `sys.rs` diff against the new header; API version
   constants come from the generator. Cargo regenerates the Rust protobuf code
   from `model_config.proto` during the build.
4. Review callback signatures, request/response lifetimes, allocator behavior,
   and error ownership against the target release. In particular, recheck the
   release-specific `TRITONSERVER_InferenceResponseError` ownership assumption in
   `src/engines/triton/native/request.rs` against Triton's implementation.
5. Run the validation below on Linux with the new SDK and backend libraries,
   including CPU/CUDA execution and the applicable ignored backend tests.
   Rebuild/test the language bindings and the installed C++ consumer as below.
   Type checking alone does not validate an SDK upgrade.

## Validation

Core builds and checks require no Triton or CUDA SDK. Run from this package:

```sh
cargo check --locked --workspace --all-targets
cargo deny --locked --workspace check licenses advisories
python packaging/check_release.py
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --test bundle --test dlpack
cargo test --locked -p leapp-rt-bindings --test binding_lifecycle
cargo test --locked --lib engines::memory:: -- --test-threads=1
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
```

Build source archives with `packaging/build_sdist.py` (Python 3.12+) so their
lockfile matches Maturin's reduced workspace.
Runtime versions remain independent of the exporter; update the manifests,
CMake version and Cargo lock together when preparing a requested release.

For execution coverage, install the selected runtimes and generate fixtures:

```sh
python tests/generators/generate_models.py --output target/triton-fixtures
LEAPP_TEST_ENGINE=triton LEAPP_TEST_CUDA=1 cargo test --locked --workspace -- --test-threads=1
LEAPP_TEST_ENGINE=native LEAPP_TEST_CUDA=1 cargo test --locked --workspace -- --test-threads=1
```

These suites require Triton 2.69.0, CUDA, ONNX Runtime, and a GPU. Configure
`TRITON_ROOT`, `CUDA_ROOT`, `LEAPP_TRITON_BACKENDS`, and library search paths for
the installation. `LEAPP_TRITON_FIXTURES` selects another fixture directory.
CUDA memory tests run only with `LEAPP_TEST_CUDA=1`; native Triton callback tests
run only with `LEAPP_TEST_ENGINE=triton`. Explicitly enabled tests must fail
when their required libraries or devices are unavailable.

Optional backend fixtures use `generate_pt2_models.py`,
`generate_torchscript_models.py`, and `generate_tensorrt_models.py` under
`tests/generators/`, each with `--output`. TensorRT fixtures are for Triton only.
Its generator accepts `--trtexec /usr/bin/trtexec`; build its fixtures on the
target GPU.
Set `LEAPP_PT2_FIXTURES`, `LEAPP_TORCHSCRIPT_FIXTURES`, or
`LEAPP_TENSORRT_FIXTURES` and run the matching ignored integration tests:

```sh
cargo test --locked public_pt2 -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::nativert -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::torchscript -- --ignored --test-threads=1
cargo test --locked engines::tests::tensorrt -- --ignored --test-threads=1
```

For Warp, install the adapters as described in
`../../docs/source/runtime/installation.rst`.
Generate `tests/generators/generate_warp_models.py` fixtures, set
`LEAPP_WARP_FIXTURES` and an empty `WARP_APIC_TMPDIR`, then run
`LEAPP_TEST_ENGINE=triton cargo test --locked engines::tests::warp -- --ignored --test-threads=1`.
Repeat with `LEAPP_TEST_ENGINE=native` to exercise native execution.
This covers ONNX/PT2 CPU and GPU instances, CPU/pinned/CUDA output placement,
changed inputs, retained results, peer sessions, and unload.
Warp replay requires CUDA even with CPU model instances.

Run the binding validation sections below, including the installed C++ consumer.
Output changes must preserve retained owners, byte offsets, failure cleanup,
feedback isolation, and CPU/pinned/CUDA behavior.

### Minimal backend installations

Use isolated environments where unrelated model SDKs and libraries are absent,
not just excluded from the backend directory. Build the core runtime and bindings
without Triton or CUDA SDKs; reuse those exact binaries for CPU-only ONNX,
GPU ONNX, TensorRT-only, LibTorch-only, and LibTorch/NativeRT installations.
Native execution must work without Triton. CPU-only ONNX must work without CUDA,
including after failed requests for Triton, GPU execution, and pinned allocation.
Build each custom plugin inside its corresponding environment. Warp is optional
and should be omitted from this matrix.

Generate all four fixture sets before isolation and set `LEAPP_TRITON_FIXTURES`,
`LEAPP_TORCHSCRIPT_FIXTURES`, `LEAPP_PT2_FIXTURES`, and `LEAPP_TENSORRT_FIXTURES`.
With the Python binding and NumPy installed (plus CuPy for native CUDA inputs),
run for each engine:

```sh
python tests/deployment/check_installation.py --backend onnx --engine native --cuda
python tests/deployment/check_installation.py --backend onnx --engine triton --cuda
python tests/deployment/check_installation.py --backend onnx --engine native --without-triton --without-cuda
```

Replace `onnx` with `torchscript`, `nativert`, or `tensorrt` in their installations,
or `none` to check startup, inspection, and missing-backend errors without any
model runtime. TensorRT installations require `--engine triton --cuda`.
`--cuda` adds GPU execution for the other backends. The ONNX check includes
convolution to exercise cuDNN on CUDA. Package smoke tests can use
`--installed-only` with fixtures for just the selected backends; the full matrix
should omit it to retain missing-backend checks. Also run the applicable
Rust integration tests and the installed C++ SDK checks. Audit
shared-library dependencies, loaded-library maps, and binary hashes to confirm
isolation and reuse. Use separate processes for loader cases: successful library
loads remain cached for the process lifetime. The C++ runtime and CPU tests need
no SDKs; `LEAPP_TEST_CUDA=ON` requires the CUDA SDK for its GPU test code only.


## Internal engine and loader contracts

Keep the engine distinction out of the consumer documentation for now, but
preserve it when maintaining the implementation. Both engines ship in the same
build; installing a supported backend does not require Cargo feature selection
or exporter packages. ONNX ports match by name. Torch positional arguments follow
bundle port order. Native PT2 can place graph/weights on the selected device;
Triton PT2 requires archive devices matching its model instance.

Prebuilt `.plan`/`.engine` TensorRT artifacts are Triton-only and require CUDA,
a compatible GPU/TensorRT version, and matching port names, shapes, and dtypes.
The adapter uses optimization profile 0; shape tensors, additional profiles,
and external custom-plugin configuration are not exposed. Native caller-owned
outputs can bind directly; Triton copies ensemble results into supplied buffers
and still requires intermediate output storage. Peers have independent feedback
and workspaces but can share mutable model state; guard Torch and Warp execution.

Set loader overrides before process startup. `TRITON_ROOT` and `CUDA_ROOT`
select explicit installations; otherwise lookup uses the system loader followed
by `/opt/tritonserver` and `/usr/local/cuda`. `LEAPP_TRITON_BACKENDS` defaults to
`$TRITON_ROOT/backends` or `/opt/tritonserver/backends`. Libraries stay loaded for
the process lifetime; missing/invalid installations must report the library and
cause rather than silently selecting another engine or device. Native CPU ONNX
with CPU I/O needs no CUDA, whereas the current Torch plugins, TensorRT, and Warp
require their CUDA libraries. Engine libraries load only when selected; CUDA
loads for GPU work, transfers, pinned allocations, or supplied streams.

## DLPack ABI maintenance

See `src/dlpack.rs` and the wrapper methods for ownership contracts.
Keep the pinned upstream v1.0 header at
`src/dlpack/include/dlpack/dlpack.h` unmodified:
https://github.com/dmlc/dlpack/blob/v1.0/include/dlpack/dlpack.h

Regenerate checked-in Rust ABI declarations with Python, clang, and rustfmt:

```sh
python3 src/dlpack/generate_bindings.py
cargo test --locked --test dlpack
```

The tests compare Rust layouts with the actual C header, then exercise deletion,
ownership transfer, legacy/versioned exports, partial failure, and retained metadata.
On upgrades, update the pinned header and generator together, review new enum/flag
semantics, adjust the protocol version actually supported, and repeat Python/C++
interoperability and CUDA tests. Normal builds do not run the generator.

## Native backend plugins

Build plugins from the same source revision as the core, using CMake 3.24+
and C++20. Follow [Model Dependencies](../../docs/source/runtime/dependencies.rst)
for ONNX, TorchScript, NativeRT, and Warp build/install commands and SDK settings.
ONNX requires API 24; it is built into the core and needs no separate plugin.
SDKs and sources are build dependencies; deployment needs the installed libraries.

The backend library paths and `LEAPP_NATIVE_BACKENDS` override are listed in
[Installation](../../docs/source/runtime/installation.rst).

### Manual plugin builds

Use these CMake recipes when developing plugins outside the package producer.
Set `LEAPP_NATIVE_BACKENDS` to the install prefix and supply matching SDKs as in
[Model Dependencies](../../docs/source/runtime/dependencies.rst). `TORCH_CXX11_ABI`
defaults to `1`; match LibTorch. `CUDAToolkit_ROOT` selects a nonstandard toolkit.

#### TorchScript

```sh
cmake -S src/engines/native/backends/torchscript -B target/torchscript \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/torchscript -j4
cmake --install target/torchscript
```

#### PT2

```sh
cmake -S src/engines/triton/backends/torch/nativert -B target/nativert-source \
  -DCMAKE_BUILD_TYPE=Release -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DTORCH_ROOT="$TORCH_ROOT"
cmake --build target/nativert-source -j4
cmake -S src/engines/native/backends/nativert -B target/pt2 \
  -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
  -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DNATIVERT_LIBRARY="$PWD/target/nativert-source/libleapp_nativert.so" \
  -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
cmake --build target/pt2 -j4
cmake --install target/pt2
```

#### Warp adapters

Build both adapters against the deployed ONNX Runtime and LibTorch, then install
and link them beside their native backends:

```sh
leapp-build-warp-runtime --build-dir <warp-build-dir> -- \
  -DLEAPP_WARP_ONNXRUNTIME_VERSION=<installed-onnxruntime-version> \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --install <warp-build-dir> --prefix <warp-adapter-prefix>
mkdir -p "$LEAPP_NATIVE_BACKENDS/onnxruntime" "$LEAPP_NATIVE_BACKENDS/nativert"
ln -s <warp-adapter-prefix>/lib/libleapp_wrp_onnx_custom_op.so \
  "$LEAPP_NATIVE_BACKENDS/onnxruntime/libleapp_wrp_onnx_custom_op.so"
ln -s <warp-adapter-prefix>/lib/libleapp_wrp_torch_custom_op.so \
  "$LEAPP_NATIVE_BACKENDS/nativert/libleapp_wrp_torch_custom_op.so"
```

### Adding a backend

Register the backend in [`src/engines/native/backends.rs`](src/engines/native/backends.rs). Rust implementations
implement its `BackendPlugin` trait; C++ implementations derive from
[`lrt::native::BackendPlugin`](src/engines/native/backends/include/leapp/native/backend_plugin.h).
The header and module comments describe required methods, registration, ABI,
ownership, and threading. Use `torch/plugin.cc` and its adjacent executor files
as the shared TorchScript/NativeRT implementation example.

#### Build a C++ plugin

The standalone SDK requires CMake 3.24+ and C++20. Run from
the runtime package directory:

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

#### Validate a C++ backend

The single CPU fixture under `cxx/tests/` exercises the Rust/C++ boundary.
Run from the runtime package directory:

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

Use the validation sections above for the existing
model SDK and CPU/CUDA suites. Format C++ with the repository's Google C++20
`.clang-format` configuration.

## Triton backend plugins

The Torch plugins execute PyTorch artifacts through Triton's backend API.
They build independently of Cargo and install beneath a backend directory selected by
the `LEAPP_TRITON_BACKENDS` environment variable. Applications use the same
`lrt::Session` API as ONNX bundles; see the [runtime guide](../../docs/source/runtime/installation.rst).

Torch build commands below run from the runtime package directory. Supply Linux,
CMake 3.24+, a C++20 compiler, nlohmann_json 3.11+, Triton 2.69.0, CUDA, and
CUDA-enabled LibTorch with the C++11 libstdc++ ABI. CUDA-enabled LibTorch is
required even for CPU execution. Builds use installed dependencies without
fetching or installing them. Set `TRITON_ROOT` and `TORCH_ROOT` to their SDK roots.

### ONNX Runtime

Use Triton's stock `backends/onnxruntime/libtriton_onnxruntime.so` with its
matching ORT libraries. No additional ONNX backend build is required.
See [Session Options](../../docs/source/runtime/session_options.rst) for
per-node TensorRT acceleration.

### PT2 / NativeRT

The PT2 instance source documents raw-archive I/O and device placement.
Models use `max_batch_size: 0` and coupled responses. Preserve input-mutation
isolation, CPU/pinned/CUDA response buffers independent of model placement, and
synchronous completion before request release, including failures.

Supply a NativeRT implementation matching LibTorch. Configuration checks its
headers and linkage. The source-build recipe in `torch/nativert` supports
PyTorch v2.12.1 source with submodules and LibTorch 2.12.1:

```sh
cmake -S src/engines/triton/backends/torch/nativert -B target/nativert-source \
  -DCMAKE_BUILD_TYPE=Release -DPYTORCH_SOURCE=/path/to/pytorch-v2.12.1 \
  -DTORCH_ROOT="$TORCH_ROOT"
cmake --build target/nativert-source -j4
```

This builds the raw-program dispatcher path without Python bindings, optional
static kernels requiring private ATen symbols, or CUDA Triton/AOTInductor
delegates. Pass the resulting library to the PT2 plugin build:

```sh
cmake -S src/engines/triton/backends/torch/pt2 -B target/pt2-backend \
  -DCMAKE_BUILD_TYPE=Release -DTRITON_ROOT="$TRITON_ROOT" \
  -DTORCH_ROOT="$TORCH_ROOT" \
  -DNATIVERT_LIBRARY="$PWD/target/nativert-source/libleapp_nativert.so"
cmake --build target/pt2-backend -j4
cmake --install target/pt2-backend --prefix "$PWD/target/plugins"
```

The installed library is
`target/plugins/backends/leapp_pt2/libtriton_leapp_pt2.so`. Make LibTorch,
NativeRT, Triton, CUDA, and backend dependencies discoverable through the dynamic
loader, for example with `LD_LIBRARY_PATH`.

For integration tests, also make the installed ONNX Runtime backend available:

```sh
mkdir -p target/plugins/backends/onnxruntime
ln -s "$TRITON_ROOT/backends/onnxruntime/"* target/plugins/backends/onnxruntime/
export LEAPP_TRITON_BACKENDS="$PWD/target/plugins/backends"
python tests/generators/generate_models.py --output target/triton-fixtures
python tests/generators/generate_pt2_models.py --output target/pt2-fixtures
cargo test --locked engines::triton::tests::nativert -- --ignored --test-threads=1
cargo test --locked public_pt2 -- --ignored --test-threads=1
```

Generate fixtures with matching PyTorch and an available CUDA GPU. The optional
checks cover weights, noncontiguous outputs, input mutation, concurrent requests,
dynamic and empty tensors, error recovery, integer tensors, incompatible archive
devices, and ONNX/PT2 ensembles. `LEAPP_TRITON_FIXTURES` and `LEAPP_PT2_FIXTURES`
override fixture directories.

### Warp operators

Build LEAPP's native Warp adapters with PyTorch matching NativeRT. Select the
ONNX Runtime header version shipped by Triton's ONNX backend, for example:

```sh
leapp-build-warp-runtime --build-dir /path/to/warp-build -- \
  -DLEAPP_WARP_ONNXRUNTIME_VERSION=1.24.4 -DCMAKE_INSTALL_LIBDIR=lib
```

Use your installed backend's version; omitting the setting uses Python's ONNX
Runtime version. Alternatively, supply `-DORT_INCLUDE_DIR=/path/to/headers`.
`-DWARP_ROOT`, `-DTorch_DIR`, and `-DCUDAToolkit_ROOT` select existing SDKs;
the build does not install them. Keep Warp's export and runtime releases aligned.
Replay requires CUDA even for CPU model instances.
The runner supports CUDA device 0 only. ONNX GPU models use a CUDA custom-op kernel,
keeping replay tensors on the GPU; CPU models retain host staging. The embedded
archive is copied to host once when loading. Warm each model serially before
concurrent inference.
PT2 model unload releases expired Warp cache entries while preserving live models.
Retained Warp results remain valid across subsequent runs and session closure;
the adapters continue to own their internal replay buffers.

Install the adapters and link them beside the backends under
`LEAPP_TRITON_BACKENDS`. Use local backend directories as above so the links do
not modify the Triton SDK:

```sh
cmake --install /path/to/warp-build --prefix "$PWD/target/plugins"
ln -s ../../lib/libleapp_wrp_onnx_custom_op.so \
  target/plugins/backends/onnxruntime/libleapp_wrp_onnx_custom_op.so
ln -s ../../lib/libleapp_wrp_torch_custom_op.so \
  target/plugins/backends/leapp_pt2/libleapp_wrp_torch_custom_op.so
```

Make the selected Warp `bin/`, LibTorch `lib/`, and CUDA library directories
available to the loader, alongside NativeRT and Triton, through `LD_LIBRARY_PATH`
or an explicitly configured `CMAKE_INSTALL_RPATH`. The installation contains the
LEAPP adapters; the SDK libraries stay in their existing locations.

Session construction discovers adapters for the backends used by the bundle.
Absent adapters leave ordinary models unchanged; Warp models need their adapter
installed. An invalid installed adapter causes a load error, even for ordinary
models using that backend. PT2 loads the library before creating model instances
and retains its handle until those instances are destroyed.

Generate and run the optional integration fixtures from this package directory:

```sh
python tests/generators/generate_warp_models.py --output target/warp-fixtures
export LEAPP_WARP_FIXTURES="$PWD/target/warp-fixtures"
export WARP_APIC_TMPDIR="$(mktemp -d)"
LEAPP_TEST_ENGINE=triton cargo test --locked engines::tests::warp -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::warp -- --ignored --test-threads=1
```

### TorchScript

The TorchScript instance source documents positional ports and supported
returns. Preserve the PT2 mutation-isolation and response-completion guarantees.
Reset restores session feedback/defaults, not mutable state inside shared models.

```sh
cmake -S src/engines/triton/backends/torch/jit -B target/torchscript-backend \
  -DCMAKE_BUILD_TYPE=Release -DTRITON_ROOT="$TRITON_ROOT" \
  -DTORCH_ROOT="$TORCH_ROOT" -DBUILD_TESTING=ON
cmake --build target/torchscript-backend -j4
ctest --test-dir target/torchscript-backend --output-on-failure
./target/torchscript-backend/torchscript_backend_test cuda
cmake --install target/torchscript-backend --prefix "$PWD/target/plugins"
python tests/generators/generate_torchscript_models.py --output target/torchscript-fixtures
cargo test --locked engines::tests::torchscript -- --ignored --test-threads=1
```

The installed library is
`target/plugins/backends/leapp_torchscript/libtriton_leapp_torchscript.so`.
Use the same backend directory and loader setup as the PT2 plugin. Integration
checks require ONNX Runtime and PT2 for mixed ensembles. They cover positional
ports, mutation isolation across inputs and ensemble consumers, feedback after a
failed mutation, dynamic shapes, integer tensors, nested subgraphs, and backend
coexistence. `LEAPP_TORCHSCRIPT_FIXTURES` overrides the fixture directory.

## Python binding validation

Generate ONNX fixtures with `tests/generators/generate_models.py`, then run from the Rust
package directory:

```sh
LEAPP_TRITON_FIXTURES=target/triton-fixtures \
LEAPP_TRITON_BACKENDS=target/public-api-backends \
LEAPP_TEST_CUDA=1 python -m pytest bindings/python/tests
```

Install pytest, PyTorch and CuPy for the complete interoperability suite.
C++ bindings are a separate crate and are not needed to use this package.

## C++ binding validation

Generate fixtures with `tests/generators/generate_models.py`. Configure with
`BUILD_TESTING=ON`, `LEAPP_FIXTURES` pointing to the generated directory, and
`LEAPP_TEST_BACKENDS` pointing to the installed backends. Set
`LEAPP_TEST_CUDA=ON` to compile CUDA input/output tests with the CUDA SDK.
Tests default to native execution; set `LEAPP_TEST_ENGINE=triton` to exercise
Triton. Then run:

```sh
cmake --build target/cxx-build -j
ctest --test-dir target/cxx-build --output-on-failure
```

Also run `bindings/cxx/tests/consumer` against the installed SDK without Rust
sources or generated bridge headers. Python tests cover the shared lifecycle too.


## Runtime distribution and backend environments

Consumer installation is in Sphinx. Backend recipes produce a **local channel**;
public hosting is separate work.

### Core release artifacts

Run with Python 3.12+, Rust, Maturin, cargo-deny 0.20.2, and the workspace
dependencies available:

```sh
python packaging/licenses.py --check
python packaging/check_release.py
cargo deny --locked --workspace check licenses advisories
cargo package --locked -p leapp-rt --list
cargo package --locked -p leapp-rt
python packaging/check_release.py --crate target/package/leapp-rt-*.crate
python packaging/build_sdist.py
```

`build_sdist.py` uses Maturin to include the core and shared binding sources. It
prunes the unrelated C++ workspace entries from the source archive's Cargo lock
file without changing any retained dependency version. The resulting archive
builds with `--locked`. The `runtime-release.yml` workflow handles published
`runtime-vVERSION` releases. It checks the tag and versions, runs
`cargo publish --dry-run`, and builds ten manylinux_2_28 wheels from that archive:
CPython 3.10–3.14 on native x86-64 and aarch64 runners. Each wheel is installed
and imported before the complete artifact set is attached to the GitHub release.
The exporter uses `publish.yml` for its `v*` and `release-v*` tags.
Neither workflow uploads packages to a registry.

The first public release contains only the core crate, sdist, and wheels;
public backend channels/wheels remain future work. `--auditwheel skip` is only
for the local SDK producer's Conda packages.

`check_release.py` checks versions and the core crate's contents. The crate keeps
Rust tests needed by source builds and only three explicitly listed YAML fixtures,
each limited to 64 KiB. The archive check rejects exported models and unexpected
payloads and enforces a 10,000,000-byte compressed size limit.

`deny.toml` contains an empty license allow-list placeholder. Runtime CI always
checks dependency advisories and skips license enforcement with a notice while
`licenses.allow` is empty, so backend checks can run. Filling in the allow-list
automatically enables the CI license check. Release attachment still requires
both checks to pass; the dependency inventory is not an approval policy. Run the
checks separately while completing approval:
`cargo deny --locked --workspace check advisories` and
`cargo deny --locked --workspace check licenses`.

`dependency-licenses.csv` inventories all external crates in `Cargo.lock`,
including build-only, inactive-feature, and other-target dependencies. Its
non-Cargo SDK and tool entries are maintained separately. After updating the
lockfile, run `python packaging/licenses.py` to refresh the Cargo rows and
`bindings/python/THIRD-PARTY-NOTICES.txt`. The generator verifies published crate
archive checksums against the lockfile and preserves upstream license expressions
and notice texts. It reads Cargo's archive cache, downloading missing archives
from crates.io. CI rejects stale generated files.

The Python sdist and wheels include these dependency notices alongside LEAPP's
license through the Python project's `license-files` metadata. Identical notice
texts are included once with their source packages listed. The notices cover
the workspace's locked crates and vendored schema/header; they do not imply
that every listed dependency is linked into every wheel.

CI and each release-wheel job run `auditwheel show` and check the wheel archive
and ELF imports. Only the LEAPP extension and system-library imports are allowed;
backend SDK binaries fail the check. The complete reports are retained as
`runtime-wheel-audit*` workflow artifacts. To audit a local wheel:

```sh
python -m pip install auditwheel==6.8.2
python packaging/check_release.py --wheel target/wheels/*.whl
```

The runtime version is independent of the exporter version. Keep the runtime's
Cargo manifests, Python metadata, CMake version, and lockfile aligned when
preparing a release, and keep the packaged license copies synchronized with the
repository license.

### Package selection

| Package | Contents |
|---|---|
| `leapp-rt-environment` | Backend discovery configuration; no SDK libraries |
| `leapp-backend-onnx` | Native ONNX Runtime installation |
| `leapp-backend-torchscript` | Native TorchScript plugin |
| `leapp-backend-nativert` | NativeRT library and native plugin |
| `leapp-triton-{backend}` | Selected backend's Triton adapter and server dependencies |
| `leapp-onnx-cuda` | Optional ONNX Runtime CUDA provider and runtime dependencies |
| `leapp-onnx-tensorrt` | Optional ONNX Runtime TensorRT provider and builder dependencies |
| `leapp-warp-{engine}-{backend}` | Optional Warp adapter for the selected engine and backend |
| `leapp-rt-python`, `leapp-rt-cxx` | Optional local bindings, built with `--bindings`, for development before publication |

`leapp-libs-*` dependencies resolve once per environment. Core-only installs
contain no model SDKs; backend dependencies follow the engine/loader contracts
above. Consumer packages exclude compilers, exporters, SDK headers, and build
tools, except the public LEAPP/DLPack headers in the C++ package.

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

PR CI installs the core wheel and optional pip extras in fresh containers
without Pixi. It checks ONNX on CPU and CUDA, TorchScript and NativeRT on CPU
and CUDA, and ONNX TensorRT engine creation and cache reuse. The core wheel is
shared across these trials; each extra gets a separate consumer environment.

### Install an application environment

Use [Installation](../../docs/source/runtime/installation.rst) for consumer
creation and CPU/CUDA selection. Internal engine variants use `--engines triton`
or `--engines native triton` (native is the default); `--warp onnx` adds Warp.
Warp requires CUDA even for CPU model instances. Dependency-only Rust/C++
consumers use `--binding none`. ONNX TensorRT needs `--onnx-tensorrt` in both the
producer and ONNX CUDA consumer commands. Deployment needs a compatible OS and
GPU driver, not a build container or CUDA SDK.

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

### Build pip backend wheels

`payloads.py` collects dependencies, licenses, provenance, and relocated SDK
libraries for both `channel.py` (Conda) and `wheels.py` (Python). Wheels support
native ONNX, TorchScript, and NativeRT, including ONNX CUDA/TensorRT; Triton and
Warp use Pixi/manual installation. On an SDK host configured below:

```sh
pixi run build-runtime --output target/pip-sdk --glibc 2.39 \
  --backends onnx --cuda --onnx-tensorrt \
  --sdk-origin 'sdk-artifact@sha256:<verified-digest>'
pixi run package-wheels --input target/pip-sdk
python -m pip install --find-links=target/pip-sdk/backend-wheels \
  'leapp-rt[onnx,tensorrt]'
```

Use `--backends torchscript nativert` to build both Torch backends, or combine
these with `onnx` to share SDK libraries across every selected backend. Supply
`TORCH_ROOT` and matching `PYTORCH_SOURCE` as required by the producer. CPU-only
ONNX needs no CUDA SDK. ONNX CUDA providers are a separate component, so the
`onnx` extra stays CPU-only even in a wheelhouse containing GPU backends.

Backend wheels normally require the clean matching `runtime-vVERSION` source.
Before publication, add `--allow-unreleased` to `package-wheels`, build the core
wheel from the same checkout, and install that wheel with its extras:

```sh
python -m pip install --find-links=target/pip-sdk/backend-wheels \
  './target/wheels/leapp_rt-0.1.0-cp312-cp312-linux_x86_64.whl[onnx,tensorrt]'
```

Use the host interpreter's actual wheel filename. `--allow-unreleased` does not
publish. All SDK components must come from one producer build; SDK changes need
a new backend release, not mixed builds under one distribution version.

Extras default to CUDA 13; explicit CUDA 12/13 extras select distinct GPU
components pinned with the core version. SDKs share `_leapp_rt_backends` and
retain native notices as regular files with relative ELF lookup paths. Neither
Pixi activation nor `LD_LIBRARY_PATH` is needed; driver/platform libraries stay
external.

The platform tag records the architecture and glibc baseline. Jetson SDK wheels
use a Linux architecture tag and enforce their recorded Jetson Linux release
when opening a session, since pip tags cannot express that release. Core wheels
retain their existing manylinux baseline and contain no backend libraries.

### Build a local channel

Run on the target architecture in a trusted, compatible SDK environment. Install
[Pixi 0.84 or newer](https://pixi.prefix.dev/latest/installation/), then run from
`packages/leapp-rt`. The producer environment in `pixi.toml` is separate
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
configuration and requires no SDK. For unreleased development builds, add
`--bindings python` to the producer command
and `--core-source channel` to the consumer command. Local C++ validation can use
`--bindings cxx` and `--core-source channel --binding cxx`; public C++ integration
uses CMake source builds. The local binding packages are not public release
artifacts.

The producer's `--onnx-tensorrt` option requires ONNX Runtime's TensorRT provider,
a compatible `TENSORRT_ROOT`, and `--cuda`. With Triton selected, the producer
packages the SDK's stock ONNX Runtime adapter.

Build natively with each architecture's SDK; CUDA, PyTorch, and TensorRT are
not cross-compiled. `--glibc` declares the deployment baseline, not the build-host
version. NativeRT uses matching PyTorch sources against prebuilt LibTorch through
the existing CMake target; missing/incompatible SDKs must fail, not omit a backend.

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
build the existing [Warp adapters](#warp-operators)
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

### Validate deployment

For a package smoke check, generate fixtures only for the selected backends in
the SDK build environment. Core-only checks need no model fixtures. TensorRT
plans must be generated on the target GPU.
Copy the fixtures and `tests/deployment/` into a clean runtime container, without
mounting the source SDK or producer environment. Set the existing fixture variables
from [AGENTS.md](#minimal-backend-installations).

Install one consumer at a time and run:

```sh
pixi run --manifest-path /app/pixi.toml python /tests/check_packages.py \
  --backends onnx --engines native --cuda --report /results/onnx.json
```

The audit runs selected engines/backends and Warp adapters, rejects build tools
and unselected runtimes, checks file hashes/library isolation, and records loaded
paths and package sizes. It uses `--installed-only`; missing-backend execution
belongs to the [minimal-installation matrix](#minimal-backend-installations).

For release validation, repeat for core only (omit `--backends`), each individual
backend, ONNX + TensorRT, and TorchScript + NativeRT. Compare the core hashes
between reports: they must remain identical. Also run the existing C++ installed
consumer and test a second installation prefix, backend add/remove, and rejection
of mismatched builds. The audit already exercises mixed ONNX/TensorRT graphs
and installed Warp adapters where selected. A clean consumer container, not merely a
restricted backend directory in a full SDK container, is required for isolation
claims. Run the SDK/GPU matrix on configured workers before releasing those
backend packages.
