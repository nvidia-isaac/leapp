<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime contributor guide

Keep the README focused on installation, public APIs, and ownership contracts.
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
   references in this file and `README.md`, and check build/CI configuration for
   old release pins.
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
   Rebuild and test the Python and C++ bindings using their READMEs, and verify
   the installed C++ SDK with `bindings/cxx/tests/consumer`.
   Type checking alone does not validate an SDK upgrade.

## Validation

Core builds and checks require no Triton or CUDA SDK. Run from this package:

```sh
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --test bundle --test dlpack --test binding_lifecycle
cargo test --locked --lib engines::memory:: -- --test-threads=1
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
```

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

For Warp, install the adapters as described in `src/engines/triton/backends/README.md`.
Generate `tests/generators/generate_warp_models.py` fixtures, set
`LEAPP_WARP_FIXTURES` and an empty `WARP_APIC_TMPDIR`, then run
`LEAPP_TEST_ENGINE=triton cargo test --locked engines::tests::warp -- --ignored --test-threads=1`.
Repeat with `LEAPP_TEST_ENGINE=native` to exercise native execution.
This covers ONNX/PT2 CPU and GPU instances, CPU/pinned/CUDA output placement,
changed inputs, retained results, peer sessions, and unload.
Warp replay requires CUDA even with CPU model instances.

The Python and C++ binding READMEs describe building and testing each adapter.
Also verify the installed C++ SDK with `bindings/cxx/tests/consumer` so installed
headers and CMake targets work without generated bridge headers or Rust sources.
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
