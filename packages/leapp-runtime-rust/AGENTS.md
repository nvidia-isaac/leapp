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
  initial values, feedback, and session-owned output caches.
- `src/engines.rs`: private engine contract; native is selected by default.
- `src/engines/memory*`: reusable CPU, pinned-host, and CUDA allocations, copies,
  DLPack validation, and output pooling. Keep engine SDK types out of this code.
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
   `LEAPP_ANALYSIS_ONLY=1` checks do not validate an SDK upgrade.

## Validation

Run from this package on Linux with Triton 2.69.0, CUDA, and the backend libraries
available. Set `TRITON_ROOT`, `CUDA_ROOT`, and `LD_LIBRARY_PATH` for the installation.
Set `LEAPP_TRITON_BACKENDS` to the backend directory and `LEAPP_TEST_CUDA=1` for
GPU integration coverage. Native memory tests require a GPU.

```sh
python tests/generators/generate_models.py --output target/triton-fixtures
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace -- --test-threads=1
RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps
```

`LEAPP_TRITON_FIXTURES` can select another fixture directory. For editor checks on
a machine without the SDKs, use `LEAPP_ANALYSIS_ONLY=1 cargo check --locked
--workspace --all-targets`; this cannot produce a runnable build.

Optional backend fixtures use `generate_pt2_models.py`,
`generate_torchscript_models.py`, and `generate_tensorrt_models.py` under
`tests/generators/`, each with `--output`. TensorRT also accepts
`--trtexec /usr/bin/trtexec`; build its fixtures on the target GPU.
Set `LEAPP_PT2_FIXTURES`, `LEAPP_TORCHSCRIPT_FIXTURES`, or
`LEAPP_TENSORRT_FIXTURES` and run the matching ignored integration tests:

```sh
cargo test --locked public_pt2 -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::nativert -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::torchscript -- --ignored --test-threads=1
cargo test --locked engines::triton::tests::tensorrt -- --ignored --test-threads=1
```

For Warp, install the adapters as described in `src/engines/triton/backends/README.md`.
Generate `tests/generators/generate_warp_models.py` fixtures, set
`LEAPP_WARP_FIXTURES` and an empty `WARP_APIC_TMPDIR`, then run
`LEAPP_TEST_ENGINE=triton cargo test --locked engines::tests::warp -- --ignored --test-threads=1`.
Repeat with `LEAPP_TEST_ENGINE=native` to exercise native execution.
This covers ONNX/PT2 CPU and GPU instances, CPU/pinned/CUDA output placement,
pool limits 0/1/64, changed inputs, retained results, peer sessions, and unload.
Warp replay requires CUDA even with CPU model instances.

The Python and C++ binding READMEs describe building and testing each adapter.
Also verify the installed C++ SDK with `bindings/cxx/tests/consumer` so installed
headers and CMake targets work without generated bridge headers or Rust sources.
Pool changes must preserve retained/weak owners, exposure exclusion, byte offsets,
failed-copy cleanup, independent peer caches, and CPU/pinned/CUDA behavior.
