<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust bundle API and shared lifecycle sources

This is the standalone common foundation extracted from the Triton-based
and native runtime implementations. It loads and validates
exported LEAPP bundles and exposes their public inspection API in Rust, Python,
and C++. The original common runtime/session lifecycle is retained unchanged
for integration with either engine. Bundle loading does not initialize an
inference backend.

The package path and library names match the runtime branches to make this a
base for either implementation. This worktree does not refactor those branches
to depend on a new crate. The two Python distributions share the `leapp_runtime`
import name and should be installed in separate environments.

## Included code

| Layer | Contents |
| --- | --- |
| Rust core | `Bundle`, YAML schema parsing, graph and artifact validation, SHA-256 checks, safetensors initial-state validation |
| Public inspection types | `PortRef`, `PortSpec`, `DataType`, `Metadata`, `MetadataValue`, `BundleError`, `ErrorCode` |
| Common lifecycle sources | Original `RuntimeHandle`, `SessionHandle`, and lifecycle errors; three focused regression tests |
| Python | PyO3 bundle bindings, detached metadata and port descriptors, typed stubs and structured exceptions |
| C++ | C++20 bundle facade, typed metadata, status and data-type headers, private cxx transport, installable CMake target |
| Tests | Existing Rust bundle regression suite and independent Python/C++ inspection and lifetime tests |

Concrete engine runtimes, sessions, results, tensor/DLPack code, graph planners,
the Triton converter, and backend adapters remain in the runtime branches.
Python and C++ share the core bundle implementation. The `bindings/common`
lifecycle source depends on concrete execution types, so it is not a separate
compiled crate in this bundle-only checkout. See the
[common lifecycle sources](bindings/common/README.md) for integration and testing.
The extraction has no Triton, CUDA, ONNX Runtime, PyTorch, or NumPy dependency.
The C++ binding currently supports Unix platforms (Linux and macOS), using native
path bytes.

## Rust

```rust
use leapp_runtime::Bundle;

let bundle = Bundle::open("export/policy.yaml")?;
for output in bundle.pipeline_outputs() {
    let spec = bundle.output_info(&output.node, &output.port).unwrap();
    println!("{}: {:?} {:?}", output.name(), spec.dtype, spec.shape);
}
# Ok::<(), leapp_runtime::BundleError>(())
```

`Bundle` is an immutable, reference-counted snapshot. Clones share the snapshot;
inspection is safe across threads. Getters preserve declaration order and
unknown metadata. Exporter parameters retain their original paths and backend
names. Internal paths are resolved relative to the YAML, falling back to the
supplied artifact path. Optional SHA-256 hashes are checked before publication.
Files must be available while opening; their bytes are not retained for later
execution.

Bundle loading validates graph ports and connections, artifacts, and optional
initial-value files. It does not check whether an inference engine can execute
the graph. Declared external inputs can remain unresolved, matching the source
implementations' inspection contract. Python and C++ represent missing input
descriptors as `None` and `std::nullopt`, respectively.

The loader normalizes known backend aliases, accepts custom nonempty backend
names, and leaves adapter availability to execution.

## Build and test

Run from this directory with a Rust toolchain:

```sh
cargo test --locked
cargo check --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

The default workspace member is the core crate, including the three common
lifecycle tests. These tests require no engine or language-binding SDK. Building
the complete workspace
also requires a C++20 compiler and Python 3.10 or newer. See
[Python bindings](bindings/python/README.md) and
[C++ bindings](bindings/cxx/README.md) for installation and binding tests.
All inspection tests create opaque artifact bytes; no model generation is needed.

## Extraction provenance

The shared sources, loader, and tests were extracted from the existing runtime
implementations for backend-independent inspection.

Execution feature gates and dependencies were removed. The C++ bundle facade
was split out of `runtime.cc`, and `DataType` moved from `tensor.h` into
`data_type.h`. Binding manifests, Python exports and stubs, and CMake packaging
now expose only implemented inspection operations.

The shared `bindings/common/src/lib.rs` and `runtime.rs` from the same source
commit are retained byte-for-byte. The lifecycle regression target compiles the
original implementation against a small test-only runtime API. No generic engine
interface or lifecycle cleanup changes are introduced by this extraction.

Binding regression tests also exposed a pre-existing tagged-metadata parsing
failure. The loader now reads YAML tags directly and separates unknown metadata
from serde's schema-field buffer before restoring it to the snapshot. Tests cover
tags at every inspection level, nested values, duplicate keys, and YAML merges.
