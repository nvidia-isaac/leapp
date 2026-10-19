<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust bundle API and shared lifecycle sources

This package loads and validates exported LEAPP bundles and exposes their public
inspection API in Rust, Python, and C++. It also provides shared runtime/session
lifecycle sources for ownership, error handling, and shutdown. Bundle loading
does not initialize an inference backend.

The inspection and full-runtime Python distributions share the `leapp_runtime`
import name and should be installed in separate environments.

## Included code

| Layer | Contents |
| --- | --- |
| Rust core | `Bundle`, YAML schema parsing, graph and artifact validation, SHA-256 checks, safetensors initial-state validation |
| Public inspection types | `PortRef`, `PortSpec`, `DataType`, `Metadata`, `MetadataValue`, `BundleError`, `ErrorCode` |
| Common lifecycle sources | `RuntimeHandle`, `SessionHandle`, error-source preservation, and cleanup after poisoned locks; five focused regression tests |
| Python | PyO3 bundle bindings, detached metadata and port descriptors, typed stubs and structured exceptions |
| C++ | C++20 bundle facade, typed metadata, status and data-type headers, private cxx transport, installable CMake target |
| Tests | Rust bundle regression suite and independent Python/C++ inspection and lifetime tests |

Concrete engine runtimes, sessions, results, tensor/DLPack code,
and backend adapters are supplied by the full runtime packages.
Python and C++ share the core bundle implementation. The `bindings/common`
lifecycle source depends on concrete execution types, so it is not a separate
compiled crate in this inspection-only package. See the
[common lifecycle sources](bindings/common/README.md) for integration and testing.
The core inspection crate has no inference SDK or NumPy dependency.
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

`model_io(node)` returns `Option<(Vec<&str>, Vec<&str>)>`: input names followed
by output names, including internal ports. Names are unqualified and borrowed
from the bundle, with declaration order and duplicates preserved. An absent
model returns `None`; a model with no ports returns two empty lists. Use these
names with `input_info(node, port)` and `output_info(node, port)` for descriptors.

`Bundle` is an immutable, reference-counted snapshot. Clones share the snapshot;
inspection is safe across threads. Getters preserve declaration order and
unknown metadata. Exporter parameters retain their original paths and backend
names. Internal paths are resolved relative to the YAML, falling back to the
supplied artifact path. Optional SHA-256 hashes are checked before publication.
Files must be available while opening; their bytes are not retained for later
execution.

Bundle loading validates graph ports and connections, artifacts, and optional
initial-value files. It does not check whether an inference engine can execute
the graph. Declared external inputs can remain unresolved. Python and C++
represent missing input descriptors as `None` and `std::nullopt`, respectively.

The loader normalizes known backend aliases, accepts custom nonempty backend
names, and leaves adapter availability to execution.

The loader preserves ordinary YAML metadata, duplicate-key handling, and YAML
merges. Custom YAML tags are unsupported and fail during parsing; LEAPP exports
do not generate them.

## Execution conversion

The internal execution layer converts validated bundles into Triton model
configurations, ensemble routing, and temporary model repositories. Conversion
checks execution constraints, including competing input producers and forward
cycles, without initializing a backend. Unsupported execution backends fail
during conversion; bundle inspection continues to accept custom backend names.
The model configuration schema targets Triton 2.69.0.

## Native tensor storage

The `leapp-triton` crate owns CPU, pinned-host, and CUDA buffers and provides
borrowed tensor views and synchronized transfers. It requires Linux, the Triton
SDK, and the CUDA runtime. Set `TRITON_ROOT` and `CUDA_ROOT` when they are not
installed under `/opt/tritonserver` and `/usr/local/cuda`. Native dependencies
must be discoverable through `LD_LIBRARY_PATH`.

```sh
cargo test --locked -p leapp-triton
```

## Build and test

Run from this directory with a Rust toolchain:

```sh
cargo test --locked
cargo check --locked -p leapp-runtime
cargo clippy --locked -p leapp-runtime --all-targets -- -D warnings
cargo fmt --all --check
```

The default workspace member is the core crate, including conversion checks and the five common
lifecycle tests. These tests require no engine or language-binding SDK. The complete workspace requires Linux, the native SDKs described above, a
C++20 compiler, and Python 3.10 or newer. See
[Python bindings](bindings/python/README.md) and
[C++ bindings](bindings/cxx/README.md) for installation and binding tests.
All inspection tests create opaque artifact bytes; no model generation is needed.

## Shared lifecycle

`bindings/common/src/runtime.rs` preserves runtime error sources and completes
shutdown cleanup even when locks are poisoned. The lifecycle regression target
compiles this implementation against a small test-only runtime API, without
inference SDKs. See the [session reset contract](bindings/common/README.md#session-reset)
for state ownership and reset behavior.
