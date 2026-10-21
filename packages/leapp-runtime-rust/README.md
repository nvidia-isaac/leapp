<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust bundle and Session API

This package loads and validates exported LEAPP bundles and exposes their public
inspection API in Rust, Python, and C++. It also exports the core Rust Session API and compiles its shared binding
wrapper. This revision defines the execution contract: `Session::open()` returns
`ErrorCode::Unsupported` until the runtime integration supplies execution.
Bundle loading does not initialize an inference backend.

The inspection and full-runtime Python distributions share the `leapp_runtime`
import name and should be installed in separate environments.

## Included code

| Layer | Contents |
| --- | --- |
| Rust core | `Bundle`, YAML schema parsing, graph and artifact validation, SHA-256 checks, safetensors initial-state validation |
| Public inspection types | `PortRef`, `PortSpec`, `DataType`, `Metadata`, `MetadataValue`, `BundleError`, `ErrorCode` |
| Core Session API | `Session`, `SessionOptions`, `OutputPlacement`, `RunResult`, device/memory types, and `RuntimeError` |
| Tensor API | Backend-independent DLPack descriptors and ownership types used by Session inputs and results |
| Common lifecycle crate | `SessionHandle`, independent closure, error-source preservation, and cleanup after poisoned locks; three focused regression tests |
| Python | PyO3 bundle bindings, detached metadata and port descriptors, typed stubs and structured exceptions |
| C++ | C++20 bundle facade, typed metadata, status and data-type headers, private cxx transport, installable CMake target |
| Tests | Rust bundle regression suite and independent Python/C++ inspection and lifetime tests |

Execution, feedback state, graph planning, native storage, and backend adapters
are supplied by the runtime integration. The Rust Session contract and the
`bindings/common` crate compile here against the same exported API types.
Python and C++ currently expose bundle inspection; their executable Session
bindings arrive with runtime integration. See the
[common lifecycle crate](bindings/common/README.md) for integration and testing.
This package has no Triton, CUDA, ONNX Runtime, PyTorch, or NumPy dependency.
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
Core bundle tests live in `tests/unit_tests/`, with shared fixtures in
`tests/fixtures/`. Language-binding tests stay under their respective bindings.
All inspection tests create opaque artifact bytes; no model generation is needed.

## Core Session contract

`Session::open(&bundle, options)` accepts immutable construction options:

```rust
use leapp_runtime::{ExecutionDevice, OutputPlacement, Session, SessionOptions};

let options = SessionOptions {
    device: ExecutionDevice::Cpu,
    output_placement: OutputPlacement::default(),
};
# let bundle = leapp_runtime::Bundle::open("export/policy.yaml")?;
let session = Session::open(&bundle, options);
// This API revision returns a RuntimeError with ErrorCode::Unsupported.
# Ok::<(), leapp_runtime::BundleError>(())
```

The public lifecycle is `open`, `info`, `is_open`, `new_session`, `run`, `reset`,
and `close`. `run` accepts named `dlpack::BorrowedTensor` inputs and returns a
`RunResult` retaining `SharedTensor` outputs. There is no separate public Runtime
owner, no per-call RunOptions, and no engine field in SessionOptions. Output
placement is configured at construction and inherited by every new session.

The execution contract requires `new_session()` to share loaded models and
immutable options while starting from captured initial state, not current
feedback. `reset()` restores that initial state. Closing one session preserves
its peers and retained results. Execution is deliberately unavailable in this
revision; the runtime integration implements these operations without changing
the public signatures. There is no runtime selector or optional backend feature.

## Shared lifecycle

`bindings/common` compiles its `SessionHandle` wrapper against the exported core
Session API. It provides serialization, error-source preservation, and independent
closure. Its existing three lifecycle tests use a small test-only session to
exercise waiting and poisoned-lock cleanup before native execution is available.
The wrapper itself is built against the real core crate by workspace checks.
See the [session reset contract](bindings/common/README.md#session-reset).
