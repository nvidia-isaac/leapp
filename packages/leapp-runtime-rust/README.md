<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust runtime

This package loads exported LEAPP bundles and executes them through embedded
Triton. Rust bundle inspection and conversion checks require no inference SDKs.
Enable the `triton` feature for synchronous execution on Linux with Triton 2.69.0
and the CUDA runtime SDK. Python and C++ bindings expose bundle inspection.

## Included code

| Layer | Contents |
| --- | --- |
| Rust core | `Bundle`, YAML schema parsing, graph and artifact validation, SHA-256 checks, safetensors initial-state validation |
| Public inspection types | `PortRef`, `PortSpec`, `DataType`, `Metadata`, `MetadataValue`, `BundleError`, `ErrorCode` |
| Common lifecycle crate | `RuntimeHandle`, `SessionHandle`, error-source preservation, and cleanup after poisoned locks; five focused regression tests |
| Python | PyO3 bundle bindings, detached metadata and port descriptors, typed stubs and structured exceptions |
| C++ | C++20 bundle facade, typed metadata, status and data-type headers, private cxx transport, installable CMake target |
| Tests | Rust bundle regression suite and independent Python/C++ inspection and lifetime tests |

The core crate provides runtime/session ownership, graph conversion, tensors,
and results. Native calls are contained in the `leapp-triton` crate; the core
forbids unsafe Rust. The `bindings/common` crate provides shared close and locking
behavior. Python and C++ inspection bindings use the core bundle implementation.
Their inspection interfaces support Linux and macOS; native execution requires Linux.

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

## Synchronous execution

```rust
use leapp_runtime::{Bundle, Runtime, RuntimeOptions, TensorView};

let bundle = Bundle::open("exports/policy.yaml")?;
let runtime = Runtime::open(&bundle, RuntimeOptions::default())?;
let mut session = runtime.create_session()?;
let observation = [1.0_f32, 2.0, 3.0, 4.0];
let bias = [10.0_f32, 20.0, 30.0, 40.0];
let result = session.run(&[
    ("preprocess/observation", TensorView::cpu(&observation, &[4])?),
    ("policy/bias", TensorView::cpu(&bias, &[4])?),
])?;
let action = result.get("policy/action").unwrap().as_cpu_slice::<f32>()?;
session.reset()?;
```

`RuntimeOptions` selects `ExecutionDevice` and the backend installation directory
(defaults: CPU and `/opt/tritonserver/backends`). `runtime.info()` borrows the
retained bundle snapshot. Every public input must be supplied exactly once, using
its LEAPP name. Generated Triton names and internal ports are hidden. Results
contain only declared public outputs, in declaration order; repeated declarations
share one named entry. `get` returns `None` for absent names and `iter` traverses
names and tensors without copying data.

Each session owns its delayed feedback: call N consumes the outputs from call
N-1. Feedback and unconnected inputs start from exported safetensors values, or
zeros when no value is exported. Dynamic state requires a concrete initial value.
Initial values for public or forward-connected inputs are superseded by supplied
inputs or graph routing. A successful call publishes all next-state tensors
together; validation and inference failures leave the session's feedback unchanged.
`reset()` restores the values captured at runtime creation, without rereading files
or changing retained results. It does not reset model-internal state.

Separate sessions share models and can run concurrently; each has independent
feedback. `run` borrows the session mutably to prevent overlapping calls on the
same session. Dropping the runtime handle leaves existing sessions usable. The
last owner stops Triton and removes the temporary model repository. Results own
their allocations and remain usable after runtime/session destruction.

`TensorView::cpu` borrows typed input, while `cpu_bytes` handles packed native-endian
numeric values (including Boolean, float16, and bfloat16). Boolean bytes must be
0 or 1. `Tensor::copy_from_cpu` and `copy_from_cpu_bytes` create immutable CPU,
pinned-host, or CUDA storage; copies complete before returning. Owned tensors
can be reused as inputs through `view()` and cloned without copying storage.
`as_cpu_slice` and `as_cpu_bytes` borrow host data; device tensors require an
explicit `copy_to_cpu` or `copy_to_cpu_bytes`. Tensor and bundle inspection use
the same public `DataType` enum.

`run_with_options` accepts `RunOptions::output_placement`: a default location and
per-output overrides keyed by public LEAPP names. Output placement is exact.
Hidden feedback outputs use CPU storage; public outputs reused as feedback retain
the requested output location. CUDA and pinned allocations are included with the `triton` feature. Execution errors expose `code()`, `phase()`, and their source chain.

A runnable example uses the generated policy fixture:

```sh
cargo run --locked --features triton --example run -- target/triton-fixtures/policy.yaml
```

## Execution conversion

The internal execution layer converts validated bundles into Triton model
configurations, ensemble routing, and temporary model repositories. Conversion
checks execution constraints, including competing input producers and forward
cycles, without initializing a backend. Unsupported execution backends fail
during conversion; bundle inspection continues to accept custom backend names.
The model configuration schema targets Triton 2.69.0.

## Embedded Triton

The `triton` feature enables the native server adapter. It loads generated
repositories and performs synchronous inference with exact output placement.
Unreferenced output allocations can be reused; retained tensors keep their
storage, and exporting a raw pointer excludes that allocation from reuse.
Server shutdown precedes repository cleanup. The runtime checks that the loaded
Triton version is 2.69.0, matching the bundled configuration schema.

Generate ONNX fixtures and run the native adapter tests on Linux:

```sh
python tests/generate_models.py --output target/triton-fixtures
cargo test --locked --features triton adapter::tests -- --test-threads=1
```

`LEAPP_TRITON_FIXTURES` and `LEAPP_TRITON_BACKENDS` select fixture and backend
directories. CPU model execution still requires the CUDA runtime SDK.

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
Inspection tests create opaque artifact bytes. Native execution tests use the
generated ONNX fixtures described above.

## Shared lifecycle

The common lifecycle crate preserves runtime error sources and completes
shutdown cleanup even when locks are poisoned. The SDK-free lifecycle regression target compiles the wrapper against a small
test-only runtime API. Native execution tests exercise session state and results. See the [session reset contract](bindings/common/README.md#session-reset)
for state ownership and reset behavior.
