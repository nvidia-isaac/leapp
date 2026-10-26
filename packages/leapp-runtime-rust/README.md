<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust runtime

This package targets Linux and executes exported LEAPP bundles through embedded
Triton 2.69.0. All builds require the Triton and CUDA runtime SDKs. Rust, Python,
and C++ expose bundle inspection and synchronous execution.

## Included code

| Layer | Contents |
| --- | --- |
| Rust core | Immutable bundles, session state, shared engines, DLPack leases, and results |
| Public inspection types | `PortRef`, `PortSpec`, `DataType`, `Metadata`, `MetadataValue`, `BundleError`, `ErrorCode` |
| Common lifecycle crate | `SessionHandle`, error-source preservation, and cleanup after poisoned locks; three focused regression tests |
| Python | Bundle inspection, synchronous sessions, DLPack protocol inputs and outputs, typed stubs and structured exceptions |
| C++ | C++20 bundle/session facade, borrowed DLPack descriptors, managed outputs, private cxx transport, installable CMake target |
| Tests | Bundle inspection, conversion, backend execution, tensor ownership, and language-binding lifecycle tests |

The core crate owns bundle inspection, graph validation, independent session state,
and results. Each `Session` holds a shared `Arc<dyn Engine>` with synchronous
execution and tensor-copy operations. The `src/engines/triton` module owns Triton
conversion, configuration, repositories, native calls, CUDA, allocation pooling,
and backend plugins. There is no separate engine crate or executor layer.
`src/engines.rs` defines the internal contract and selects an implementation at
session initialization using `SessionOptions::engine`. Engine implementations
and installation settings are private. Each subdirectory under `src/engines/`
contains one engine; currently the only engine is `triton/`. Its model backends
remain nested there. Additional engines can be included in the same binary and
selected independently for each session.

Module entry points use `name.rs` beside `name/` subdirectories. Within Triton,
`native` owns server calls, requests, errors, and C bindings;
`memory` owns buffers, pooling, CUDA, and DLPack conversion; and `convert` owns
model configuration, repository creation, and the protobuf schema.

The `src/dlpack` module provides the ABI and lifetime guards. The core uses
explicit unsafe borrows only for private initial state and feedback descriptors.
The `bindings/common` crate adds one lock per session and explicit closure for
Python and C++. The Rust core and both bindings require Linux and the native SDKs.

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
names with `input_info(node, port)` and `output_info(node, port)` for
descriptors.

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
names, and leaves engine availability to execution.

The loader preserves ordinary YAML metadata, duplicate-key handling, and YAML
merges. Custom YAML tags are unsupported and fail during parsing; LEAPP exports
do not generate them.

## Synchronous execution

Rust callers can borrow numeric CPU slices directly, without allocation or unsafe
code. The constructor infers the dtype and checks the shape:

```rust
use leapp_runtime::dlpack::BorrowedTensor;

let values = [1.0_f32, 2.0, 3.0, 4.0];
let input = BorrowedTensor::cpu(&values, &[4])?;
let result = session.run(&[("preprocess/observation", input)])?;
let action = result.get("policy/action").unwrap();
let values = action.to_cpu_vec::<f32>()?;
```

`BorrowedTensor::cpu` retains immutable borrows of the values and shape. It accepts
fixed-width Rust integers and `f32`/`f64`; booleans and half floats use raw descriptors.
`to_cpu_vec` accepts CPU, pinned CPU, and CUDA outputs, validates dtype and layout,
and returns an independent Rust vector. CUDA transfers select the source device,
finish before returning, and restore the caller's device. It rejects outputs already exposed through
`descriptor()` or `export()` on any clone, since foreign consumers may still write
them. Copying and first exposure are synchronized. CPU copies do not exclude the
original storage from pool reuse. An existing `BorrowedTensor` can also be read
without copying through its safe `as_cpu_slice` method when its storage is on the host.
CPU-slice inputs work with both CPU and GPU execution; output placement does not
change the safe input or output-copy APIs.

Array libraries and CUDA callers can use standard DLPack descriptors through
`unsafe { BorrowedTensor::new(&descriptor) }`. They must retain the descriptor,
shape/strides, and ready, immutable storage through the synchronous call. CUDA
producer work must already be complete. Outputs expose their descriptor through
`descriptor()` or transfer a fresh managed lease with `export(true).into_raw()`;
the receiving consumer must invoke that lease's deleter exactly once.

See `examples/run.rs` for a complete CPU example.

`SessionOptions` selects the engine, `ExecutionDevice`, and output placement,
defaulting to `EngineKind::Triton`, CPU execution, and CPU outputs:

```rust
use leapp_runtime::{EngineKind, Session, SessionOptions};

let options = SessionOptions {
    engine: EngineKind::Triton,
    ..Default::default()
};
let session = Session::open(&bundle, options)?;
```

Triton is currently the only supported engine. The engine is selected when
opening a session, and `new_session()` retains that selection by sharing the
loaded engine. The Triton adapter reads `LEAPP_TRITON_BACKENDS` when opening
a session, falling back to `$TRITON_ROOT/backends` or `/opt/tritonserver/backends`
when `TRITON_ROOT` is unset. Set installation variables before starting the process;
`new_session()` shares the already-loaded engine without rediscovering it.
`session.info()` borrows the retained bundle snapshot. Every public input must be supplied
exactly once, using its LEAPP name. Generated Triton names and internal ports
are hidden. Results contain only declared public outputs, in declaration order;
repeated declarations share one named entry. `get` returns `None` for absent
names and `iter` traverses names and tensors without copying data.

Each session owns its delayed feedback: call N consumes the outputs from call
N-1. Feedback and unconnected inputs start from exported safetensors values, or
zeros when no value is exported. Dynamic state requires a concrete initial
value. Initial values for public or forward-connected inputs are superseded by
supplied inputs or graph routing. A successful call publishes all next-state
tensors together; validation and inference failures leave the session's feedback
unchanged. `reset()` restores the values captured when the session was opened, without
rereading files or changing retained results. It does not reset model-internal
state.

`Session::open(&bundle, options)` loads models and captures initial state.
`session.new_session()` shares the loaded engine and construction options while
starting from captured initial state. It does not copy current feedback or reread
files. Separate sessions can run concurrently with independent feedback; `run`
borrows one session mutably. `close()` releases only that session's state and
engine reference. Peers and retained results remain usable. The last engine
owner stops Triton and removes its temporary repository. Model-internal mutable
state remains shared, and `reset()` affects only explicit session state.

Tensor exchange uses DLPack 1.0 descriptors directly. The small `dlpack` module
contains generated ABI definitions and lifetime guards; it has no tensor operations.
Inputs must use supported numeric dtypes and contiguous layouts. The Triton engine
honors byte offsets and rejects unsupported layouts before inference. Frameworks
or callers perform explicit contiguous copies, dtype conversion, and transfers.

Outputs retain their allocation and metadata independently of the runtime. Exports
may alias other exports of the same result and may be writable. Outputs that also
feed back are copied to the requested public location before publication, protecting
private feedback and reset snapshots. Do not mutate an allocation while inference
or another consumer reads it. The Triton engine pools up to 64 allocations and 64 MiB of
storage, matched by byte size and memory location. Reuse requires the pool to be
the only owner, with no weak owners. Descriptor access or DLPack export permanently
excludes an allocation from reuse because foreign GPU work may outlive a lease.
Internal bounded borrows and copies keep their source eligible for reuse. Public
feedback copies use the same pool; fresh copy destinations skip zero-initialization
and enter the pool only after a successful copy. Retained results and feedback
always prevent reuse. Bundle inspection retains its descriptive `DataType`.

`SessionOptions::output_placement` sets a default location and per-output
overrides keyed by public LEAPP names at construction. `run` accepts only inputs.
New sessions inherit this placement, and every output uses its exact requested location.
Initial state, default inputs, and private feedback stay on the execution device.
Public outputs retain their requested location, including outputs also used as
feedback. CUDA and pinned allocations are provided by the Triton engine.
Execution errors expose `code()`, `phase()`, and their source chain.

A runnable example uses the generated policy fixture:

```sh
cargo run --locked --example run -- target/triton-fixtures/policy.yaml
```

## Execution conversion

Session setup checks competing input producers and forward cycles. The Triton
engine converts the validated graph into model configurations, ensemble routing,
and a temporary model repository. Unsupported Triton backends fail during
conversion; bundle inspection continues to accept custom backend names.
The model configuration schema targets Triton 2.69.0.

## Embedded Triton

The native Triton engine loads generated
repositories and performs synchronous inference with exact output placement.
Each output allocation is retained by its managed DLPack leases until the last
owner releases it. Eligible idle allocations remain in the bounded pool; evicted
allocations are freed outside its lock. Server shutdown precedes pool and repository
cleanup. The runtime checks that the loaded Triton version is 2.69.0, matching the
bundled configuration schema.

Generate ONNX fixtures and run the native engine tests on Linux:

```sh
python tests/generators/generate_models.py --output target/triton-fixtures
cargo test --locked engines::triton::tests::engine -- --test-threads=1
```

`LEAPP_TRITON_FIXTURES` and `LEAPP_TRITON_BACKENDS` select fixture and backend
directories. CPU model execution still requires the CUDA runtime SDK.

## Native tensor storage

The `engines::triton` module owns CPU, pinned-host, and CUDA buffers and provides
direct DLPack conversion, pooled output storage, and synchronized copies for private
state. It requires Linux, the Triton SDK, and the CUDA runtime. Set `TRITON_ROOT` and `CUDA_ROOT` when they are not
installed under `/opt/tritonserver` and `/usr/local/cuda`. Native dependencies
must be discoverable through `LD_LIBRARY_PATH`.

```sh
cargo test --locked engines::triton -- --test-threads=1
```

## Build and test

Run from this directory on Linux with a Rust toolchain, the native SDKs, and
generated ONNX fixtures:

```sh
cargo test --locked -- --test-threads=1
cargo check --locked -p leapp-runtime
cargo clippy --locked -p leapp-runtime --all-targets -- -D warnings
cargo fmt --all --check
```

The default workspace member is the core crate, including native engine tests,
conversion checks, DLPack ABI/ownership tests, and the three common lifecycle tests.
The complete workspace also requires a C++20 compiler and Python 3.10 or newer. See [Python
bindings](bindings/python/README.md) and [C++ bindings](bindings/cxx/README.md)
for installation and binding tests. Inspection tests create opaque artifact
bytes. Native execution tests use the generated ONNX fixtures described above.

For a complete Linux build with the native SDKs available:

```sh
cargo check --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace -- --test-threads=1
```

Generate the ONNX fixtures before running native tests. Optional backend tests
are invoked explicitly in the backend sections below. Language-binding tests
require building the corresponding extension or SDK.

`LEAPP_ANALYSIS_ONLY=1 cargo check --locked --workspace --all-targets` supports
editor analysis without native SDKs. It skips native linking and cannot produce
a runnable build. Core tests follow the repository's `tests/unit_tests/` and
`tests/functional_tests/` split, with Triton-specific tests under `triton/` in
each. Tests that inspect private state are included by their implementation
modules; Cargo also runs the bundle and DLPack test targets from `unit_tests/`.
Shared fixtures and generators remain in `tests/fixtures/` and
`tests/generators/`. Language-binding tests stay under their respective bindings.

## Shared lifecycle

The common lifecycle crate preserves runtime error sources and completes
shutdown cleanup even when locks are poisoned. The lifecycle regression
target compiles the wrapper against a small test-only session API. Native
execution tests exercise session state and results. See the [session reset
contract](bindings/common/README.md#session-reset) for state ownership and reset
behavior.

## PT2 execution

The runtime configures PT2 artifacts for an externally installed `leapp_pt2`
Triton backend. Set `LEAPP_TRITON_BACKENDS` to a directory containing
`leapp_pt2/libtriton_leapp_pt2.so` and any other backends used by the bundle.
Backend implementations and their dependencies are installed separately. See
the [plugin build guide](src/engines/triton/backends/README.md) to build `leapp_pt2` from source.

PT2 inputs and outputs follow exported-program order. Archive devices must match
the selected model instance. The runtime supplies feedback, default inputs, and
output placement through the same session API used for ONNX models.

Generate fixtures with a compatible PyTorch installation and run the optional
integration checks against the installed backend:

```sh
python tests/generators/generate_pt2_models.py --output target/pt2-fixtures
cargo test --locked public_pt2 -- --ignored --test-threads=1
```

`LEAPP_PT2_FIXTURES` selects the fixture directory. `LEAPP_TRITON_BACKENDS`
selects the backend installation. CUDA fixtures require an available GPU.

## TorchScript execution

The runtime configures scripted and traced artifacts for an externally installed
`leapp_torchscript` Triton backend. Install
`leapp_torchscript/libtriton_leapp_torchscript.so` beneath the runtime's backend
directory, with its dependencies discoverable by the dynamic loader. See the
[plugin build guide](src/engines/triton/backends/README.md#torchscript) for source builds.

Models accept positional tensor arguments and return a tensor or a flat tuple
of tensors. Conversion maps YAML port order to `INPUT__0`, `INPUT__1`, and
`OUTPUT__0`, `OUTPUT__1`; public LEAPP names remain unchanged. Session reset
restores explicit feedback and default inputs, without resetting model-internal
mutable state.

The optional integration tests exercise the runtime against an installed backend:

```sh
python tests/generators/generate_torchscript_models.py --output target/torchscript-fixtures
cargo test --locked engines::triton::tests::torchscript -- --ignored --test-threads=1
```

`LEAPP_TORCHSCRIPT_FIXTURES` selects the fixture directory, and
`LEAPP_TRITON_BACKENDS` selects the backend installation. The plugin must preserve
borrowed inputs and live outputs to satisfy the runtime's ownership contract.

## TensorRT engines

Bundles can reference prebuilt serialized TensorRT engines with `backend: trt`
or its `tensorrt` alias:

```yaml
parameters:
  backend: trt
  model_path: policy.engine
```

The artifact filename is preserved through `default_model_filename`, so `.plan`
and `.engine` files work without renaming. The converter emits `backend:
tensorrt`, `platform: tensorrt_plan`, `max_batch_size: 0`, and a GPU instance
for the selected `ExecutionDevice::Cuda`. Port names, dtypes, and shapes must
describe the engine's I/O. Triton validates engine compatibility and input
shapes; the Rust runtime neither builds engines nor inspects their payloads.

Install the supplied plugin at
`<backend_directory>/tensorrt/libtriton_tensorrt.so`, with its TensorRT and CUDA
dependencies available to the loader. Execution uses `SessionOptions`,
`Session`, output placement, feedback, and reset APIs. CPU, pinned-host, and
CUDA input/output buffers are supported by the Triton engine. TensorRT
model execution always requires a GPU.

Dynamic dimensions use the engine's first optimization profile (profile 0). This
integration targets ordinary tensor I/O. Shape-tensor configuration, selection
of additional profiles, custom TensorRT plugins, and automatic engine
compilation are not exposed.

Build fixture engines on the target GPU with the supplied TensorRT toolchain:

```sh
python tests/generators/generate_tensorrt_models.py --output target/tensorrt-fixtures \
  --trtexec /usr/bin/trtexec
cargo test --locked engines::triton::tests::tensorrt -- --ignored --test-threads=1
```

The generator requires NumPy and ONNX and uses `trtexec` only in test tooling.
`LEAPP_TENSORRT_FIXTURES` selects another fixture directory and
`LEAPP_TRITON_BACKENDS` selects the backend installation for tests. Engines must
be compatible with the deployed TensorRT version and GPU; tests build them in
that environment instead of checking binary engines into the repository.
Integration checks cover TensorRT-only and ONNX→TensorRT ensembles, host/CUDA
buffers, retained outputs, and dynamic profile bounds and recovery. Converter
tests also check CPU execution rejection.
