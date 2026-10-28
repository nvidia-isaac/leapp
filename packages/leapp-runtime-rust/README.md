<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust runtime

This package targets Linux and executes exported LEAPP bundles through either
embedded Triton 2.69.0 or native backend orchestration. Both engines are included
in one runtime and support CPU and CUDA execution. Rust, Python, and C++ expose
the same bundle inspection and synchronous Session API.

## Installation

Build on Linux with Rust 1.88 or newer, the Triton 2.69.0 SDK, and the CUDA
runtime SDK. Set `TRITON_ROOT` and `CUDA_ROOT` if they are not installed under
`/opt/tritonserver` and `/usr/local/cuda`. Make their shared libraries available
through `LD_LIBRARY_PATH`. CPU execution also requires the CUDA runtime SDK.

```sh
cargo build --locked --release
```

The Cargo package is `leapp-runtime`; its Rust import name is `lrt`.
See the [Python bindings](bindings/python/README.md) for `import lrt` and the
[C++ bindings](bindings/cxx/README.md) for the `lrt` namespace and `lrt::runtime`
CMake target.

Install the model runtimes and plugins required by your bundle and selected
engine; see [model support](#model-support).

For native execution, follow the
[native backend installation guide](src/engines/native/backends/INSTALLATION.md).
`LEAPP_NATIVE_BACKENDS` defaults to `/opt/leapp/backends`; `ORT_DYLIB_PATH` can
select an ONNX Runtime library directly. For Triton, use its ONNX Runtime and
TensorRT backends and the [Torch/Warp plugins](src/engines/triton/backends/README.md).
`LEAPP_TRITON_BACKENDS` defaults to `$TRITON_ROOT/backends` or
`/opt/tritonserver/backends`.

Set installation variables before starting the process. Backend dependencies
must also be discoverable through `LD_LIBRARY_PATH` or the system loader.

## Rust

```rust
use lrt::Bundle;

let bundle = Bundle::open("export/policy.yaml")?;
for output in bundle.pipeline_outputs() {
    let spec = bundle.output_info(&output.node, &output.port).unwrap();
    println!("{}: {:?} {:?}", output.name(), spec.dtype, spec.shape);
}
# Ok::<(), lrt::BundleError>(())
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
use lrt::dlpack::BorrowedTensor;
use std::collections::HashMap;

let values = [1.0_f32, 2.0, 3.0, 4.0];
let input = BorrowedTensor::cpu(&values, &[4])?;
let result = session.run(&HashMap::from([("preprocess/observation", input)]))?;
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

`SessionOptions` configures execution and output placement. Defaults use CPU
execution and CPU outputs. Triton is selected by default:

```rust
use lrt::{Session, SessionOptions};

let mut session = Session::open(&bundle, SessionOptions::default())?;
```

Set `device` to `ExecutionDevice::Cuda { device_id: 0 }` for GPU execution.
Select native execution with `SessionOptions { engine: EngineKind::Native, ..Default::default() }`,
Python `SessionOptions(engine="native")`, or C++ `EngineKind::kNative`.
Both engines are available in the same build, and peers inherit the selection.

`session.info()` borrows the retained bundle snapshot. Every public input must be supplied
in a `HashMap<&str, BorrowedTensor<'_>>`, using its LEAPP name. Names and
tensors are borrowed for the call; input map iteration order has no effect on routing. Generated Triton names and internal ports
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
files. Separate sessions may be called concurrently with independent feedback;
the native engine gives each session private fixed workspace buffers. `run`
borrows one session mutably. `close()` releases only that session's state and
engine reference. Peers and retained results remain usable. The last engine
owner releases loaded models and workspace storage. Triton additionally stops
its server and removes the temporary repository. Model-internal state sharing
depends on the backend, and `reset()` affects only explicit session state.

Tensor exchange uses DLPack 1.0 descriptors directly. The small `dlpack` module
contains generated ABI definitions and lifetime guards; it has no tensor operations.
Inputs must use supported numeric dtypes and contiguous layouts. Both engines
honor byte offsets and reject unsupported layouts before inference. Frameworks
or callers perform explicit contiguous copies, dtype conversion, and transfers.

Outputs retain their allocation and metadata independently of the runtime. Exports
may alias other exports of the same result and may be writable. Outputs that also
feed back are copied to the requested public location before publication, protecting
private feedback and reset snapshots. Do not mutate an allocation while inference
or another consumer reads it.

Each session owns its output cache. `SessionOptions::output_pool.max_entries`
limits cached allocations (default 64); zero disables caching while inference
continues to allocate normally. The byte budget is the bundle's largest static
output size multiplied by this entry limit, and grows when execution produces a
larger dynamic output. There is no fixed byte cap or eager allocation. This policy
limits cache retention, not memory held by models, live outputs, or feedback.
`new_session()` inherits the policy with an empty cache; close releases cached
references while retained outputs stay valid.

Both engines use this cache for output storage. Native execution copies results
from its fixed workspace into session-owned buffers using its existing transfer
streams. Reusing output storage does not change workspace addresses or prepared
backend metadata; live or exported results remain protected from reuse.

Reuse matches byte size and memory location and requires exclusive ownership,
including no weak owners. Descriptor access or DLPack export permanently excludes
an allocation from reuse because foreign GPU work may outlive its lease. Safe
`to_cpu_vec` copies preserve reuse eligibility. Public feedback copies use the
same cache; retained results and feedback always prevent reuse.

`SessionOptions::output_placement` sets a default location and per-output
overrides keyed by public LEAPP names at construction. `run` accepts only inputs.
New sessions inherit this placement, and every output uses its exact requested location.
Initial state, default inputs, and private feedback stay on the execution device.
Public outputs retain their requested location, including outputs also used as
feedback.
Execution errors expose `code()`, `phase()`, and their source chain.

A runnable example uses the generated policy fixture:

```sh
cargo run --locked --example run -- target/triton-fixtures/policy.yaml
```

## Model support

Both engines support ONNX, TorchScript, and PT2 on CPU or CUDA, including
mixed-backend graphs. Triton additionally supports TensorRT on CUDA.
Unsupported backends fail when opening a session; bundle inspection still accepts
custom backend names. Models must be exported before use; the runtime does not
compile them.

ONNX ports are matched by name, independently of bundle declaration order.

### Native engine

Native sessions share loaded models and weights, with private reusable workspace
buffers. Concrete-shaped models reuse fixed addresses and prepared bindings where
supported. Returned results own storage independent of the workspace.

Peers may run concurrently, but TorchScript, NativeRT, and ONNX models with
custom operators serialize execution across peers. Nodes within one graph
execute sequentially. Each peer requires memory for its workspace and execution
resources. See the [installation guide](src/engines/native/backends/INSTALLATION.md)
for backend setup, and the
[plugin contract](src/engines/native/backends/README.md) to contribute a backend.

### PT2 execution

PT2 uses raw `torch.export.save` archives with tensor-only inputs and outputs in
exported-program order. AOTInductor and CUDA Triton delegates are unsupported.
The native engine places the graph and weights on the selected device; for the
Triton engine, archive devices must match the selected model instance.

### TorchScript execution

Scripted and traced models accept positional tensor inputs and return a tensor or
flat tuple of tensors. The native engine also accepts a flat tensor list.
Bundle port order determines argument order; public LEAPP names remain unchanged.
Model-internal mutations are shared between peers and are not reset by `reset()`.

### TensorRT engines

Triton accepts prebuilt `.plan` and `.engine` artifacts. Port names, shapes,
and dtypes must describe the engine's I/O. Models execute on CUDA; inputs and
outputs may use CPU, pinned-host, or CUDA memory. Engines must be compatible with
the deployed TensorRT version and GPU.

Triton uses profile 0; its shape-tensor and additional-profile configuration
are not exposed. External custom-plugin configuration is unsupported.

### Warp operators

Embedded Warp operators in ONNX/PT2 bundles use optional adapters installed
beside their Triton backends. See
[Triton adapter installation](src/engines/triton/backends/README.md#warp-operators)
for SDK requirements and installation.
For contributor layout and validation commands, see [AGENTS.md](AGENTS.md).
