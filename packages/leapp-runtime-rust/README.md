<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Rust runtime

This package targets Linux and executes exported LEAPP bundles through embedded
Triton 2.69.0. All builds require the Triton and CUDA runtime SDKs. Rust, Python,
and C++ expose bundle inspection and synchronous execution.

## Installation

Build on Linux with Rust 1.88 or newer, the Triton 2.69.0 SDK, and the CUDA
runtime SDK. Set `TRITON_ROOT` and `CUDA_ROOT` if they are not installed under
`/opt/tritonserver` and `/usr/local/cuda`. Make their shared libraries and the
selected model backends available through `LD_LIBRARY_PATH`.

The Cargo package is `leapp-runtime`; its Rust import name is `lrt`.
See the [Python bindings](bindings/python/README.md) for `import lrt` and the
[C++ bindings](bindings/cxx/README.md) for the `lrt` namespace and `lrt::runtime`
CMake target. CPU execution still requires the CUDA runtime SDK.

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
execution and CPU outputs. Triton is selected automatically:

```rust
use lrt::{Session, SessionOptions};

let mut session = Session::open(&bundle, SessionOptions::default())?;
```

Set `device` to `ExecutionDevice::Cuda { device_id: 0 }` for GPU execution.
`LEAPP_TRITON_BACKENDS` selects the backend directory when opening a session,
falling back to `$TRITON_ROOT/backends` or `/opt/tritonserver/backends`.
Set installation variables before starting the process. `new_session()` shares
the already-loaded models without rediscovering them.
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
or another consumer reads it.

Each session owns its output cache. `SessionOptions::output_pool.max_entries`
limits cached allocations (default 64); zero disables caching while inference
continues to allocate normally. The byte budget is the bundle's largest static
output size multiplied by this entry limit, and grows when execution produces a
larger dynamic output. There is no fixed byte cap or eager allocation. This policy
limits cache retention, not memory held by models, live outputs, or feedback.
`new_session()` inherits the policy with an empty cache; close releases cached
references while retained outputs stay valid.

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

Session setup validates graph routing and converts the bundle to a temporary
Triton model repository. The loaded Triton version must be 2.69.0. ONNX models
use Triton's ONNX Runtime backend. Unsupported backends fail when opening a
session; bundle inspection still accepts custom backend names.

## PT2 execution

The runtime configures PT2 artifacts for an externally installed `leapp_pt2`
Triton backend. Set `LEAPP_TRITON_BACKENDS` to a directory containing
`leapp_pt2/libtriton_leapp_pt2.so` and any other backends used by the bundle.
Backend implementations and their dependencies are installed separately. See
the [plugin build guide](src/engines/triton/backends/README.md) to build `leapp_pt2` from source.

PT2 inputs and outputs follow exported-program order. Archive devices must match
the selected model instance. The runtime supplies feedback, default inputs, and
output placement through the same session API used for ONNX models.

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

Engines must match the deployed TensorRT version and GPU. The runtime does not
compile models. See [AGENTS.md](AGENTS.md) for contributor layout and validation
commands.
