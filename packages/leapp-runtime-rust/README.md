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

The core is distributed as the `leapp-runtime` Cargo crate and Python package.
C++ consumers build the source with CMake and link `lrt::runtime`. Pixi provisions
the selected backend libraries independently; see the
[packaging guide](packaging/README.md). Backend recipes currently produce local
channels. Until the first registry release, use the source-build commands below.

For a published Rust release, add `leapp-runtime = "0.1"` to `[dependencies]`.
Python users install `leapp-runtime` from PyPI and import `lrt`. Core packages
include both orchestration engines and require no model SDKs at installation.

Build on Linux with Rust 1.88 or newer. Building the core runtime does not
require the Triton or CUDA SDKs.

```sh
cargo build --locked --release
```

The Cargo package is `leapp-runtime`; its Rust import name is `lrt`.
See the [Python bindings](bindings/python/README.md) for `import lrt` and the
[C++ bindings](bindings/cxx/README.md) for the `lrt` namespace and `lrt::runtime`
CMake target.

Native C++ backend authors can start with
[`BackendPlugin`](src/engines/native/backends/include/leapp/native/backend_plugin.h) and the
[backend contribution guide](src/engines/native/backends/README.md#adding-a-backend).

Install only the model runtimes and plugins required by your bundle and selected
engine; see [model support](#model-support). They load when a session opens;
bundle inspection does not load them. Cargo does not download or install these
runtimes. Installing a supported backend does not require rebuilding the core
binary or selecting Cargo features. Running exported bundles does not require Python
model-export packages.

For native execution, follow the
[native backend installation guide](src/engines/native/backends/README.md#installation).
For Triton, use its ONNX Runtime and TensorRT backends and the
[Torch/Warp plugins](src/engines/triton/backends/README.md).
`LEAPP_TRITON_BACKENDS` defaults to `$TRITON_ROOT/backends` or
`/opt/tritonserver/backends`.

Triton loads only when selected. CUDA loads when GPU execution, transfers,
pinned-memory allocation, or supplied CUDA streams require it. Native CPU ONNX
execution with CPU inputs and outputs needs only a CPU-compatible ONNX Runtime.
The current Torch plugins, TensorRT, and Warp still require their CUDA libraries.

Set `TRITON_ROOT` or `CUDA_ROOT` to select an explicit installation. Otherwise,
the runtime uses the system loader, then `/opt/tritonserver` or `/usr/local/cuda`.
Set installation variables before starting the process; loaded libraries remain
in use for its lifetime. Backend dependencies must also be discoverable through
`LD_LIBRARY_PATH` or the system loader. Load errors include the library and cause;
the runtime does not fall back to another engine or device.

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
them. Copying and first exposure are synchronized. An existing `BorrowedTensor` can also be read
without copying through its safe `as_cpu_slice` method when its storage is on the host.
Native backends that bind inputs directly require them on the execution device.
Native dynamic calls also borrow session inputs directly and require them on the
execution device.
Output placement does not change input requirements.

Array libraries and CUDA callers can use standard DLPack descriptors through
`unsafe { BorrowedTensor::new(&descriptor) }`. They must retain the descriptor,
shape/strides, and ready, immutable storage through the synchronous call. CUDA
producer work must already be complete. Outputs expose their descriptor through
`descriptor()` or transfer a fresh managed lease with `export(true).into_raw()`;
the receiving consumer must invoke that lease's deleter exactly once.

See `examples/run.rs` for a complete CPU example.

`SessionOptions` configures execution and output placement. Defaults use CPU
execution and CPU outputs. Native is selected by default:

```rust
use lrt::{Session, SessionOptions};

let mut session = Session::open(&bundle, SessionOptions::default())?;
```

Set `device` to `ExecutionDevice::Cuda { device_id: 0 }` for GPU execution.
Select Triton with `SessionOptions { engine: EngineKind::Triton, ..Default::default() }`,
Python `SessionOptions(engine="triton")`, or C++ `EngineKind::kTriton`.
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

`run(inputs)` returns freshly allocated outputs. Use `run_into(inputs, &mut outputs)`
with named `dlpack::OutputBuffer` values to write into caller-owned storage.
Native backends bind these buffers directly. Triton copies ensemble results into
them; supplying buffers does not eliminate Triton's intermediate output storage.
`OutputBuffer::cpu` borrows a mutable CPU slice; other allocations use its unsafe
DLPack constructor. Supplied buffers must match the output dtype and shape, meet
backend device/alignment requirements, and be disjoint from inputs and each other.
Only outputs without supplied buffers appear in the returned `RunResult`.
A backend may reject a data-dependent output buffer if it requires more capacity
than the tensor's final shape.
On failure, supplied buffer contents are unspecified; feedback does not advance.

`SessionOptions::output_placement` sets a default location and per-output
overrides keyed by public LEAPP names at construction.
New sessions inherit this placement. Allocated outputs use the requested location;
supplied buffers determine their own storage location.
Initial state, default inputs, and private feedback stay on the execution device.
Public outputs retain their requested location, including outputs also used as
feedback.
Execution errors expose `code()`, `phase()`, and their source chain.

A runnable example uses the generated policy fixture:

```sh
cargo run --locked --example run -- target/triton-fixtures/policy.yaml
```

### Caller-owned CUDA streams

Rust callers can set `SessionOptions.cuda_streams` to a map from CUDA device ID
to `CudaStream`. Construct each borrowed handle with
`unsafe { CudaStream::from_raw(raw_stream) }` from a `cudaStream_t` or `CUstream`.
Native execution and transfers use these streams; missing entries use internal
streams. Triton ignores the option. Python and C++ do not yet expose it.

Keep each stream and its primary context alive until the session and all peers
are closed or destroyed. Initialization validates stream/device compatibility.
Null and `cudaStreamLegacy` select the legacy default stream;
`cudaStreamPerThread` is unsupported because sessions may move between threads.
Inputs must already be ready, and `run()` completes before returning.

Peers inherit supplied streams. ONNX peers share their configured compute stream
and its ordering. Backends may use auxiliary streams with synchronization.

## Model support

The engines support the following exported artifacts and mixed-backend graphs.
Unsupported backends fail when opening a session; bundle inspection still accepts
custom backend names. Models must be exported before use; the runtime does not
compile them.

| Model | Bundle backend | Execution device | Required model runtime |
| --- | --- | --- | --- |
| ONNX | `onnx` | CPU or CUDA | ONNX Runtime; CUDA providers and cuDNN for GPU |
| TorchScript | `jit`, `jit-script`, `jit-trace` | CPU or CUDA | CUDA-enabled LibTorch |
| Exported program | `pt2` | CPU or CUDA | CUDA-enabled LibTorch and matching NativeRT |
| TensorRT engine (Triton only) | `trt`, `tensorrt` | CUDA | TensorRT compatible with the engine and GPU |

ONNX ports are matched by name, independently of bundle declaration order.

### Native engine

Native sessions share loaded models and weights, with private reusable workspace
buffers. Eligible inputs are borrowed directly on ONNX, TorchScript, and NativeRT
(CPU/GPU). Copies remain where needed to protect caller inputs,
isolate feedback, or satisfy output placement. Returned results own storage
independent of the workspace.

Peers may run concurrently, but TorchScript, NativeRT, and Warp models
serialize execution across peers. Nodes within one graph
execute sequentially. Each peer requires memory for its workspace and execution
resources. See the [installation guide](src/engines/native/backends/README.md#installation)
for backend setup, and the
[contributor notes](src/engines/native/backends/README.md#adding-a-backend) to add a backend.

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

### ONNX TensorRT acceleration

Both engines accept per-node TensorRT accelerator parameters through
`SessionOptions.onnx_tensorrt`. Install the optional
[`leapp-onnx-tensorrt` package](packaging/README.md), or an ONNX Runtime TensorRT
provider with its TensorRT dependencies beside the ONNX Runtime library.

```python
options = lrt.SessionOptions(
    device="cuda:0",
    onnx_tensorrt={"policy": {
        "precision_mode": "FP16",
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": "/var/cache/my-app/policy",
    }},
)
```

Rust and C++ expose the same node-to-parameter maps. An entry enables TensorRT;
empty parameters use SDK defaults. Unlisted nodes retain their existing execution.
Parameters follow [Triton's ONNX Runtime accelerator options](https://github.com/triton-inference-server/onnxruntime_backend/blob/r26.05/README.md#onnx-runtime-with-tensorrt-optimization),
including precision, workspace, caching, and min/opt/max profile strings.
Native execution forwards additional options directly to ORT. Triton forwards
options to its unmodified ONNX Runtime backend. Each backend validates keys and
values and rejects unsupported options. ORT-only options such as
`trt_op_types_to_exclude` require native execution and a supporting ORT version.
The native engine rejects CUDA graph capture (`trt_cuda_graph_enable`) because
borrowed runs and peer sessions can use different input and output addresses.
ONNX Runtime executes supported subgraphs with TensorRT and uses CUDA/CPU fallback
for other operations. Compilation and cache management follow ONNX Runtime's
behavior; dynamic shapes can trigger compilation on the first inference.

### TensorRT engines

The Triton engine accepts prebuilt `.plan` and `.engine` artifacts. The native
engine does not support TensorRT engines. Port names, shapes, and dtypes must
match the engine's I/O. Triton executes them on CUDA; inputs and outputs may use
CPU, pinned-host, or CUDA memory. Engines must be compatible with the deployed
TensorRT version and GPU. Triton uses optimization profile 0; shape-tensor,
additional-profile, and external custom-plugin configuration are not exposed.

### Native Warp support

Both engines support embedded Warp operators in ONNX and PT2 through optional
adapters. Replay requires CUDA device 0 even for CPU model placement. Warm each
model serially before concurrent use. These adapters do not support Warp operators
in TorchScript or TensorRT artifacts.

See [native adapter installation](src/engines/native/backends/README.md#warp-operators)
or [Triton adapter installation](src/engines/triton/backends/README.md#warp-operators).
For contributor layout and validation commands, see [AGENTS.md](AGENTS.md).
