<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Warp support

LEAPP can trace NVIDIA Warp operations as part of a pipeline node. Instead of
converting each Warp kernel into PyTorch operations, LEAPP captures a sequence
of Warp calls as one `leapp::warp_runner` operation. The captured Warp program
and its compiled modules are bundled with the exported node.

Warp tracing is optional. Installing or importing `warp-lang` does not affect
LEAPP's PyTorch and NumPy tracing unless Warp operations are performed on
values that belong to an active LEAPP node.

## Requirements

Warp APIC with LEAPP currently requires:

- Linux
- A CUDA-capable GPU
- `warp-lang>=1.15.0`
- `leapp.start(..., global_patching=True)`, which is the default

Use the repository lock file when setting up a local test environment. The lock
keeps the CUDA/CUPTI/ONNX Runtime pieces on versions that are known to work
together for Warp tracing.

```bash
uv sync --extra test
```

This installs local LEAPP plus the locked test dependencies, including Warp.
Do not install the latest `cupti-python` or `onnxruntime-gpu` independently
when validating this feature; mismatched CUDA or ONNX Runtime API versions can
fail at runtime.

The committed lock currently uses:

- `cupti-python==12.8.0`
- `onnxruntime==1.26.0`
- `warp-lang>=1.15.0`

If you need to validate the exported ONNX model, install the GPU build of ONNX
Runtime at the same API version as the locked CPU package:

```bash
uv pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
```

### Build and configure the ONNX runner

Exported ONNX models containing `com.nvidia.warp::WrpRunner` require LEAPP's
ONNX Runtime custom operator. Build it from the repository root with CMake,
using the same Python environment created by `uv sync`. Rebuild this library
after changing the `uv.lock` or ONNX Runtime version; the custom op must be
compiled against the same ONNX Runtime API version that will load it.

```bash
RUNTIME_DIR="leapp/leapp_graph/custom_operator_registry/warp_operator/runtime"

cmake \
    -S "${RUNTIME_DIR}" \
    -B "${RUNTIME_DIR}/build" \
    -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "${RUNTIME_DIR}/build" --config Release -j
```

The build requires a C++17 compiler, CMake 3.18 or newer, and the CUDA Toolkit.
CMake locates Warp and ONNX Runtime through the selected Python interpreter
and downloads the matching ONNX Runtime C API headers.

Point LEAPP at the resulting shared library before loading or running an
exported ONNX model:

```bash
export LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY="$(pwd)/leapp/leapp_graph/custom_operator_registry/warp_operator/runtime/build/libleapp_wrp_onnx_custom_op.so"
```

The runner uses ONNX Runtime's CUDA execution provider, so a working
`onnxruntime-gpu` installation and its compatible CUDA dependencies must be
available in the inference environment.

## How it works

LEAPP uses two executions of each node that contains Warp work:

1. **Discovery pass:** LEAPP records the Warp calls, inputs, outputs, and
   segment boundaries.
2. **Capture pass:** the same code executes again. LEAPP validates that it
   follows the discovered call sequence and captures each segment with Warp
   APIC.

The node body must therefore run twice before `leapp.stop()`. Both executions
must take the same Warp control-flow path and invoke the same Warp operations
in the same order. `leapp.compile_graph()` reports an error if a discovered
segment has not completed its second capture pass.

Each captured segment becomes a `leapp::warp_runner` node in the exported
model. LEAPP stores the Warp APIC archive, input/output metadata, shapes, and
dtypes with that operation. Warp graphs currently support the ONNX Dynamo and
ExportedProgram (`pt2`) export paths.

For automatic detection, LEAPP wraps public Warp API calls and observes CUDA
activity that may require the current segment to close.

Warp nodes must currently be exported with ONNX. Use `export_with="onnx"` when
calling `annotate.output_tensors()` for nodes that contain Warp operations;
other export backends are not supported for Warp graphs yet.

## Explicit `warp_op` annotation

Use `annotate.warp_op(node_name)` when the intended Warp region has a clear
scope. All Warp calls inside the context manager belong to one segment.

```python
import leapp
from leapp import annotate
import warp as wp


@wp.kernel
def add_one(values: wp.array(dtype=wp.float32)):
    index = wp.tid()
    values[index] += 1.0


source = wp.array(
    [1.0, 2.0, 3.0],
    dtype=wp.float32,
    device="cuda",
)

leapp.start(name="explicit_warp")

for _ in range(2):  # discovery, then APIC capture
    values = annotate.input_tensors("warp_node", {"values": source})

    with annotate.warp_op("warp_node"):
        output = wp.empty_like(values)
        wp.copy(output, values)
        wp.launch(
            add_one,
            dim=output.size,
            inputs=[output],
            device=output.device,
        )

    annotate.output_tensors(
        "warp_node",
        {"output": output},
        export_with="onnx",
    )

leapp.stop()
leapp.compile_graph()
```

The explicit context is useful when application structure already defines the
correct segment boundary. Avoid synchronization, host readback, or unrelated
CUDA work inside the context: the capture pass opens a live APIC capture for
the body of the context manager.

## Automatic segment detection

With global patching enabled, LEAPP observes public `warp.*` calls. A Warp call
automatically opens a segment when it receives a traced Warp value produced by
`annotate.input_tensors()` or by an existing traced Warp operation. Additional
calls on values from the same LEAPP node continue that segment.

No `warp_op` annotation is needed:

```python
import leapp
from leapp import annotate
import warp as wp


@wp.kernel
def add_scalar(
    values: wp.array(dtype=wp.float32),
    amount: wp.float32,
    output: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    output[index] = values[index] + amount


def launch_add(values, amount):
    output = wp.empty_like(values)
    wp.launch(
        add_scalar,
        dim=values.size,
        inputs=[values, wp.float32(amount)],
        outputs=[output],
        device=values.device,
    )
    return output


source = wp.array(
    [1.0, 2.0, 3.0],
    dtype=wp.float32,
    device="cuda",
)

leapp.start(name="automatic_warp")

for _ in range(2):  # discovery, then APIC capture
    values = annotate.input_tensors("warp_node", {"values": source})

    values = launch_add(values, 1.0)  # opens automatic segment 0
    wp.synchronize_device(values.device)  # closes segment 0
    output = launch_add(values, 2.0)  # opens automatic segment 1

    annotate.output_tensors(
        "warp_node",
        {"output": output},
        export_with="onnx",
    )

leapp.stop()
leapp.compile_graph()
```

In this example, the explicit device synchronization is a hard boundary, so
LEAPP emits two Warp runner operations. Conversion and readback operations
such as `wp.to_torch()`, `wp.from_torch()`, `wp.array.numpy()`, and
`wp.from_numpy()` can also create boundaries as LEAPP propagates values between
Warp, PyTorch, and NumPy.

Automatic routing follows these rules:

- Calls using traced Warp values from the same LEAPP node stay in the active
  segment.
- A call using values from another LEAPP node closes an unowned automatic
  segment and starts a segment for the new node.
- A single call that mixes traced values from multiple LEAPP nodes is an
  error.
- Calls without traced Warp values do not start a LEAPP Warp segment.

## Current limitations

- **Linux only:** Warp tracing is currently available only on Linux. Other
  LEAPP tracing backends remain usable on Windows.
- **CUDA only:** CPU Warp segments are not currently supported by LEAPP's
  exported Warp runtime.
- **Two identical passes:** discovery and capture must encounter the same Warp
  regions, calls, and boundaries in the same order.
- **Homogeneous boundary dtypes:** primitive scalar and densely packed,
  homogeneous compound dtypes such as vectors (`wp.vec3`), quaternions, and
  matrices can cross Warp region and LEAPP node boundaries. Node and runtime
  metadata use the expanded scalar Torch layout, for example `(B, N, 3)` and
  `float32` for a logical `(B, N)` `wp.vec3` array. Heterogeneous `@wp.struct`
  arrays are not supported.
- **Conversion inside an open segment:** converting a newly written Warp
  output with `wp.to_torch()` before its segment closes can retain a stale
  tracing proxy. Close the segment before converting the output.
- **Explicit capture restrictions:** synchronization, CUDA-to-host readback,
  printing CUDA arrays, and unrelated CUDA work should not occur inside an
  explicit `warp_op` block.
- **Mixed node ownership:** one Warp call cannot consume traced arrays owned
  by different LEAPP nodes.
- **Export backends:** Warp runner graphs currently require ONNX Dynamo or
  ExportedProgram (`pt2`); other LEAPP export backends reject graphs containing
  Warp segments.
- **Annotation arguments:** `annotate.warp_op()` currently uses only the node
  name; its `inputs`, `outputs`, and additional keyword arguments are not wired
  into Warp capture.
- **Runtime integration:** running an exported ONNX model requires LEAPP's
  Warp ONNX Runtime custom operator library.

When automatic detection cannot express a desired boundary safely, use an
explicit `annotate.warp_op()` block and keep the block limited to replayable
Warp operations.

## Try an inference pass

After exporting a Warp graph and configuring `LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY`,
run a full-pipeline smoke test with `InferenceManager`. The general workflow is
covered in the LEAPP [runtime and validation guide](docs/source/guides/runtime.rst)
and in the published docs at
[Runtime and validation](https://nvidia-isaac.github.io/leapp/guides/runtime.html).

```python
from leapp import InferenceManager

manager = InferenceManager("explicit_warp/explicit_warp.yaml")

print(manager.inputs)
print(manager.outputs)

mock_inputs = manager.get_mock_input()
outputs = manager.run_policy(mock_inputs)

print(outputs)
```

For Warp ONNX exports, run this in the same environment used to build the ONNX
custom op. If the custom op is missing, built against a different ONNX Runtime
version, or loaded without ONNX Runtime's CUDA execution provider, the smoke
test will fail before the Warp runner can execute.
