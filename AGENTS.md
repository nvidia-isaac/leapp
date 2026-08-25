<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP Agent Playbook

This file teaches coding agents how to apply LEAPP quickly in user projects.

## What LEAPP is for

LEAPP traces PyTorch computations into a graph of named nodes, then exports:

- per-node models (`.pt` or `.onnx`)
- a pipeline spec (`<graph_name>.yaml`)
- optional graph visualization (`<graph_name>.png`)

Primary goal: export complex pipelines with small annotation inserts and no functional code rewrites (unless absolutely needed for tracing/export edge cases).

## Repository architecture and ownership

When changing LEAPP internals, place behavior with the component that owns its
meaning rather than the component that happens to observe it:

This is the current ownership model. Warp array construction remains in
`warp/patching.py` because interception begins before the carrier is initialized.

```text
TracedData
├── TracedTensor   owns Torch value semantics and Torch dispatch
├── TracedNpArray  owns NumPy value semantics and NumPy dispatch
└── TracedWpArray  owns Warp value semantics and simulated Warp dispatch

backend patcher    owns interception and global guards
trace session      owns capture/segment lifecycle
node/graph layer   owns ports, edges, compilation, and export
```

Use these rules when deciding where code belongs:

- Put backend-neutral tracing state and operations in
  `leapp/leapp_graph/datatypes/traced_data.py`: proxy/name/context/output-port
  state, shared structure traversal, context validation, and common port or
  assignment policy. Lift a helper here only when at least two datatype
  backends share its complete contract.
- Put native value behavior on its carrier:
  `torch/traced_tensor.py`, `numpy/traced_np_array.py`, or
  `warp/traced_wp_array.py`. Finding an anchor, unwrapping native values,
  wrapping or promoting results, alias decisions, and datatype-specific port
  propagation are carrier responsibilities.
- Keep patch backends focused on interception concerns that exist outside a
  particular value: installing/restoring wrappers, imported-alias discovery,
  re-entrancy guards, and global boundary detection. Do not leave value
  semantics in a patcher merely because the patcher detected the call.
- Keep lifecycle in its lifecycle owner. Warp segment creation, ownership,
  discovery/capture matching, and close policy belong to `warp/session.py`,
  `warp_op.py`, and `warp/warp_segment.py`, not the carrier.
- Keep node ports, graph connectivity, compilation, and export behavior in
  `traced_node.py`, `leapp_graph.py`, and the backend/export layers. Datatype
  carriers may preserve or clear an existing port, but they do not invent
  graph edges.
- Keep cross-backend conversion logic with the backend function being invoked.
  For example, `wp.from_torch` and `wp.to_torch` are Warp boundaries and belong
  to Warp dispatch, while shared alias checks remain in `proxy_view.py`.

Warp has two explicit exceptions that should remain visible:

1. Warp has no native equivalent of `__torch_function__`, so
   `WarpPatchBackend` intercepts public Warp calls and delegates their value
   semantics to the simulated `TracedWpArray.__warp_function__` protocol.
2. Warp currently rejects array subclasses at launch-time boundaries, so
   `TracedWpArray` must provide an exact, non-owning raw `wp.array` alias before
   calling Warp. Do not remove this normalization until upstream Warp accepts
   subclasses.

Useful paths:

| Area | Location |
| --- | --- |
| Shared traced-data state and policy | `leapp/leapp_graph/datatypes/traced_data.py` |
| Shared alias / `ProxyView` logic | `leapp/leapp_graph/datatypes/proxy_view.py` |
| Torch carrier and patching | `leapp/leapp_graph/datatypes/torch/` |
| NumPy carrier and patching | `leapp/leapp_graph/datatypes/numpy/` |
| Warp carrier, patching, and session | `leapp/leapp_graph/datatypes/warp/` |
| Warp operation and APIC segment integration | `leapp/leapp_graph/warp_op.py` |
| Node I/O and FX compilation | `leapp/leapp_graph/traced_node.py` |
| Pipeline connectivity | `leapp/leapp_graph/leapp_graph.py` |
| Public annotation entry points | `leapp/export_manager.py` and `leapp/leapp.py` |

Prefer conceptual symmetry across Torch, NumPy, and Warp over forced identical
code. Preserve real backend differences explicitly, and after moving a
responsibility, delete the old implementation instead of leaving forwarding
twins.

## Core workflow (always in this order)

```python
import leapp
from leapp import annotate

leapp.start(name="my_graph", save_path=".")
# ... trace nodes ...
leapp.stop()
leapp.compile_graph(visualize=True, validate=True)
```

Do not call `compile_graph()` before `stop()`.
Use `leapp.start()`, `leapp.stop()`, and `leapp.compile_graph()` for graph lifecycle control.
Use `annotate` only for annotation APIs such as `method()`, `input_tensors()`, and `output_tensors()`.

## Optional runtime/export settings (important knobs)

Use these to control tracing cost, validation coverage, and output artifacts.

- `leapp.start(..., max_cached_io=N)`:
  - Controls how many re-entry I/O examples LEAPP caches per node for multi-example validation.
  - Higher values improve confidence for looped/stateful pipelines, but increase memory/time.
  - Practical default: keep `N` small (`3-5`) unless user explicitly wants stronger replay coverage.
- `leapp.compile_graph(validate=True, rtol=..., atol=..., strict=True)`:
  - `validate=True` compares exported model outputs against traced outputs.
  - Tune `rtol`/`atol` if expected numeric drift exists (especially ONNX/cross-device).
  - Use `validate=False` only for rapid iteration or when user asks for speed over checks.
- `leapp.start(..., dry_run=True)`:
  - Skips real model compilation/export, but still traces graph structure.
  - Only available on `start()`, not on `compile_graph()`.
  - Useful for debugging node boundaries, names, and pipeline wiring before expensive export.
- `leapp.start(..., non_traced=["node_a", "node_b"])`:
  - Selectively disables export for the listed nodes while still registering them in the pipeline.
  - Those nodes are still traced, capture inputs/outputs, contribute to graph connectivity, and appear in YAML.
- `leapp.compile_graph(visualize=True)`:
  - `True` emits `<graph_name>.png` graph visualization.
  - `False` is faster for CI/headless runs when the image is not needed.
- Also useful:
  - `leapp.start(..., verbose=True)` for detailed trace logs, including FX graph dumps for traced nodes.
  - `leapp.start(..., global_patching=False)` if numpy-related patching causes environment issues.


## Critical node declaration rule

For `TracedTensorNode` workflows (`input_tensors` / `output_tensors`), agents must follow this exactly:

- `annotate.input_tensors("node_name", ...)` can be called multiple times for the same node.
  - This is valid across helper functions, class methods, and even different files, as long as it is the same active trace and same node name.
  - Use this to accumulate/declare node inputs wherever they naturally appear in the code.
  - For raw tensors, always pass a top-level dict of named tensors. Bare tensors are not supported; use `TensorSemantics(...)` if you want a single named semantic input without a dict.
- `annotate.output_tensors("node_name", ...)` is the node finalization declaration and should be done once for the initial trace of that node.
  - After this, the node is compiled/finalized.
  - Any later calls in re-entry loops are validation/source-update behavior, not a second independent output declaration.


Example:

```python
leapp.start(
    name="my_graph",
    max_cached_io=5,
    dry_run=False,
    verbose=True,
)
# ... trace ...
leapp.stop()
leapp.compile_graph(
    visualize=True,
    validate=True,
    rtol=1e-3,
    atol=1e-5,
    strict=True,
)
```

## Semantic info injection (for downstream frameworks)

Use semantic metadata when deployers need tensors to carry meaning (not just shape/dtype).
LEAPP supports semantic injection via `TensorSemantics` wrappers passed to `input_tensors()` / `output_tensors()`.

- Current supported semantic fields:
  - `kind`: high-level semantic role string/enum for a tensor.
  - `element_names`: per-element labels (for vector/joint/channel interpretability).
- For `kind`, LEAPP provides two semantic enum families:
  - `InputKindEnum` for input/state/command-like inputs.
  - `OutputKindEnum` for output/target/control-like outputs.
- Output location:
  - semantic fields are serialized into the generated YAML tensor entries.
- Input format rules:
  - pass a single `TensorSemantics` or a list of `TensorSemantics`.
  - do not wrap `TensorSemantics` in dicts, and do not mix raw tensors with `TensorSemantics` in the same list.

Example:

```python
import torch
import leapp
from leapp import annotate, TensorSemantics, InputKindEnum, OutputKindEnum

leapp.start("semantic_graph")

annotate.input_tensors("policy", [
    TensorSemantics("joint_pos", torch.randn(12), kind=InputKindEnum.JOINT_POSITION),
    TensorSemantics("joint_vel", torch.randn(12), kind=InputKindEnum.JOINT_VELOCITY),
])

action = torch.randn(12)
annotate.output_tensors("policy", [
    TensorSemantics(
        "torques",
        action,
        kind=OutputKindEnum.JOINT_TORQUES,
        element_names=[f"joint_{i}" for i in range(12)],
    )
], export_with="jit")

leapp.stop()
leapp.compile_graph(validate=True)
```

## API chooser (how to pick the right LEAPP method)

Use this decision table:

- `@annotate.method(...)`:
  - Best for self-contained Python functions where arguments/returns naturally define node I/O.
  - Easiest entry point for agents.
  - `node_name` is optional; by default LEAPP uses the function name. Set `node_name` only when you want to override it.
- `annotate.input_tensors(...)` + `annotate.output_tensors(...)`:
  - Best when node logic spans multiple helper functions, branches, or dynamic code.
  - Most flexible and most reliable for complex flows.
- `annotate.module(node_name, model, buffer_names=None)`:
  - Use when tracing an `nn.Module` that has internal buffers/state.
  - Auto-detects reassigned buffers and turns them into feedback state.
- `annotate.state_tensors(...)` + `annotate.update_state(...)`:
  - Use for explicit recurrent/stateful inputs (e.g., hidden state, history windows).
  - Creates feedback semantics only for states explicitly passed to `update_state()`.
- `annotate.register_buffer(...)`:
  - Use for constants you want embedded in the exported model.
  - No feedback loop.
- `@annotate._method(...)`:
  - Legacy/private path. Use only if `method()` cannot express the case.

## Backends (practical defaults)

- Alias mapping (important):
  - `export_with="jit"` is an alias for `jit-script`.
  - `export_with="onnx"` is an alias for `onnx-dynamo`.
  - `export_with="pt2"` is an alias for `exported-program`.
- Recommended default:
  - Start with `export_with="jit"` (`jit-script`) for fastest bring-up.
- ONNX backend differences:
  - `onnx-dynamo`:
    - Modern/default ONNX path.
    - Best first choice for non-recurrent models and typical feedforward pipelines.
  - `onnx-torchscript`:
    - TorchScript-based ONNX export path.
    - Prefer for recurrent models (notably `nn.GRU`/`nn.LSTM`) when dynamo export can produce problematic graphs.
- ExportedProgram backend:
  - `export_with="exported-program"` (alias `pt2`) saves a `torch.export` `.pt2` artifact.
  - YAML `backend` is written as `pt2`.
  - Good when you want the modern exported-program representation instead of TorchScript or ONNX.
  - `InferenceManager` loads `.pt2` artifacts; pass runtime inputs on the model device when CUDA is available.
- No-export / BYO-model option:
  - `export_with=None` uses `NoneExportBackend` (no compilation/export for that node by default).
  - You can still supply your own artifact via `backend_params={"model_path": ".../model.pt"}` or `...onnx`.
  - Optional `copy_original_model=True` in `backend_params` copies the provided model into the graph output directory.
- Selective non-export:
  - `non_traced=[...]` is the preferred public API when only some nodes should skip export.
  - Those nodes force `export_with=None` while still tracing and keeping I/O capture and graph edges.
- Additional explicit names supported: `jit-script`, `jit-trace`, `onnx-dynamo`, `onnx-torchscript`, `exported-program`.

## Fast integration recipe for user projects

1. Identify the pipeline boundaries:
   - graph inputs (external runtime inputs)
   - intermediate stages
   - graph outputs
2. Give each stage a stable `node_name`.
3. Wrap each stage with `method()` or `input_tensors()/output_tensors()`.
4. Pick backend per stage (`jit` first unless user requests ONNX).
5. Export and validate:
   - `leapp.compile_graph(validate=True)`
6. Smoke test runtime with `InferenceManager`.

## Copy-paste patterns

### Pattern A: Minimal function-based node

```python
import torch
import leapp
from leapp import annotate

@annotate.method(export_with="jit")  # node_name defaults to function name: "preprocess"
def preprocess(obs):
    return (obs - obs.mean()) / (obs.std() + 1e-6)

def run():
    x = torch.randn(1, 32)
    leapp.start("demo_graph")
    y = preprocess(x)
    leapp.stop()
    leapp.compile_graph(visualize=True, validate=True)
```

### Pattern B: Flexible traced node (recommended for multi-step logic)

```python
import torch
import leapp
from leapp import annotate

leapp.start("demo_graph")

obs = torch.randn(1, 32)
x = annotate.input_tensors("policy", {"obs": obs})

h = torch.relu(x)
out = torch.tanh(h)

annotate.output_tensors("policy", {"action": out}, export_with="jit")
leapp.stop()
leapp.compile_graph(validate=True)
```

## Automatic feedback detection

LEAPP automatically detects feedback connections when an output from a later node is consumed by an earlier node (graph cycle).

- To detect cycles reliably, execute the loop at least twice in one trace session (`start()` ... repeated calls ... `stop()`).
- Detected feedback edges are written to `pipeline.feedback_flow` in the exported YAML.
- Initial feedback tensor values are saved and used by `InferenceManager` prepopulation.
- See `docs/3_advanced_graph.md` for the detailed feedback workflow and examples.

### Pattern C: Explicit state loop

```python
import torch
import leapp
from leapp import annotate

leapp.start("stateful_graph")

x = annotate.input_tensors("rnn_step", {"obs": torch.randn(1, 16)})
h = annotate.state_tensors("rnn_step", {"h": torch.zeros(1, 32)})

h_next = torch.tanh(torch.cat([x, h], dim=-1))[..., :32]
annotate.update_state("rnn_step", {"h": h_next})
annotate.output_tensors("rnn_step", {"policy_h": h_next}, export_with="jit")

leapp.stop()
leapp.compile_graph(validate=True)
```

### Pattern D: Track module buffers automatically

```python
import torch
import torch.nn as nn
import leapp
from leapp import annotate

class StatefulPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("h", torch.zeros(1, 32))
        self.linear = nn.Linear(16 + 32, 32)

    def forward(self, obs):
        h_next = torch.tanh(self.linear(torch.cat([obs, self.h], dim=-1)))
        self.h = h_next  # reassignment is tracked
        return h_next

model = StatefulPolicy().eval()

leapp.start("module_graph")
obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 16)})
annotate.module("policy", model)
action = model(obs)
annotate.output_tensors("policy", {"action": action}, export_with="onnx-torchscript")
leapp.stop()
leapp.compile_graph(validate=True)
```

## Runtime recipe (using exported YAML)

```python
from leapp import InferenceManager

im = InferenceManager("module_graph/module_graph.yaml")

print(im.inputs)   # expected "node/input" keys
print(im.outputs)  # produced "node/output" keys

sample_inputs = im.get_mock_input()
out = im(sample_inputs)  # same as im.run_policy(sample_inputs)
```

## High-value tips and tricks for agents

- Reuse names consistently:
  - keep `node_name`, input keys, output keys stable across traces.
- Keep the API split straight:
  - import `leapp` whenever you need `start()`, `stop()`, or `compile_graph()`.
  - do not assume `annotate` exposes lifecycle or internal manager state.
- Prefer `non_traced=[...]` over global `dry_run=True` when only specific nodes should skip export.
- Prefer one node at a time while tracing:
  - complete `output_tensors()` for a node before starting another traced context.
- Handle copied tensors:
  - a node output carries the producing node and output port that build the graph edge.
  - copies that keep the values, shape and dtype carry it automatically: torch `clone`/`detach`/`contiguous`/`cpu`/`cuda`/device-only `to`/full buffer overwrite, numpy `np.copy`/`.copy()`/`np.asanyarray`, and full-range `wp.copy`.
  - any other operation deliberately drops the port, so the next node reports a dangling input instead of a false edge.
  - for a preallocated raw `np.ndarray` destination, call `annotate.mirror_leapp_tags(source, target)` and use its return value.
- Understand state choices:
  - `state_tensors` = input+output feedback state.
  - `register_buffer` = frozen constant in exported model.
- For stateful `nn.Module`:
  - `annotate.module()` detects reassignment (`self.h = new_h`), not in-place updates (`self.h.copy_(...)`).
- Validate aggressively during integration:
  - use `compile_graph(validate=True, rtol=..., atol=...)`.
- If user pipeline has loops/re-entry:
  - run multiple iterations between `start()` and `stop()` so cached I/O paths are exercised.
- If NumPy conversion causes trace issues:
  - try `leapp.start(..., global_patching=False)` as a debugging fallback.
- With `validate=True`, intentionally `non_traced` / dry-run nodes will skip model validation because they do not have a compiled model.

## Common failure modes and fixes

- "node already exists":
  - duplicate `node_name`; rename node or avoid creating node twice.
- output tracing complains about non-traced tensors:
  - forgot to use returned values from `input_tensors()`.
- context mismatch / mixed tracing contexts:
  - tensors from one node were used in another node before finalization.
- a node reports a dangling input when you expected an edge:
  - the value reaching it was derived from a node output rather than being an equivalent copy of one, so it carries no output port. Pass the output itself, use one of the copies listed above, or mirror the state onto the value.
- stop() errors about active tracing:
  - ensure wrapped function exited and no active legacy `_method` trace.
- ONNX export fails on recurrent models:
  - switch to `export_with="onnx-torchscript"`.

## Agent execution checklist (when helping a user)

- Confirm desired export target (`jit` vs `onnx`).
- Implement annotation with explicit node and tensor names.
- Run export flow end-to-end and ensure artifacts exist.
- Run a small `InferenceManager` smoke test.
- Report:
  - graph inputs/outputs
  - generated files
  - any caveats (state handling, backend-specific constraints)

