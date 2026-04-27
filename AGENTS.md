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
  - Useful for debugging node boundaries, names, and pipeline wiring before expensive export.
- `leapp.start(..., non_traced=["node_a", "node_b"])`:
  - Selectively disables tracing/export for the listed nodes while still registering them in the pipeline.
  - Those nodes still capture inputs/outputs, contribute to graph connectivity, and appear in YAML.
- `leapp.compile_graph(visualize=True)`:
  - `True` emits `<graph_name>.png` graph visualization.
  - `False` is faster for CI/headless runs when the image is not needed.
- `leapp.compile_graph(dry_run=True)`:
  - Declares dry-run at compile time after normal tracing has already happened.
  - Keeps traced FX graphs and YAML generation, but skips compile/save/validate of exported artifacts.
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
  - Any later calls in re-entry loops are validation/tag-update behavior, not a second independent output declaration.


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
- Recommended default:
  - Start with `export_with="jit"` (`jit-script`) for fastest bring-up.
- ONNX backend differences:
  - `onnx-dynamo`:
    - Modern/default ONNX path.
    - Best first choice for non-recurrent models and typical feedforward pipelines.
  - `onnx-torchscript`:
    - TorchScript-based ONNX export path.
    - Prefer for recurrent models (notably `nn.GRU`/`nn.LSTM`) when dynamo export can produce problematic graphs.
- No-export / BYO-model option:
  - `export_with=None` uses `NoneExportBackend` (no compilation/export for that node by default).
  - You can still supply your own artifact via `backend_params={"model_path": ".../model.pt"}` or `...onnx`.
  - Optional `copy_original_model=True` in `backend_params` copies the provided model into the graph output directory.
- Selective non-tracing:
  - `non_traced=[...]` is the preferred public API when only some nodes should behave like placeholder / metadata-only nodes.
  - Those nodes effectively force `export_with=None` while keeping I/O capture and graph edges.
- Additional explicit names supported: `jit-script`, `jit-trace`, `onnx-dynamo`, `onnx-torchscript`.

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
- Prefer `non_traced=[...]` over global `dry_run=True` when only specific nodes should be metadata-only.
- Prefer one node at a time while tracing:
  - complete `output_tensors()` for a node before starting another traced context.
- Handle copied tensors:
  - if data is copied in non-standard ways, call `annotate.mirror_leapp_tags(source, target)`.
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

