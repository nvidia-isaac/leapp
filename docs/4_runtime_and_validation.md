# Validating Export Correctness

This guide focuses on how to validate that a LEAPP export is correct.
In practice there are three layers of confidence:

1. automatic per-node validation during `leapp.compile_graph(validate=True)`
2. replay validation across cached re-entry examples using `max_cached_io`
3. end-to-end runtime checks with `InferenceManager`

## Automatic Validation During `compile_graph()`

The main validation entry point is:

```python
leapp.compile_graph(
    validate=True,
    rtol=1e-3,
    atol=1e-5,
    strict=True,
)
```

When `validate=True`, LEAPP:
- runs each exported node model
- feeds it the captured traced inputs
- compares exported outputs against the outputs seen during tracing
- logs if any deviation is detected
- returns a dict mapping node names to validation results when validation runs

`strict=True` raises if any node fails validation.
`strict=False` still runs validation, but lets you inspect the result dict even when some nodes fail.

### What is compared

Validation happens per node, not just at the whole-graph level.
For each node, LEAPP compares:
- number of outputs
- output tensor values using `torch.allclose(..., rtol=..., atol=...)`
- NaN / Inf presence in exported and traced outputs

If a node has no compiled model, validation is skipped for that node and treated as successful.
This is expected for metadata-only nodes such as `non_traced=[...]` or dry-run cases.

## Multi-Example Validation with Cached Inputs

If a node is re-entered multiple times within the same trace session, LEAPP can cache multiple input/output examples and validate the exported model against all of them.

Enable this with:

```python
leapp.start(name="my_graph", max_cached_io=5)
```

### How cached validation works

- The first execution becomes the main traced example
- Later re-entries are validated for structural consistency and stored as cached examples
- During `compile_graph(validate=True)`, LEAPP validates the exported model against:
  - the original traced example
  - every cached example collected during re-entry

This is especially useful for:
- looped pipelines
- stateful nodes
- cases where one example is not enough to trust the export

### What gets cached

For each node, LEAPP caches:
- input values
- output values
- updated tags needed for feedback detection across re-entry

The validation log labels these examples using numeric sample indices:
- `sample 0` for the original traced example
- `sample 1`, `sample 2`, ... for cached re-entry examples

So if a later example fails while the first one passes, you can see exactly which replayed example exposed the mismatch.

## What LEAPP Reports on Validation Failure

When validation fails, LEAPP prints more than just “allclose failed”.
Depending on the failure mode, LEAPP reports different diagnostics.

### Output count mismatch

If the exported model returns a different number of outputs than the traced node, LEAPP logs:
- node name
- example label (`sample 0`, `sample 1`, ...)
- actual output count
- expected output count

### NaN / Inf analytics

If either the exported output or the traced output contains NaN or Inf values, LEAPP logs:
- which node/output failed
- whether NaNs/Infs came from the exported output or traced source output
- counts and percentages of NaN / Inf values

This is useful for quickly distinguishing export corruption from already-unstable traced outputs.

### Numeric mismatch analytics

When values differ outside `rtol` / `atol`, LEAPP logs:
- node name and output name
- example label (`sample N`)
- active `rtol` and `atol`
- shape and dtype of source and exported outputs
- source and exported value ranges
- absolute-difference statistics:
  - max
  - mean
  - percentiles `p50`, `p75`, `p90`, `p99`, `p995`
- the path to the LEAPP log file for deeper inspection

This is the main numeric-debugging signal when an export is “close but wrong”.

### Validation summary

At the end, LEAPP prints a summary like:
- passed node count
- failed node count
- node count that errored during execution

If `strict=True`, LEAPP then raises with the list of failed node names.

## Re-Entry Validation Before Export

Before final model validation even runs, LEAPP also checks that repeated executions of a node are structurally consistent.

On re-entry LEAPP validates:
- input/output names
- shape and dtype descriptions
- tags used for graph connectivity

These checks catch cases where later executions no longer match the original trace shape, dtype, or connection structure.

## Python Runtime Tooling with `InferenceManager`

After export, `InferenceManager` serves as a lightweight Python-side deployment and testing tool for exported LEAPP graphs. You can use it to smoke-test the full pipeline, validate exported artifacts end to end, and run the graph directly from Python before handing it off to a production deployer.

```python
from leapp import InferenceManager

manager = InferenceManager("my_graph/my_graph.yaml")

print(manager.inputs)
print(manager.outputs)

mock_inputs = manager.get_mock_input()
outputs = manager.run_policy(mock_inputs)
```

`InferenceManager` is useful for:
- smoke-testing the full exported pipeline from Python
- checking that the YAML, models, and graph wiring are internally consistent
- validating that runtime inputs and outputs look sane after export
- prototyping or deploying the exported graph from a Python runtime without writing a custom runner first

Runtime note for ONNX models:
- LEAPP uses `onnxruntime` by default, which is CPU-safe on all systems.
- If you want `InferenceManager` to use ONNX Runtime's CUDA execution provider, install `onnxruntime-gpu` in the environment where you run inference.
- When the CUDA execution provider is available, LEAPP will prefer it automatically for ONNX-backed nodes and can use the faster CUDA I/O binding path.

On construction it:
- loads the YAML description
- loads all referenced JIT/ONNX models
- validates pipeline connection shape/dtype compatibility
- preallocates node input buffers
- prepopulates feedback inputs from `pipeline.initial_values` when present

### Feedback-state checks

For graphs with feedback:
- feedback inputs are auto-initialized from the exported safetensors file when available
- you can inspect feedback targets via `manager.feedback_inputs`
- you can manually override any feedback input with `set_input_value(...)`

```python
manager = InferenceManager("my_graph/my_graph.yaml")
manager.set_input_value("stateful_node", "h", torch.zeros(1, 32))
```

## Recommended Validation Workflow

For high-confidence exports, a practical workflow is:

1. Trace representative executions, not just one trivial example
2. Increase `max_cached_io` when nodes are re-entered or stateful
3. Run `leapp.compile_graph(validate=True, strict=True)`
4. If needed, relax `rtol` / `atol` only when the mismatch is expected numerical drift
5. Run a quick `InferenceManager` smoke test on the exported YAML
6. If desired, keep using `InferenceManager` as a simple Python deployment/runtime wrapper for the exported graph

## When Validation Fails

Use the failure signal to decide what to inspect next:

- If only `cached[i]` fails, your export may not generalize across re-entry examples. If you missed some variable inputs, LEAPP treats those as constants.
- If NaN / Inf appears only in exported outputs, the export backend likely introduced instability
- If NaN / Inf already appears in the traced source outputs, the original computation is unstable too
- If output counts differ, inspect the export backend and output declarations first
- If ranges and diff percentiles look systematically shifted, suspect backend conversion semantics or tolerance settings
