---
name: leapp-integrate
description: Annotate and export downstream LEAPP pipelines with a single-model ONNX default, function patches, state and deployment semantics, runtime validation, and optional Warp setup/capture. Use for application integration rather than tracer internals.
---

# Integrate LEAPP into a project

**Start by exporting the full compute pipeline as one model (one LEAPP node), unless the user requests multiple models or a concrete technical/deployment requirement justifies splitting it.** Existing preprocessing, policy, and postprocessing helpers can all execute within that node; their function boundaries do not require separate exports.

Preserve the existing computation. **Avoid rewriting pipelines, model implementations, or library call sites to make them traceable. Use `leapp.FunctionPatch` for unsupported module-level functions instead.** Keep adapters in the integration layer, scoped to the behavior the application actually uses. Supported Python helpers can execute unchanged inside a node.

**Recommend ONNX unless the user specifies another format.** Explicitly set `export_with="onnx"` (the `onnx-dynamo` alias); omitting `export_with` can select no export. Try compatible alternative exporters when the first fails, before considering application changes.

## Discover docs and establish the environment

Start with the [online LEAPP documentation](https://nvidia-isaac.github.io/leapp/). **Do not assume LEAPP is installed or that a source checkout exists.** Read only the guides relevant to the application:

| Need | Online reference |
|------|------------------|
| Public signatures and annotation patterns | [API](https://nvidia-isaac.github.io/leapp/api/index.html), [nodes](https://nvidia-isaac.github.io/leapp/guides/nodes.html) |
| Export formats, parameters, prebuilt models | [Export](https://nvidia-isaac.github.io/leapp/guides/export.html) |
| Unsupported functions and trace loss | [Function patches](https://nvidia-isaac.github.io/leapp/guides/patching.html), [Torch limitations](https://nvidia-isaac.github.io/leapp/tensor_libraries/torch_limitations.html), [NumPy limitations](https://nvidia-isaac.github.io/leapp/tensor_libraries/limitations.html) |
| State and resets | [State](https://nvidia-isaac.github.io/leapp/guides/graph.html) |
| Optional deployment metadata | [Semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [standard kinds](https://nvidia-isaac.github.io/leapp/semantics/kind_element_names.html), [temporal axes](https://nvidia-isaac.github.io/leapp/semantics/temporal.html) |
| Validation and deployment | [Debugging](https://nvidia-isaac.github.io/leapp/guides/debugging.html), [runtime](https://nvidia-isaac.github.io/leapp/leapp_runtime.html), [bundle/YAML](https://nvidia-isaac.github.io/leapp/generated_configs.html) |
| Warp | [Setup](https://nvidia-isaac.github.io/leapp/tensor_libraries/installation.html), [capture](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp.html), [limitations](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp_limitations.html) |

Use the project's established Python environment and dependency manager. If LEAPP is absent, add `leapp` using that workflow (for example, `python -m pip install leapp` inside a virtual environment). Respect project version constraints; do not broadly upgrade dependencies. Warp needs the extras and native build described below.

Once available, record `leapp.__version__`, `leapp.__file__`, and relevant public signatures. Online docs can be newer than an installed package; verify compatibility before using an API. Resolve conflicting examples against the version's signature or matching upstream source/tests, without requiring a local checkout. Keep the user's single-model/ONNX preferences when docs illustrate other choices.

## Start with one model; split only for a reason

1. Run or inspect the original entrypoint. Map external inputs, meaningful computation stages, final outputs, mutable state, and external services. Record shapes, dtypes, devices, layout, and representative inputs.
2. Place one input boundary before the pipeline's compute and one output boundary after its final result. Include preprocessing, feature fusion, policy inference, and action conversion in that same node where applicable. Multiple helpers, branches, model calls, or stateful components alone are not reasons to split.
3. Keep acquisition, simulator stepping, file/network I/O, visualization, and orchestration outside exported compute nodes. Explicitly identify the runtime responsible for any external computation between nodes; a disconnected graph is not a complete exported pipeline.
4. Keep **every input-dependent transformation** inside the exported compute, including normalization, slicing, reshaping, dtype conversion, clipping, and concatenation. These operations must run on new inputs at inference time, not be omitted or frozen to values from tracing. If multiple nodes are needed, include each transformation in a producer or consumer; unannotated math between nodes can become an extra graph input instead of an edge.
5. Split only when requested or justified by a concrete requirement: separately deployed components, different devices/rates required by the target runtime, incompatible export backends, or an unsupported external operation that a faithful function patch cannot capture. For export failures, investigate patches and compatible exporters before partitioning. State the reason for each split, then choose cohesive stage boundaries with clear tensor contracts.
6. When using multiple nodes, use stable names, finalize producers before consumers, and explicitly connect published outputs at branches and joins. Do not nest independently annotated nodes inside an active node. Node-name reuse means re-entry of the same logical stage and annotation origin; independently exported instances with different inputs, weights, or state need distinct names.

## Annotation API chooser

| API | Use |
|-----|-----|
| `input_tensors` + `output_tensors` | Primary interface for explicit names, semantic metadata, and nodes spanning existing helpers or files. |
| `@annotate.method(...)` | Shorthand when one existing function naturally defines a node. `node_name` defaults to the function name. |
| `leapp.FunctionPatch(module, name, replacement)` | Traceable adapter for an unsupported function, registered through `start(patching=[...])`; not a node declaration. |
| `annotate.module(name, model)` | Track registered `nn.Module` buffers used in a node; reassigned buffers become feedback. |
| `state_tensors` + `update_state` | Explicit recurrent state and its next value. |
| `annotate.register_buffer` | Frozen constants in the exported model, without feedback. |
| `output_tensors(..., static_outputs=...)` | Deliberately constant output tensors. Never use to conceal lost tracing. |
| `annotate.warp_op` | Explicit Warp segment within an existing node; see the Warp subsection below. |

Avoid legacy private `annotate._method` for new integrations; investigate public interfaces first.

## Complete annotation and validation recipe

This runnable stateless example keeps an original function unchanged, caches multiple trace samples, exports one model, and checks fresh inputs. Replace the illustrative function and sample generator with the application's existing compute and representative data. For mutable pipelines, isolate baseline inputs/state and reset both executions to the same initial state; see [state and episode lifecycle](references/state.md).

```python
import torch
import leapp
from leapp import annotate, InferenceManager


def original_pipeline(obs):
    features = (obs - obs.mean()) / (obs.std() + 1e-6)
    return torch.tanh(features[..., :12])


samples = [torch.randn(1, 32) * scale + 0.2 for scale in (0.3, 0.5, 1, 2, 4)]
expected = [original_pipeline(obs) for obs in samples]

leapp.start(name="demo_graph", save_path=".", max_cached_io=5)
try:
    for obs, reference in zip(samples, expected):
        x = annotate.input_tensors("pipeline", {"obs": obs})
        action = original_pipeline(x)
        annotate.output_tensors("pipeline", {"action": action}, export_with="onnx")
        torch.testing.assert_close(action, reference)
finally:
    leapp.stop()
leapp.compile_graph(visualize=True, validate=True, rtol=1e-3, atol=1e-5, strict=True)

manager = InferenceManager("demo_graph/demo_graph.yaml")
assert set(manager.inputs) == {"pipeline/obs"}
assert set(manager.outputs) == {"pipeline/action"}
for scale in (0.7, 1.5, 3.0):
    inputs = manager.get_mock_input()
    fresh = torch.randn_like(inputs["pipeline/obs"]) * scale - 0.4
    reference = original_pipeline(fresh)
    inputs["pipeline/obs"] = fresh
    actual = manager.run_policy(inputs)["pipeline/action"]
    torch.testing.assert_close(actual, reference.to(actual.device), rtol=1e-3, atol=1e-5)
```

`max_cached_io=5` only sets cache capacity; **execute the annotated pipeline repeatedly on distinct inputs** to collect samples. A mock-input run or parity on the first captured input is insufficient. Preserve application weights, execution mode, state, and random seeds as appropriate. Never call `compile_graph()` before `stop()`. Restore session patches on errors; resolve active-wrapper errors before retrying in that process.

- Raw I/O uses a top-level **dict of named tensors**. A single input returns its value, not a dict; multiple inputs return a tuple in dict insertion order. Always use these returned values for computation.
- Nested dictionaries, lists, and tuples can remain nested within named values. Inspect generated flattened port names rather than guessing them or rewriting application containers.
- `input_tensors("node", ...)` may be called at multiple sites while assembling that node's initial trace. One `output_tensors()` finalizes it. Later iterations re-enter the same stage to validate/cache I/O and detect feedback; they do not create additional graph nodes.
- `method()` derives return names from source. Use explicit I/O annotations for ambiguous returns or unavailable source rather than rewriting a function solely for name extraction.

## Export methods and fallback strategy

Select per node through `export_with` on `output_tensors()` or `method()`. LEAPP emits per-node artifacts plus `<graph_name>.yaml`, optional PNG, and feedback initial-value data when applicable.

| `export_with` | Method / artifact | When to try |
|---------------|-------------------|-------------|
| `"onnx"`, `"onnx-dynamo"` | Dynamo ONNX exporter → `.onnx` | Default recommendation; modern tensor graphs and supported Warp nodes. These two names select the same backend. |
| `"onnx-torchscript"` | TorchScript-based ONNX exporter → `.onnx` | Alternative ONNX lowering when Dynamo fails or validates incorrectly; often appropriate first for GRU/LSTM. |
| `"jit"`, `"jit-script"` | TorchScript scripting → `.pt` | A TorchScript target or an alternative when ONNX lowering is unsupported. Both names select the same backend. |
| `"jit-trace"` | TorchScript tracing → `.pt` | Scripting cannot handle the captured graph; requires representative input and execution-path coverage. |
| `"pt2"`, `"exported-program"` | `torch.export` → `.pt2` | A requested ExportedProgram target or a compatible modern PyTorch runtime; supports Warp with its runtime adapter. Both names select the same backend. |
| `None` | No compilation; optionally reference an existing model | Deliberate external/prebuilt model integration or an explicitly incomplete placeholder. Not an export-success fallback. |

When the first export fails:

1. Identify the failing node and whether the problem is tracing, exporter lowering, missing dependencies, or runtime/numerical validation. An exporter switch cannot recover computation already lost during tracing; use a function patch for an unsupported source operation.
2. For ONNX lowering failures, try the **other ONNX exporter** with unchanged computation. Switching `"onnx"` to `"onnx-dynamo"` is not a new attempt. For recurrent graphs, try `onnx-torchscript` early. Warp supports Dynamo ONNX and PT2, not the legacy ONNX or JIT routes.
3. If both ONNX routes are unsuitable and no user requirement mandates ONNX, try compatible `pt2`, `jit-script`, or `jit-trace` based on the error and target runtime. For JIT scripting failures, try `jit-trace`. Mixed per-node backends are possible when the consuming runtime supports them; report any departure from the ONNX recommendation.
4. Honor a user-specified format. Other formats can help diagnose a failure but do not satisfy a required ONNX/PT2/JIT deliverable. Do not silently substitute formats or drop failing nodes.
5. Change the annotation's backend settings, then retrace in a fresh session/output directory with original state reset. Validate each candidate; keep a concise record of backend, parameters, and failure. Do not retry the same failing configuration without new evidence. If suitable alternatives fail, report the narrow unsupported operation and remaining deployment gap before proposing a scoped application change.

Useful `backend_params` (check the installed implementation):

- Both ONNX paths: `opset_version`, `report`; Dynamo also `verify`, `optimize`; legacy ONNX also `prescript`.
- PT2: `strict` controls `torch.export`, distinct from `compile_graph(strict=True)` validation.
- `None`: `{"model_path": "/path/model.onnx", "copy_original_model": True}` references/copies an existing `.onnx`, `.pt`, or `.pt2`. Verify its I/O contract and runtime execution.

Choose opset and optimizations for the actual runtime. Disabling validation, `skip_validation=True`, dry-run, or `non_traced` is diagnostic only, not evidence of a working export.

## State, constants, and loops

For recurrent policies, history buffers, or episode resets, read [state and episode lifecycle](references/state.md).

- `state_tensors` requires an existing node and a **flat dict of named Torch tensors**; nested state is unsupported. `update_state` must use declared names and preserve each state's shape and dtype. One entry returns a value; multiple entries return a tuple in insertion order.
- Only states passed to `update_state()` become explicit feedback; an unupdated state remains a regular input. Keep evolving state distinct from frozen parameters and constants.
- After opening the node, call `annotate.module("node", model, buffer_names=[...])` before model execution to select registered buffers; omit `buffer_names` to track all. It tracks reassignment (`self.h = new_h`), not in-place updates (`self.h.copy_(...)`) or plain attributes. Use explicit state annotations for existing mutation patterns where possible; a pure function patch cannot represent mutation.
- Inferred cross-node feedback needs **at least two iterations** between `start()` and `stop()`. Explicit state/module feedback can be declared in one trace; still validate multiple time steps and initial/reset state. Confirm `pipeline.feedback_flow` and the saved initial values.
- Re-entry does not make a Python data-dependent branch, a loop count, or shape dynamic. Verify the supported input/shape domain; do not promise dynamic shapes from a single successful example.

## Preserve graph connections and computation

Current tracing connects a published `(producing node, output port)` to a consumer. Verify the installed version's behavior if using an older release.

- Equivalent copies can preserve the port: Torch `clone`/`detach`/`contiguous`/device-only transfers/full buffer overwrites, NumPy `np.copy`/`.copy()`/`np.asanyarray`, and full-range `wp.copy`.
- Arithmetic, slicing, reshaping, dtype casts, and partial writes between nodes can clear the port. Move the annotation boundary to include that existing computation; do not relabel the result as its unmodified source.
- `annotate.mirror_leapp_tags(source, target)` is only for **equivalent copies between finished nodes**, with matching values, shape, and dtype. Never use it to repair a transformed value or a missing computation. Assign its return value, especially for raw NumPy destinations that cannot be upgraded in place.
- Use `static_outputs` only for intentional constants (named raw Torch tensors or supported semantic wrappers). Capture unsupported numerical functions with [FunctionPatch](references/function-patching.md). If that cannot preserve behavior, identify an explicit external runtime boundary and report that the export is partial.

## Diagnose trace loss before changing exporters

Calls in NumPy, SciPy, scikit-learn, scikit-image, OpenCV, or other libraries **may return ordinary untraced arrays/tensors**, depending on the operation. A library name alone does not determine support: helpers built entirely from supported operations can trace unchanged. If an input-dependent result loses tracing, patch the unsupported module-level function with a faithful [FunctionPatch adapter](references/function-patching.md). For a scikit estimator method, locate the unsupported module-level call or a suitable module-level application helper; `FunctionPatch` does not target arbitrary bound methods.

| Hazard | What the agent should do |
|--------|--------------------------|
| Unsupported NumPy/native operation | Intercepted NumPy calls use validated ATen capture and raise on missing lowerings or parity failures. Calls that bypass the hooks can still return plain values that become constants. Patch unsupported computation and compare new inputs against the original. |
| `.item()`, `.tolist()`, `float()`, `int()` | Audit explicitly: Torch extraction can be silent. Keep dynamic data in traced operations; logs alone cannot prove correctness. |
| NumPy indexing down to a scalar | Keep an array dimension to preserve runtime dependence. Constant NumPy scalar operands are accepted, but cannot restore a dependency lost by indexing. |
| NumPy negative-stride slicing | Static basic reads use `flip` or normalized indices with `index_select`. Runtime bounds, mixed advanced indices, reverse assignment, and mutation through reversed views need separate support. |
| Unsupported `astype` dtype | Eager and exported dtypes can disagree. Use a supported dtype and check replay against the original. |
| Tensor-dependent Python `if`/`while` or iteration counts | Only the executed path is captured. Use a faithful traceable expression in an adapter where possible; otherwise document the supported domain or external control-flow boundary. Another exporter cannot recover a branch already lost. |

Supported NumPy/Torch conversions such as `torch.from_numpy()` and `.numpy()` can preserve tracing with global patching enabled. Do not rewrite all NumPy code or treat every conversion as a trace break.

## Optional deployment semantics

Add semantics when a user requests them or an identified downstream consumer needs tensor meaning to bind observations, map joint order, interpret commands, or apply action chunks. They are not required for tracing, export, numerical validation, or state feedback. Default to named raw tensors for ordinary internal node connections; do not annotate every intermediate value or add nodes solely for metadata.

When needed, read [deployment semantics](references/semantics.md). **Prefer standard `InputKindEnum` / `OutputKindEnum` kinds** over custom strings or `extra`. Keep annotations compact: reuse existing ordered names, specify only consumed fields, and avoid repeated wrappers or metadata factories. Do not guess physical meaning for normalized features, latent vectors, or mixed observations. State/reset contracts are covered separately in [state](references/state.md).

## Warp setup and export (experimental)

Warp is part of this skill. **Before integrating Warp, read [the complete Warp guide](references/warp.md)** for prerequisites, native runtime setup, automatic/explicit capture, limitations, replay, and troubleshooting. No separate skill or LEAPP source checkout is required.

- Use the matching `leapp[warp-cu12]` or `leapp[warp-cu13]` extra, then run `leapp-build-warp-runtime` and `leapp-build-warp-runtime --status` in that environment.
- Export capture currently requires Linux, a CUDA GPU, the matching CUDA toolkit, and Warp >=1.16. Verify current requirements in the online setup guide.
- Initialize Warp before `start(global_patching=True)`. Run the same Warp path at least twice: discovery then capture. A segment groups Warp calls inside a node; it is not a reason to split the full pipeline into multiple models.
- Use ONNX Dynamo by default; PT2 is the other supported Warp exporter. Deploy the corresponding native runtime dependency along with the bundle.

## Validate the whole exported pipeline

`compile_graph(validate=True)` compares all cached node outputs, including exported state updates. Consecutive samples validate the original pipeline's stateful transitions, not just its actions. Each sample supplies recorded reference inputs/state; sequential runtime replay instead feeds exported state forward and checks accumulated numerical drift and runtime feedback/reset handling. Cache size controls transition coverage, not feedback replay. See [stateful validation](https://nvidia-isaac.github.io/leapp/guides/debugging.html#stateful-transitions-and-sequential-replay).

1. Compare the original application and annotated execution using the same inputs, weights, state, execution mode, and random seeds where relevant. Check function adapters against original functions separately.
2. Run `compile_graph(validate=True, strict=True)` with justified tolerances. Inspect warnings and generated YAML: expected external inputs, final outputs, forward/feedback edges, initial values, backend types, and existing artifact paths. Explain any unexpected dangling input.
3. Load the bundle with `InferenceManager` and compare **fresh representative inputs not used during tracing** against the original pipeline. A mock-input smoke test alone is insufficient. Include realistic value ranges and relevant branches/shapes; for stateful pipelines compare a sequence from matching initial/reset state.
4. Verify the runtime/device and any custom operators required by deployment. ONNX Runtime defaults to CPU-safe execution; GPU/Warp requirements must be checked in the actual environment.

Use actual `manager.inputs` / `manager.outputs` keys (`node/port`). Inspect intermediate node values via `manager.value_dict`; seed feedback when needed with `set_input_value(node, port, value)` or a fresh manager.

## Debugging controls

- Read `<save_path>/<graph_name>/log.txt`; it includes DEBUG output even when the console is quiet. Search for `Compiled graph module for <node name>` to inspect the captured FX graph: live inputs should be placeholders, expected math should be present, and constants should be intentional. Unused inputs may be pruned; investigate a missing input expected to affect the output.
- On a mismatch, use the reported node, output, sample index, shapes/dtypes, and error statistics. A first-sample pass followed by later failures suggests frozen values or state problems. `strict=False` exposes diagnostic results; it does not turn failures into successful validation.
- `start(verbose=True)` enables detailed tracing logs. Keep `max_cached_io` around `3–5` initially and expand when representative replay coverage needs it.
- `start(dry_run=True)` checks tracing/structure without model export or model validation. Current `compile_graph()` does not accept `dry_run`; inspect signatures on older installations.
- `start(non_traced=["node"])` skips selected exports while retaining I/O/edges. Use to isolate a failure, and restore real exports before declaring completion.
- `global_patching=False` can diagnose patch interference, but disables user `FunctionPatch` and automatic patch-dependent tracing; it is not a transparent compatibility fix.
- `visualize=False` avoids optional PNG rendering. Model export and validation remain required.

## Deliver the bundle

Deliver the complete export directory: model artifact(s), graph YAML, `*_initial_values.safetensors` when feedback exists, and all referenced model/external-data files. Preserve paths relative to the YAML. PNG and logs are useful diagnostics; they do not replace runnable artifacts. For Warp, document/build the matching native adapter and its discovery/override configuration on the destination.

Confirm runtime input/output keys, shapes, dtypes, device expectations, semantics when used, state initialization/reset behavior, and supported input domain. Preserve YAML positional I/O order for consumers loading the model directly. Where practical, load and run a copied bundle from its delivery location to detect hidden source-path dependencies.

Report the model count and any reasons for splitting, node boundaries and graph I/O, annotation/adapter changes, chosen backend per node and fallback attempts, generated bundle paths, fresh-input and stateful parity results, and any remaining runtime or input-domain limitations. Distinguish a complete runnable bundle from a traced graph, partial export, or diagnostic-only result.
