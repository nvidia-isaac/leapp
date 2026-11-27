# Optional deployment semantics

Read this when a user requests semantic metadata or a deployment consumer needs it. For recurrent buffers and episode resets, use [state](state.md) independently.

Online references: [usage and passing conventions](https://nvidia-isaac.github.io/leapp/semantics/usage.html), [standard kinds and axes](https://nvidia-isaac.github.io/leapp/semantics/kind_element_names.html), [convenience helpers](https://nvidia-isaac.github.io/leapp/semantics/convenience.html), [temporal axes](https://nvidia-isaac.github.io/leapp/semantics/kind_element_names.html#semantics-time-axes), [API](https://nvidia-isaac.github.io/leapp/api/index.html). Check these for the current API; a local source checkout is not required.

## Decide whether metadata has a consumer

Semantics describe what a tensor means beyond its name, shape, and dtype. They let a deployment integration distinguish measured joint positions from target commands, match elements to actuators, or interpret a sequence of actions. They do not change numerical computation, repair tracing, or implement state feedback.

Before adding a field, identify the consumer and the decision it enables. Apply metadata primarily to external graph inputs and outputs where this contract is useful. An internal port needs semantics only when a specific consumer actually inspects it, such as an independently deployed component. A node boundary alone is not a reason to add them.

| Situation | Guidance |
| --- | --- |
| Robot adapter needs measured positions and actuator target order | Add the corresponding standard kinds and actual joint names at the external ports. |
| Controller consumes an action chunk | Describe its actual time axis and spacing when the controller supports that contract. |
| Preprocessor passes normalized features to an internal policy node | Use a named raw tensor; no semantic wrapper is needed. |
| Hidden state or history stays inside the exported pipeline | Capture state as needed; do not add semantics just because it persists. |
| Consumer already binds named tensors and needs no semantic metadata | Keep named raw tensors, including at graph inputs and outputs. |

Do not split a model, convert a working `method` annotation, or label every internal edge merely to populate metadata. Preserve requested deployment contracts, but do not invent one when the task is simply export and replay.

## Use standard kinds first

For new integrations, prefer quantity helpers such as `joint_position` or the matching `Kind` enum with `Axis`. These APIs are experimental and imported directly from `leapp`; confirm consumer support. Direction comes from the input/output list. Set `is_setpoint=True` only for incoming references; outputs reject it. A tensor named `action` is not necessarily an effort command.

Use a custom kind only when no standard kind fits **and** the consumer explicitly supports the agreed label. Do not create synonyms for existing kinds or repeat `kind` in `extra`. Unknown meaning should remain unspecified. In particular, normalized joint features, learned latent vectors, and concatenated mixed observations must not be mislabeled as physical joint positions.

## Keep the annotation small

- Start with named raw tensors. Add `TensorSemantics` only to ports needing metadata and include only fields the consumer uses; omit empty/default fields.
- Reuse ordered joint/feature names from the application's existing configuration. Define a shared list once if necessary; do not paste large lists into each annotation or reconstruct guessed ordering.
- Keep short wrappers inline, with one readable entry per port. A small import alias is fine if repeated names dominate the code; avoid generic builders, classes, or registries solely to shorten a few calls.
- Keep capture and export lifecycle calls in their existing locations. Metadata should add a small amount of code around the relevant boundary, not duplicate the pipeline.
- Use standard fields before `extra`. Do not add units, frames, IDs, or timing just to make the export look complete. Retain every field genuinely required by the consumer even if it takes more lines.

For an existing `[batch, joints]` output whose consumer needs target positions and joint order, the annotation can stay compact (`actions` and `joint_names` come from the application):

```python
from leapp import annotate, joint_position, Axis, AxisKind

annotate.output_tensors("pipeline", joint_position(
    "joint_targets", actions,
    axes=[None, Axis(AxisKind.ELEMENT, names=joint_names)],
), export_with="onnx")
```

## Declare semantics independently of values

Use `annotate.set_semantics(node_name, *, inputs=None, outputs=None, states=None, allow_unregistered_ports=False)` when descriptions are known separately from tensor values. Supply a list or tuple of named `TensorSemantics` descriptors for each direction. Their names identify the public ports, including dotted feature keys such as `observation.state`; generated FX identifiers are internal. `ref` is optional for this call. Quantity convenience helpers still require their tensor argument.

Call after `leapp.start()` and before `compile_graph()`, including after `stop()`. Any function or module can supply the descriptions without holding the tensors or original data source. Existing ports are checked by default. Use `allow_unregistered_ports=True` to predefine descriptions and create an ordinary named node before values exist. Tensor annotations reuse that node. Unmatched predefinitions fail compilation; known shapes and intrinsic semantic rules are validated as soon as available.

Each valid call replaces the entire description of the supplied ports, clearing omitted fields and preserving other ports. Updates validate atomically. `states=[...]` applies each named description to both input and output namespaces; it does not create feedback. Both state ports must exist unless predefinition is enabled. Input and output names belong to separate namespaces, so the same name may occur in both. Duplicate declarations within one direction are rejected, including multiple predefinitions that resolve to one shared port.

The node keeps a persistent direction/name map. Tensor registration does not consume entries or copy them onto port descriptions. YAML generation and visualization look up current metadata for surviving ports. Pruned inputs and buffers baked as constants retain their metadata without becoming executable inputs. Backend renaming preserves the declared identity internally. A new `start()` resets the map. Register replacements before compilation; later changes do not update saved artifacts, and loaded-artifact editing has no API here.

For method-based nodes, keep function arguments as ordinary tensors and use `set_semantics()` with the method's actual argument and return port names. Preserve existing method boundaries when the task only adds metadata.

## Inline annotations and axis contracts

For metadata supplied with values, pass one bound `TensorSemantics` object or a homogeneous list directly to `input_tensors`, `output_tensors`, `state_tensors`, or `update_state`. Each descriptor needs a concrete `ref`. Wrappers inside a dict or mixed with raw tensors in a list are unsupported. Use separate calls for semantic and raw ports when needed; keep names unique within each node's inputs or outputs. Inline annotations write to the same node map and use full-description replacement. Semantic fields are optional, so an empty bound wrapper adds no value over a named raw tensor.

`axes` supplies one `Axis` or `None` per tensor dimension. For `[batch, joints]`, use `[None, Axis(AxisKind.ELEMENT, names=joint_names)]`. Supplied names must match the dimension size. Preserve actual layout and order; do not call a batch axis `SYSTEM` unless it truly indexes complete systems (such as robots, vehicles, or machines) or invent labels for latent dimensions.

Use at most one `Axis(AxisKind.TIME, period_ms=...)`; its period must be finite and positive. LEAPP emits `{kind: time, period_ms: ...}` in the corresponding axis entry. Include timing when the consumer needs it, not on every tensor with multiple dimensions.

`GraphConfigs(frequency=...)` records graph execution frequency in Hz; `Axis(AxisKind.TIME, period_ms=...)` records sample spacing in milliseconds. They describe different things and need not be reciprocals (for example, overlapping action chunks). LEAPP does not validate their consistency or make `InferenceManager` schedule calls in real time. Supply actual deployment timing only when needed, never a guessed default.

## Example when a controller needs action chunks

This runnable example predefines an external joint-position contract with four future targets, then registers ordinary named tensors. Use the application's actual joint order, quantity, and timing. Names, timing, and graph configs remain optional when the consumer does not require them.

```python
import torch
import leapp
from leapp import annotate, TensorSemantics, Kind, Axis, AxisKind, GraphConfigs

joint_names = ["hip", "knee", "ankle"]
leapp.start("semantic_graph")
try:
    annotate.set_semantics(
        "pipeline",
        inputs=[TensorSemantics(
            "joint_pos", kind=Kind.JOINT_POSITION,
            axes=[Axis(AxisKind.ELEMENT, names=joint_names)],
        )],
        outputs=[TensorSemantics(
            "joint_targets", kind=Kind.JOINT_POSITION,
            axes=[Axis(AxisKind.TIME, period_ms=20),
                  Axis(AxisKind.ELEMENT, names=joint_names)],
        )],
        allow_unregistered_ports=True,
    )
    q = annotate.input_tensors("pipeline", {"joint_pos": torch.randn(3)})
    actions = q.unsqueeze(0).repeat(4, 1)
    annotate.output_tensors("pipeline", {"joint_targets": actions}, export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True,
                    graph_configs=GraphConfigs(frequency=50))
```

## Custom fields and validation

Use `extra` only for agreed consumer-specific metadata that standard fields cannot express. Entries are flattened into the tensor YAML mapping; graph-level extras are flattened into `pipeline.configs`. Avoid collisions with built-ins such as `name`, `shape`, `dtype`, `type`, `kind`, `axes`, `is_setpoint`, `spatial`, `reference`, or `expressed_in`. Units, frames, and external IDs are not automatic conversion or routing instructions; arbitrary fields do not become portable merely by appearing in YAML.

Inspect the exported YAML for the intended standard kinds, actual element order and axes, and any required timing/config fields. Confirm that the deployment consumer recognizes the contract. Numerical parity tests alone cannot detect mislabeled units, joint order, or physical meaning. Report missing consumer information instead of filling it with plausible metadata.
