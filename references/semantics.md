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

For new integrations, prefer quantity helpers such as `joint_position` or the matching `Kind` enum with `Axis`. These APIs are experimental and imported directly from `leapp`; confirm consumer support. Direction comes from the input/output list. Set `is_setpoint=True` only for incoming references; outputs reject it. A tensor named `action` is not necessarily an effort command. `InputKindEnum`, `OutputKindEnum`, `TemporalAxis`, and `element_names` are deprecated and emit terminal-visible `FutureWarning` messages. Their behavior and YAML representation are retained; migrate in coordination with the consumer.

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

## Passing and axis contracts

`TensorSemantics` works with `input_tensors` / `output_tensors`, not `method`. Pass one object or a homogeneous list directly, never wrappers inside a dict or mixed with raw tensors in a list. Use separate calls for semantic and raw ports when needed; keep names unique within each node's inputs or outputs. Semantic fields are optional, so an empty wrapper adds no value over a named raw tensor.

`axes` supplies one `Axis` or `None` per tensor dimension. For `[batch, joints]`, use `[None, Axis(AxisKind.ELEMENT, names=joint_names)]`. Supplied names must match the dimension size. Preserve actual layout and order; do not call a batch axis `SYSTEM` unless it truly indexes complete systems (such as robots, vehicles, or machines) or invent labels for latent dimensions.

Use at most one `Axis(AxisKind.TIME, period_ms=...)`; its period must be finite and positive. LEAPP emits `{kind: time, period_ms: ...}` in the corresponding axis entry. Include timing when the consumer needs it, not on every tensor with multiple dimensions.

`GraphConfigs(frequency=...)` records graph execution frequency in Hz; `Axis(AxisKind.TIME, period_ms=...)` records sample spacing in milliseconds. They describe different things and need not be reciprocals (for example, overlapping action chunks). LEAPP does not validate their consistency or make `InferenceManager` schedule calls in real time. Supply actual deployment timing only when needed, never a guessed default.

## Example when a controller needs action chunks

This runnable example illustrates an external joint-position contract with four future targets. It is not a template requiring all pipelines to add names, timing, or graph configs. In a real project, keep the existing computation and use its actual joint order, quantity, and timing. No custom fields are needed here.

```python
import torch
import leapp
from leapp import annotate, joint_position, Axis, AxisKind, GraphConfigs

joint_names = ["hip", "knee", "ankle"]
leapp.start("semantic_graph")
try:
    q = annotate.input_tensors("pipeline", joint_position(
        "joint_pos", torch.randn(3),
        axes=[Axis(AxisKind.ELEMENT, names=joint_names)],
    ))
    actions = q.unsqueeze(0).repeat(4, 1)
    annotate.output_tensors("pipeline", joint_position(
        "joint_targets", actions,
        axes=[Axis(AxisKind.TIME, period_ms=20),
              Axis(AxisKind.ELEMENT, names=joint_names)],
    ), export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True,
                    graph_configs=GraphConfigs(frequency=50))
```

## Custom fields and validation

Use `extra` only for agreed consumer-specific metadata that standard fields cannot express. Entries are flattened into the tensor YAML mapping; graph-level extras are flattened into `pipeline.configs`. Avoid collisions with built-ins such as `name`, `shape`, `dtype`, `type`, `kind`, `axes`, `is_setpoint`, `spatial`, `reference`, `expressed_in`, `element_names`, `temporal_period_ms`, or `__temporal_axis__`. Units, frames, and external IDs are not automatic conversion or routing instructions; arbitrary fields do not become portable merely by appearing in YAML.

Inspect the exported YAML for the intended standard kinds, actual element order and axes, and any required timing/config fields. Confirm that the deployment consumer recognizes the contract. Numerical parity tests alone cannot detect mislabeled units, joint order, or physical meaning. Report missing consumer information instead of filling it with plausible metadata.

## Migrating an existing integration

The [legacy API](https://nvidia-isaac.github.io/leapp/semantics/legacy.html) is deprecated; using legacy kinds, `element_names`, or `TemporalAxis` emits `FutureWarning` messages visible in the terminal by default. No removal version is scheduled. Existing YAML representations remain supported, so coordinate migration with the consumer. Confirm the installed checkout exposes `Kind` and the keyword-only fields before using them; do not bump package or config versions to enable annotations.

- Coordinate changes with the YAML consumer: old `state/`, `command/`, and `target/` kind strings are not converted automatically. Old and new annotations can coexist in one graph or annotation list.
- Replace joint position/velocity kinds with `joint_position` / `joint_velocity`; `JOINT_EFFORT`, `JOINT_TORQUES`, and `COMMAND_JOINT_TORQUES` map to `joint_effort`. Incoming `COMMAND_*` kinds need `is_setpoint=True`; output targets do not.
- Body pose/position/orientation, velocity, and acceleration map to the corresponding `frame_*` quantities; `BODY_VEL` maps to `frame_twist`, `BODY_ACC` to `frame_acceleration`, and `WRENCH` to `frame_wrench`. Inspect actual representations before relabeling: an old body rotation is not automatically a quaternion. Helpers validate supplied pose/orientation/position/velocity component labels, but do not convert values. The generic constructor permits custom conventions.
- Replace `element_names` with `axes` without changing shape or order. Include `None` for unspecified dimensions; labels must match dimension sizes. Replace each `TemporalAxis(period_ms=...)` with `Axis(AxisKind.TIME, period_ms=...)`. Never supply both `element_names` and `axes`.
- Use `reference` for pose/motion reference frames and `expressed_in` for motion/wrench bases. `None` means global. `ExpressionFrame.SELF` / `REFERENCE` serialize as selectors; literal strings serialize as frame names. Pose helpers accept only `reference`; wrench and force-application-position helpers accept only `expressed_in`, and forbid `REFERENCE`. No coordinate transform is performed.
- `Kind` or a new field opts into experimental validation. A custom string alone, even `"frame/twist"`, retains legacy behavior. Existing `.update()` calls store new field names in `extra` until opted in. Construct a new annotation with `Kind` or new fields when migrating; do not expect changing metadata strings to opt in.

For example, an existing output:

```python
TensorSemantics("actions", actions, kind=OutputKindEnum.JOINT_TORQUES,
                element_names=[TemporalAxis(period_ms=20), joint_names])
```

becomes (after importing the new helpers):

```python
joint_effort("actions", actions,
             axes=[Axis(AxisKind.TIME, period_ms=20),
                   Axis(AxisKind.ELEMENT, names=joint_names)])
```

Verify the consumer reads `kind: joint/effort` and `axes` instead of `kind: target/joint/torques`, `element_names`, and `temporal_period_ms`. Compare exported metadata and numerical outputs; preserve values and ordering throughout migration.
