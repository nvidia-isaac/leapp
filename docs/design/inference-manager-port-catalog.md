<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# InferenceManager port catalog

Implementation spec for a self-reporting I/O API on `InferenceManager`.
Downstream runtimes (Isaac Sim, ROS, tests) must be able to bind a live robot
to an exported graph **without parsing LEAPP YAML**.

Primary code: `leapp/inference_manager.py`.
Semantics source of truth: `leapp/utils/tensor_description.py`, `leapp/utils/enums.py`.
Existing runtime docs: `docs/source/leapp_runtime.rst`.

## Goal

`InferenceManager` already loads the YAML, knows external pipeline I/O, and can
`get_mock_input()` / `run_policy()`. Today it only reports **keys**
(`"node/name"`). Hosts still join `pipeline.inputs` to `models[node].inputs`
to read `kind`, `element_names`, `shape`, and device.

After this change, a host should do:

```python
from leapp import InferenceManager

manager = InferenceManager("bundle.yaml")

joint_pos = manager.input_ports["policy/joint_pos"]
joint_pos.kind            # "state/joint/position"
joint_pos.element_names   # ["left_hip", "left_knee", ...]
joint_pos.shape           # (1, 12)
joint_pos.dtype           # torch.float32
joint_pos.device          # torch.device(...)

target = manager.output_by_kind("target/joint/position")
command = manager.input_by_kind("command/body/velocity")

inputs = manager.zeros_input()  # or get_mock_input()
inputs[joint_pos.key] = live_tensor
outputs = manager.run_policy(inputs)
action = outputs[target.key]
```

Do **not** require hosts to read `manager.leapp_description`, `manager.pipeline`,
or `manager.models`. Those may remain as internal implementation details; they
are not the deploy API.

## Non-goals

- Mapping `kind` onto a simulator (articulation gather/scatter, body-frame
  velocity, PD vs LSTM). That stays in the host.
- Changing YAML on-disk format.
- Changing `run_policy` key format (`"node/name"`).

## Current API (keep working)

| Member | Today |
|---|---|
| `inputs` | `list[str]` of `"node/name"` |
| `outputs` | `list[str]` of `"node/name"` |
| `feedback_inputs` | `list[str]` of `"node/name"` |
| `get_mock_input()` | `dict[str, Tensor]` keyed like `run_policy` |
| `run_policy(inputs)` | same keys; returns output dict with same key style |
| `nodes[name].device` | per-node device |
| `nodes[name].input_descriptions` | raw YAML dicts |

## Design

### Port object

One type for external inputs and outputs (name e.g. `InferencePort`).

Keys are **`"node/name"`**, not bare tensor names. Names are not unique across
nodes. This matches `run_policy`.

```python
class InferencePort:
    key: str                    # "policy/joint_pos"
    node: str                   # "policy"
    name: str                   # "joint_pos"
    kind: str | None            # YAML kind string, e.g. "state/joint/position"
    element_names: list[str]    # flattened; empty if absent
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    temporal_period_ms: float | None
    extra: dict[str, Any]       # remaining semantic fields not listed above
```

Access is attribute-style: `port.kind`, `port.element_names`.

**`kind`:** serialize as the YAML string (`"state/joint/position"`, `"kp"`,
custom strings). Do not drop custom kinds. Enums (`InputKindEnum` /
`OutputKindEnum`) may be accepted in `*_by_kind` lookups by comparing `.value`.

**`element_names` flattening:** export writes nested lists such as
`[["j1", "j2"]]`. The port must return a 1-D `list[str]` of element labels.
Temporal axes (`TemporalAxis` / `"__temporal_axis__"`) must **not** appear in
that list; put the period on `temporal_period_ms` instead. If names are missing,
return `[]`, not `None`.

**`shape` / `dtype`:** YAML `shape` is sometimes a JSON string. Decode the same
way `get_mock_input()` already does (`json.loads` if str). Map dtype through
`map_to_torch_dtype`. Never leave shape as a string on the port.

**`extra`:** flattened leftover semantic keys from the tensor description
(`TensorSemantics.to_dict()` minus fields already on the port). Hosts must not
need the raw YAML dict.

### Catalogs on InferenceManager

Prefer new names so existing `inputs` / `outputs` list properties stay
backward compatible:

| Member | Type | Meaning |
|---|---|---|
| `input_ports` | `Mapping[str, InferencePort]` | external pipeline inputs only |
| `output_ports` | `Mapping[str, InferencePort]` | external pipeline outputs only |
| `feedback_ports` | `Mapping[str, InferencePort]` optional | same keys as `feedback_inputs` |

Iteration order should match current `inputs` / `outputs` list order.

Lookups (required):

```python
def input_by_kind(self, kind: str | InputKindEnum) -> InferencePort | None: ...
def output_by_kind(self, kind: str | OutputKindEnum) -> InferencePort | None: ...
def inputs_by_kind(self, kind: str | InputKindEnum) -> list[InferencePort]: ...
def outputs_by_kind(self, kind: str | OutputKindEnum) -> list[InferencePort]: ...
```

- Singular helpers return the **first** match, or `None`.
- Plural helpers return all matches (kinds can repeat).
- Compare against the string kind on the port.

Build catalogs once in `__init__` after nodes exist (device is known then).
Do not re-parse the YAML file; use `self.pipeline` + `self.nodes[*].*_descriptions`
already loaded by `_load_description` / `_create_nodes`.

### Graph-level (optional but wanted)

If `pipeline["configs"]` is present, expose it without YAML diving:

```python
@property
def frequency(self) -> float | None:
    """Graph frequency in Hz from pipeline.configs, if exported."""
```

A read-only `configs: Mapping[str, Any]` is fine. Do not invent frequency when
absent.

### Fill helpers

Keep `get_mock_input()`. Add:

```python
def zeros_input(self) -> dict[str, torch.Tensor]:
    """Zeros (not randn) for every external input, correct shape/dtype/device."""
```

Same keys as `run_policy`. Sim hosts prefer zeros over random mock data.

## Implementation notes

- Reuse existing shape/dtype parsing from `get_mock_input()`; extract a small
  helper used by ports, mock, and zeros.
- `NodeManager.input_descriptions` / `output_descriptions` stay as loaded dicts.
  Ports are the public view of **pipeline-external** I/O only, not every node
  tensor.
- Do not make `leapp_description` part of the public contract.
- Update `docs/source/leapp_runtime.rst` with the port catalog example.
- Export `InferencePort` from `leapp/__init__.py` if it is a public type.

## Tests

Add functional tests (exported tiny graph with `TensorSemantics` kinds +
element_names) covering:

1. `input_ports` keys equal `inputs`.
2. `port.kind` and flattened `element_names` match what was annotated.
3. Nested YAML `[["a", "b"]]` flattens to `["a", "b"]`.
4. `shape` is `tuple`, `dtype` is `torch.dtype`, `device` matches the node.
5. `input_by_kind("state/joint/position")` finds the annotated input.
6. Custom kind string (not in the enum) is preserved and findable.
7. `output_by_kind("target/joint/position")` and `"kp"` / `"kd"` if present.
8. `zeros_input()` keys/shapes/devices match `get_mock_input()` but are zeros.
9. `run_policy(zeros_input())` still runs (smoke).
10. Temporal `element_names` do not leak `"__temporal_axis__"` into
    `element_names`; `temporal_period_ms` is set.

Do not require Isaac Sim for these tests.

## Host contract after this ships

Isaac Sim `LeappPolicyRunner` should drop `_load_yaml` / `_external_bindings`
and:

- iterate `manager.input_ports.values()` to fill `run_policy`
- use `output_by_kind("target/joint/position")` (required), `"kp"`, `"kd"`
- gather/scatter joints with `port.element_names`

Kind → simulation mapping stays in Isaac Sim. Gravity currently hacks
`state/vector3d` + name containing `"gravity"`; that is a semantics gap, not
part of this API. Prefer a dedicated kind later; do not encode name sniffing
in `InferenceManager`.

## Out of scope unless cheap

- Changing `inputs` from `list[str]` to a Mapping (breaking). New `*_ports`
  properties avoid that.
- Quaternion convention, command layout, or actuator models.
- Documenting `value_dict` / `"==out=="` further.
