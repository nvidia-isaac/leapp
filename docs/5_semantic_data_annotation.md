# Semantic Data Annotation in LEAPP

This guide covers how to add semantic metadata to your tensors using `TensorSemantics`. Semantic annotations let you describe **what** a tensor represents (e.g., joint positions, target torques) and provide element-level naming, making generated YAML specifications self-documenting and enabling downstream consumers to interpret the data correctly.

> **Note:** Semantic annotation is only available for `annotate.input_tensors()` and `annotate.output_tensors()`. It is not supported for `@annotate.method()` nodes.

## Basic Usage

Instead of passing raw tensors to `input_tensors` or `output_tensors`, wrap them in `TensorSemantics` objects. Pass them as a single `TensorSemantics` or a list of `TensorSemantics`:

```python
import torch
import leapp
from leapp import annotate, TensorSemantics
from leapp.utils.enums import InputKindEnum, OutputKindEnum

joint_pos = torch.randn(1, 12)
joint_vel = torch.randn(1, 12)

leapp.start("my_robot")

# Pass a list of TensorSemantics as inputs
traced_pos, traced_vel = annotate.input_tensors("policy", [
    TensorSemantics("joint_pos", joint_pos,
                     kind=InputKindEnum.JOINT_POSITION,
                     element_names=["hip_l", "knee_l", "ankle_l",
                                    "hip_r", "knee_r", "ankle_r",
                                    "shoulder_l", "elbow_l", "wrist_l",
                                    "shoulder_r", "elbow_r", "wrist_r"]),
    TensorSemantics("joint_vel", joint_vel,
                     kind=InputKindEnum.JOINT_VELOCITY),
])

# Compute output
command = traced_pos + traced_vel

# Pass a list of TensorSemantics as outputs
annotate.output_tensors("policy", [
    TensorSemantics("command", command,
                     kind=OutputKindEnum.JOINT_TORQUES,
                     element_names=["hip_l", "knee_l", "ankle_l",
                                    "hip_r", "knee_r", "ankle_r",
                                    "shoulder_l", "elbow_l", "wrist_l",
                                    "shoulder_r", "elbow_r", "wrist_r"]),
])

leapp.stop()
leapp.compile_graph()
```

The generated YAML will include the semantic metadata:

```yaml
models:
  policy:
    inputs:
    - name: joint_pos
      dtype: float32
      shape: [1, 12]
      type: tensor
      kind: state/joint/position
      element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
          shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]
    - name: joint_vel
      dtype: float32
      shape: [1, 12]
      type: tensor
      kind: state/joint/velocity
    outputs:
    - name: command
      dtype: float32
      shape: [1, 12]
      type: tensor
      kind: target/joint/torques
      element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
          shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]
```

## Semantic Fields

### `kind`

The `kind` field describes the **semantic role** of a tensor — what physical quantity or command it represents. LEAPP provides two separate enums for inputs and outputs to clearly distinguish between observed state and commanded targets.

#### `InputKindEnum` — for input tensors

Used with `annotate.input_tensors()`. These represent **observed state** or **commanded references** flowing into a node.

| Enum Value | YAML String | Description |
|------------|-------------|-------------|
| `JOINT_POSITION` | `state/joint/position` | Observed joint positions (e.g., encoder readings) |
| `JOINT_VELOCITY` | `state/joint/velocity` | Observed joint velocities |
| `BODY_LINEAR_ACCELERATION` | `state/body/linear_acceleration` | Body linear acceleration (e.g., from IMU) |
| `BODY_LINEAR_VELOCITY` | `state/body/linear_velocity` | Body linear velocity |
| `BODY_ANGULAR_ACCELERATION` | `state/body/angular_acceleration` | Body angular acceleration |
| `BODY_ANGULAR_VELOCITY` | `state/body/angular_velocity` | Body angular velocity (e.g., gyroscope) |
| `BODY_ROTATION` | `state/body/rotation` | Body rotation / orientation |
| `COMMAND_JOINT_POSITION` | `command/joint/position` | Commanded joint position reference |
| `COMMAND_JOINT_VELOCITY` | `command/joint/velocity` | Commanded joint velocity reference |
| `COMMAND_BODY_ROTATION` | `command/body/rotation` | Commanded body rotation reference |
| `COMMAND_BODY_VELOCITY` | `command/body/velocity` | Commanded body velocity reference |
| `COMMAND_JOINT_TORQUES` | `command/joint/torques` | Commanded joint torques reference |

```python
from leapp.utils.enums import InputKindEnum

TensorSemantics("joint_pos", tensor, kind=InputKindEnum.JOINT_POSITION)
TensorSemantics("imu_gyro", tensor, kind=InputKindEnum.BODY_ANGULAR_VELOCITY)
TensorSemantics("target_pos", tensor, kind=InputKindEnum.COMMAND_JOINT_POSITION)
```

#### `OutputKindEnum` — for output tensors

Used with `annotate.output_tensors()`. These represent **target commands** or **control outputs** produced by a node.

| Enum Value | YAML String | Description |
|------------|-------------|-------------|
| `KP` | `kp` | Proportional gain |
| `KD` | `kd` | Derivative gain |
| `JOINT_POSITION` | `target/joint/position` | Target joint position |
| `JOINT_VELOCITY` | `target/joint/velocity` | Target joint velocity |
| `JOINT_TORQUES` | `target/joint/torques` | Target joint torques |
| `BODY_POSITION` | `target/body/position` | Target body position |
| `BODY_LINEAR_ACCELERATION` | `target/body/linear_acceleration` | Target body linear acceleration |
| `BODY_ORIENTATION` | `target/body/orientation` | Target body orientation |
| `BODY_LINEAR_VELOCITY` | `target/body/linear_velocity` | Target body linear velocity |
| `BODY_ANGULAR_ACCELERATION` | `target/body/angular_acceleration` | Target body angular acceleration |

```python
from leapp.utils.enums import OutputKindEnum

TensorSemantics("torques", action, kind=OutputKindEnum.JOINT_TORQUES)
TensorSemantics("kp_gains", kp, kind=OutputKindEnum.KP)
```

> **Note:** While LEAPP does not enforce using `InputKindEnum` exclusively for inputs or `OutputKindEnum` exclusively for outputs, it is strongly recommended to follow this convention for clarity. The `kind` field accepts any enum value.

### `element_names`

The `element_names` field provides human-readable names for the elements along each dimension of a tensor. This is useful for documenting what each element in a tensor represents (e.g., which joint corresponds to index 3).

The canonical format is `List[List[str]]`, where the outer list corresponds to tensor dimensions and each inner list names the elements in that dimension. However, LEAPP accepts several shorthand formats and normalizes them automatically:

| Input Format | Normalized To | Use Case |
|---|---|---|
| `"hip"` | `[["hip"]]` | Single named element |
| `["hip", "knee", "ankle"]` | `[["hip", "knee", "ankle"]]` | Flat list — names for one dimension |
| `[["batch"], ["x", "y", "z"]]` | `[["batch"], ["x", "y", "z"]]` | Already canonical — per-dimension names |
| `[None, None, ["r", "g", "b"]]` | `[None, None, ["r", "g", "b"]]` | Partial — only name specific dimensions |

```python
# Name elements along the last dimension
TensorSemantics("joint_pos", tensor,
                 element_names=["hip", "knee", "ankle", "shoulder", "elbow", "wrist"])

# Name elements per dimension (e.g., for a [batch, 3] tensor)
TensorSemantics("position", tensor,
                 element_names=[None, ["x", "y", "z"]])

# Name a single element
TensorSemantics("gravity", tensor, element_names="z")
```

## Passing Conventions

TensorSemantics are passed as a **single object** or as a **list**. They cannot be placed inside a dict — use the standard dict format for raw tensors and the list format for TensorSemantics. The only supported top-level formats are a dict of named raw tensors, a single `TensorSemantics`, or a list of `TensorSemantics`. Bare top-level tensors and other unnamed top-level collections are not supported.

```python
# ✅ Single TensorSemantics
annotate.input_tensors(
    "node",
    TensorSemantics("pos", tensor, kind=InputKindEnum.JOINT_POSITION),
)

# ✅ List of TensorSemantics
annotate.input_tensors("node", [
    TensorSemantics("pos", pos_tensor, kind=InputKindEnum.JOINT_POSITION),
    TensorSemantics("vel", vel_tensor, kind=InputKindEnum.JOINT_VELOCITY),
])

# ✅ Regular dict (no semantic metadata)
annotate.input_tensors("node", {"pos": pos_tensor, "vel": vel_tensor})

# ❌ TensorSemantics inside a dict — NOT supported
annotate.input_tensors("node", {
    "pos": TensorSemantics("pos", pos_tensor, kind=InputKindEnum.JOINT_POSITION),
})

# ❌ Mixing TensorSemantics and raw tensors — NOT supported
# use multiple calls to input_tensors in this case
annotate.input_tensors("node", [
    TensorSemantics("pos", pos_tensor, kind=InputKindEnum.JOINT_POSITION),
    vel_tensor,  # raw tensor mixed with semantics
])
```

## Limitations

1. **`input_tensors` and `output_tensors` only** — Semantic annotations are not available for `@annotate.method()` nodes. These nodes derive their I/O descriptions automatically from function signatures and traced values.

2. **No mixing** — When using TensorSemantics, all items must be TensorSemantics. You cannot mix raw tensors and TensorSemantics in the same list.

3. **No dict wrapping** — TensorSemantics must be passed directly or in a list. Each TensorSemantics carries its own name, so the dict key is unnecessary.

4. **Name uniqueness** — Each TensorSemantics' `name` must be unique within the same node's inputs (or outputs). Duplicate names will raise an error.

5. **Semantic fields are optional** — All semantic fields (`kind`, `element_names`) are optional. A TensorSemantics with no semantic fields behaves identically to passing a raw tensor with the same name.
