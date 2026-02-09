# Semantic Data Annotation in LEAPP

This guide covers how to add semantic metadata to your tensors using `TensorDescription`. Semantic annotations let you describe **what** a tensor represents (e.g., joint positions, target torques) and provide element-level naming, making generated YAML specifications self-documenting and enabling downstream consumers to interpret the data correctly.

> **Note:** Semantic annotation is only available for `annotate.input_tensors()` and `annotate.output_tensors()`. It is not supported for `@annotate.method()` or `annotate.block()` nodes.

## Basic Usage

Instead of passing raw tensors to `input_tensors` or `output_tensors`, wrap them in `TensorDescription` objects. Pass them as a single `TensorDescription` or a list of `TensorDescription`s:

```python
import torch
from leapp import annotate, TensorDescription
from leapp.utils.enums import inputKindEnum, outputKindEnum

joint_pos = torch.randn(1, 12)
joint_vel = torch.randn(1, 12)

annotate.start("my_robot")

# Pass a list of TensorDescriptions as inputs
traced_pos, traced_vel = annotate.input_tensors([
    TensorDescription("joint_pos", joint_pos,
                       kind=inputKindEnum.JOINT_POSITION,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
    TensorDescription("joint_vel", joint_vel,
                       kind=inputKindEnum.JOINT_VELOCITY),
], "policy")

# Compute output
command = traced_pos + traced_vel

# Pass a list of TensorDescriptions as outputs
annotate.output_tensors("policy", [
    TensorDescription("command", command,
                       kind=outputKindEnum.JOINT_TORQUES,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
])

annotate.stop()
annotate.compile_graph()
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

#### `inputKindEnum` — for input tensors

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
| `COMMAND_JOINT_TORQUES` | `command/joint/torques` | Commanded joint torques reference |

```python
from leapp.utils.enums import inputKindEnum

TensorDescription("joint_pos", tensor, kind=inputKindEnum.JOINT_POSITION)
TensorDescription("imu_gyro", tensor, kind=inputKindEnum.BODY_ANGULAR_VELOCITY)
TensorDescription("target_pos", tensor, kind=inputKindEnum.COMMAND_JOINT_POSITION)
```

#### `outputKindEnum` — for output tensors

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
from leapp.utils.enums import outputKindEnum

TensorDescription("torques", traced_output, kind=outputKindEnum.JOINT_TORQUES)
TensorDescription("kp_gains", traced_output, kind=outputKindEnum.KP)
```

> **Note:** While LEAPP does not enforce using `inputKindEnum` exclusively for inputs or `outputKindEnum` exclusively for outputs, it is strongly recommended to follow this convention for clarity. The `kind` field accepts any enum value.

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
TensorDescription("joint_pos", tensor,
                   element_names=["hip", "knee", "ankle", "shoulder", "elbow", "wrist"])

# Name elements per dimension (e.g., for a [batch, 3] tensor)
TensorDescription("position", tensor,
                   element_names=[None, ["x", "y", "z"]])

# Name a single element
TensorDescription("gravity", tensor, element_names="z")
```

## Passing Conventions

TensorDescriptions are passed as a **single object** or as a **list**. They cannot be placed inside a dict — use the standard dict format for raw tensors and the list format for TensorDescriptions.

```python
# ✅ Single TensorDescription
annotate.input_tensors(
    TensorDescription("pos", tensor, kind=inputKindEnum.JOINT_POSITION),
    "node"
)

# ✅ List of TensorDescriptions
annotate.input_tensors([
    TensorDescription("pos", pos_tensor, kind=inputKindEnum.JOINT_POSITION),
    TensorDescription("vel", vel_tensor, kind=inputKindEnum.JOINT_VELOCITY),
], "node")

# ✅ Regular dict (no semantic metadata)
annotate.input_tensors({"pos": pos_tensor, "vel": vel_tensor}, "node")

# ❌ TensorDescriptions inside a dict — NOT supported
annotate.input_tensors({
    "pos": TensorDescription("pos", pos_tensor, kind=inputKindEnum.JOINT_POSITION),
}, "node")

# ❌ Mixing TensorDescriptions and raw tensors — NOT supported
annotate.input_tensors([
    TensorDescription("pos", pos_tensor, kind=inputKindEnum.JOINT_POSITION),
    vel_tensor,  # raw tensor mixed with TDs
], "node")
```

## Limitations

1. **`input_tensors` and `output_tensors` only** — Semantic annotations are not available for `@annotate.method()` or `annotate.block()` decorated functions. These nodes derive their I/O descriptions automatically from function signatures and traced values.

2. **No mixing** — When using TensorDescriptions, all items must be TensorDescriptions. You cannot mix raw tensors and TensorDescriptions in the same list.

3. **No dict wrapping** — TensorDescriptions must be passed directly or in a list. Each TensorDescription carries its own name, so the dict key is unnecessary.

4. **Name uniqueness** — Each TensorDescription's `name` must be unique within the same node's inputs (or outputs). Duplicate names will raise an error.

5. **Semantic fields are optional** — All semantic fields (`kind`, `element_names`) are optional. A TensorDescription with no semantic fields behaves identically to passing a raw tensor with the same name.
