# Getting Started with LEAPP

Welcome to LEAPP - Lightweight Export Annotations for Policy Pipelines! This guide will walk you through the basics of using LEAPP to trace and export computational graphs from your PyTorch code.

## What You'll Learn

In this guide, you'll learn how to:
- Use traced tensors to annotate input and output tensors
- Build multi-node graphs by connecting exported node outputs to later node inputs
- Understand where LEAPP writes models, YAML, and graph visualizations
- Use `annotate.method()` as a shorthand for simple self-contained functions

## Simple Example: Robot Sensor Processing Pipeline

Let's create a simple robot sensor processing pipeline using LEAPP's primary annotation path:
`annotate.input_tensors()` and `annotate.output_tensors()`.

Our pipeline will:
1. Preprocess robot observations into a feature vector
2. Run a tiny policy stage on those features

```python
import torch
import leapp
from leapp import annotate

_POS_MEAN = torch.zeros(6)
_POS_STD = torch.ones(6) * 0.5
_VEL_SCALE = 4.0
_ACTION_SCALE = 0.25
_JOINT_LIMIT = 1.0

def normalize_joints(pos: torch.Tensor, vel: torch.Tensor):
    pos_norm = (pos - _POS_MEAN) / (_POS_STD + 1e-6)
    vel_norm = vel / _VEL_SCALE
    return pos_norm, vel_norm

def capture_joint_observations(joint_pos: torch.Tensor, joint_vel: torch.Tensor):
    return annotate.input_tensors("obs_processor", {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    })

def capture_context(orientation: torch.Tensor, cmd_vel: torch.Tensor):
    return annotate.input_tensors("obs_processor", {
        "orientation": orientation,
        "cmd_vel": cmd_vel,
    })

def project_gravity(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return torch.stack([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])

def scale_and_clip(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(raw * _ACTION_SCALE, min=-_JOINT_LIMIT, max=_JOINT_LIMIT)

torch.manual_seed(42)
_W = torch.randn(18, 6) * 0.05
_b = torch.zeros(6)

def main():
    # Example robot state
    joint_pos = torch.randn(6)
    joint_vel = torch.randn(6)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cmd_vel = torch.tensor([0.5, 0.0, 0.1])

    # Start tracing our computational graph
    leapp.start(name="sample_pipeline")

    # ===== NODE 1: observation preprocessing =====
    # Multiple input_tensors() calls can contribute inputs to the same node.
    pos, vel = capture_joint_observations(joint_pos, joint_vel)
    quat, cmd = capture_context(orientation, cmd_vel)

    pos_norm, vel_norm = normalize_joints(pos, vel)
    gravity_vec = project_gravity(quat)
    obs_features = torch.cat([pos_norm, vel_norm, gravity_vec, cmd])

    annotate.output_tensors(
        "obs_processor",
        {"obs_features": obs_features},
        export_with="jit",
    )

    # ===== NODE 2: policy =====
    feat = annotate.input_tensors("policy", {
        "obs_features": obs_features,
    })
    raw_action = feat @ _W + _b
    joint_targets = scale_and_clip(raw_action)

    annotate.output_tensors(
        "policy",
        {"joint_targets": joint_targets},
        export_with="jit",
    )

    # Stop tracing and compile the graph
    leapp.stop()
    leapp.compile_graph()

if __name__ == "__main__":
    main()
```

## Understanding the Example

### 1. Traced Tensors (`annotate.input_tensors` / `annotate.output_tensors`)

```python
# Create traced inputs - returns TracedTensor objects
sensor_input = annotate.input_tensors('feature_extractor', {
    'sensor_data': clean_data
})

# All operations on TracedTensors are recorded - even through function calls!
result = some_helper_function(sensor_input)
result = result * 2 + 1

# Mark outputs to finalize the node
annotate.output_tensors('feature_extractor', {
    'result': result
}, export_with="jit")
```

Traced tensors are the backbone of all LEAPP tracing. The input_tensors/output_tensors provide the most flexible approach for capturing operations:
- **Spans function calls**: Operations through helper functions are automatically traced
- **Inline operations**: Mix function calls with inline tensor operations
- **Programmatic control**: Define nodes dynamically
- **Distributed inputs**: Call `input_tensors()` multiple times with the same `node_name` before one final `output_tensors()`


### 2. Function Decorator (`annotate.method`)

```python
@annotate.method(export_with="jit", node_name="preprocess")
def preprocess(raw_readings):
    return torch.clamp(raw_readings, min=0.0, max=1.0)
```

`annotate.method()` is a shorthand for simple self-contained functions. Internally it uses the same traced-tensor machinery as `input_tensors()` / `output_tensors()`, so LEAPP still builds a `TracedTensorNode` under the hood.

Use `annotate.method()` when:
- the node fits cleanly in one function
- function arguments and return values already define the node I/O well

Prefer `input_tensors()` / `output_tensors()` when:
- node logic spans helper functions or multiple call sites
- you want explicit control over input and output names
- you want to inject semantic data into the final configs

### 3. Graph Flow

The example demonstrates how data flows through the computational graph:

```
joint_pos, joint_vel, orientation, cmd_vel
                    ↓
             [obs_processor]
                    ↓
              obs_features
                    ↓
                [policy]
                    ↓
              joint_targets
```

### 4. Tracing Lifecycle

```python
# 1. Start tracing
leapp.start(name="sample_pipeline")

# 2. Run your annotated code
# ... your pipeline code ...

# 3. Stop tracing
leapp.stop()

# 4. Compile and export
leapp.compile_graph()
```

## Generated Output Files

After running `compile_graph()`, LEAPP generates:

- **`sample_pipeline/sample_pipeline.yaml`** - Complete graph specification with metadata
- **`sample_pipeline/sample_pipeline.png`** - Visual diagram of your computational graph
- **`sample_pipeline/*.pt`** or **`sample_pipeline/*.onnx`** - Exported models for each exported node

## Try It Yourself

1. Run the maintained example: `python examples/getting_started.py`
2. Open the generated `sample_pipeline/` directory
3. Check the generated files to see your exported pipeline!

## Understanding the Generated Output

When you run the example, LEAPP generates several files that help you understand and deploy your computational graph.

### Graph Visualization

LEAPP writes a graph visualization to `sample_pipeline/sample_pipeline.png`. The image shows:
- exported nodes
- graph inputs and outputs
- data-flow connections between nodes

This visualization is useful for verifying that LEAPP detected the node boundaries and cross-node connections you intended.

![Getting Started Graph](images/getting_started_graph.png)

### Graph Specification File

LEAPP also generates a complete specification of your pipeline in `sample_pipeline/sample_pipeline.yaml`:

```yaml
models:
  obs_processor:
    inputs:
    - dtype: float32
      name: joint_pos
      shape: [6]
      type: tensor
    outputs:
    - dtype: float32
      name: obs_features
      shape: [18]
      type: tensor
    parameters:
      backend: jit
      model_path: obs_processor.pt

pipeline:
  inputs:
    obs_processor: [joint_pos, joint_vel, orientation, cmd_vel]
  outputs:
    policy: [joint_targets]
  data_flow:
    obs_processor/obs_features: [policy/obs_features]
  feedback_flow: {}
```

This YAML file contains:
- **Complete node specifications** with input/output tensor descriptions
- **Shape and data type information** for all tensors
- **Graph structure metadata** ready for deployment systems
- **Export configuration** showing how each node should be compiled

## Next Steps

Now that you understand the basics, you can:
- Explore more complex pipelines in the [examples](../examples/) directory
- Learn about advanced features in [1_advanced_nodes.md](1_advanced_nodes.md), [2_advanced_export.md](2_advanced_export.md), and [3_advanced_graph.md](3_advanced_graph.md)
- Get detailed explanation on the api at [api.md](api.md)
- Integrate LEAPP into your existing pipelines
