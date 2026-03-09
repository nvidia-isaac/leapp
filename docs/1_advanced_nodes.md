# Advanced Node Patterns with LEAPP

This guide covers advanced node patterns in the current public LEAPP API.
The primary interface is still `annotate.input_tensors()` / `annotate.output_tensors()`.
Use `annotate.method()` later when a node fits cleanly inside one function and you want a shorthand.

## Distributed `input_tensors()`: one node, many call sites

`input_tensors()` and `output_tensors()` are the most flexible way to mark node boundaries.
Each node should normally have one finalizing `output_tensors()` call, but it may have multiple `input_tensors()` calls with the same `node_name`.

This is useful when a node collects data from multiple helpers, methods, or files before producing an output.

```python
import torch
import leapp
from leapp import annotate

def get_lidar_data(env):
    lidar_data = env.get("lidar_data")
    return annotate.input_tensors("sensor_fusion", {"lidar_data": lidar_data})

def get_camera_features(env):
    camera_features = env.get("camera_features")
    return annotate.input_tensors("sensor_fusion", {"camera_features": camera_features})

def run_pipeline(env, model):
    leapp.start(name="distributed_inputs_example")

    lidar = get_lidar_data(env)
    camera = get_camera_features(env)

    fused = torch.cat([lidar, camera], dim=-1)
    annotate.output_tensors("sensor_fusion", {"model_input": fused}, export_with="jit")

    leapp.stop()
    leapp.compile_graph()
```

Both `input_tensors()` calls reference the same node name, so LEAPP treats them as one node with multiple inputs.

## `annotate.method()` as a shorthand

`annotate.method()` is still part of the public API, but it should be thought of as a convenience wrapper over the traced-tensor path.

```python
import torch
from leapp import annotate

@annotate.method(export_with="jit", node_name="preprocess")
def preprocess(obs):
    return (obs - obs.mean()) / (obs.std() + 1e-6)
```

Use it when:
- one function naturally defines the node input and output boundary
- parameter names and return values already match the API you want to export

Prefer `input_tensors()` / `output_tensors()` when:
- node logic spans multiple helpers
- you want explicit tensor names
- you need explicit state handling

## Explicit recurrent state with `state_tensors()` and `update_state()`

For recurrent or feedback-like node-local state, use the explicit state APIs.

```python
import torch
import leapp
from leapp import annotate

leapp.start("stateful_node")

obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 16)})
h = annotate.state_tensors("policy", {"h": torch.zeros(1, 32)})

h_next = torch.tanh(torch.cat([obs, h], dim=-1))[..., :32]
annotate.update_state("policy", {"h": h_next})
annotate.output_tensors("policy", {"action_features": h_next}, export_with="jit")

leapp.stop()
leapp.compile_graph()
```

Key points:
- `state_tensors()` creates values that are both node inputs and node outputs
- `update_state()` sets the next value of that state
- if you omit `update_state()` for a registered state tensor, LEAPP treats it as passthrough state

## Automatic module buffer tracking with `annotate.module()`

If a stateful `nn.Module` already stores state in registered buffers, `annotate.module()` can track that automatically.

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
        self.h = h_next
        return h_next

model = StatefulPolicy().eval()

leapp.start("module_state_example")
obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 16)})
annotate.module("policy", model)
action = model(obs)
annotate.output_tensors("policy", {"action": action}, export_with="onnx-torchscript")
leapp.stop()
leapp.compile_graph()
```

Use `annotate.module()` when you want LEAPP to discover buffer-based feedback automatically.
It tracks reassignment like `self.h = h_next`, not in-place updates like `self.h.copy_(h_next)`.

See `examples/stateful_gru_export.py` for a complete working example using a GRU policy with `annotate.module()` and `export_with="onnx-torchscript"`.

## `annotate.register_buffer()` for fixed-location tensors inside a traced node

`annotate.register_buffer()` is primarily for tensors that already exist at a fixed location and should participate in tracing.
The common use case is a preallocated tensor that you update in place to avoid repeated allocations, for example:

```python
obs_buffer[:] = observation
action = policy(obs_buffer)
```

Without `annotate.register_buffer()`, that in-place write is just a raw tensor mutation and LEAPP cannot trace it as part of the node. Registering the buffer wraps that preallocated tensor so operations like `buffer[:] = traced_input` are traced too.

```python
import torch
import leapp
from leapp import annotate

class PolicyWrapper:
    def __init__(self):
        self.obs_buffer = torch.zeros(6)  # fixed storage, reused each call

    def forward(self, obs):
        self.obs_buffer = annotate.register_buffer(
            "policy",
            {"obs_buffer": self.obs_buffer},
        )
        self.obs_buffer[:] = obs
        return self.obs_buffer * 2.0

wrapper = PolicyWrapper()

leapp.start("buffer_example")

obs = annotate.input_tensors("policy", {"obs": torch.randn(6)})
action = wrapper.forward(obs)
annotate.output_tensors("policy", {"action": action}, export_with="jit")

leapp.stop()
leapp.compile_graph()
```

Use `register_buffer()` for:
- preallocated tensors that are updated with in-place writes
- fixed-location staging buffers used before calling another function or module
- cases where you need `tensor[:] = traced_value` to become part of the traced computation

Important notes:
- Call `input_tensors()` first so the traced node already exists
- Reassign the return value back to the attribute or variable you will mutate
- The original payload passed to `register_buffer()` must be raw tensor data, not already-traced tensors

Use `state_tensors()` or `annotate.module()` when the value should behave like explicit recurrent feedback across calls.

## Static Outputs: Constant Output Tensors

Sometimes a node needs to output a **constant tensor that is not derived from any input**. Passing it as a regular output will fail because LEAPP expects all outputs to be traced computations. The `static_outputs` parameter on `output_tensors()` handles this case:

```python
import torch
import leapp
from leapp import annotate

leapp.start(name="static_example")

input_tensor = torch.tensor([1.0, 2.0, 3.0])
traced_input = annotate.input_tensors('my_node', {'input': input_tensor})

# Computed output — derived from the traced input
computed_output = traced_input + 1.0

# Static output — a constant, NOT derived from any input
static_tensor = torch.tensor([4.0, 5.0, 6.0])

annotate.output_tensors(
    'my_node',
    {'computed': computed_output},            # regular traced outputs
    static_outputs={'static': static_tensor}, # constant outputs
    export_with="jit"
)

leapp.stop()
leapp.compile_graph()
```

The exported model will return both outputs: `computed` (input-dependent) and `static` (always `[4, 5, 6]`).

**Key rules:**
- Static outputs must be **raw `torch.Tensor`** values. Using a `TracedTensor` (anything derived from `input_tensors()`) as a static output will raise an error.
- If you pass a single tensor without a dict, LEAPP assigns the default name `static_output` and logs a warning. Always prefer a named dict.
- Static outputs are merged with the regular outputs in the compiled model — downstream nodes can consume them like any other output.

## Nested Data Connections

LEAPP can track data connections through complex nested structures. Each individual tensor within nested dictionaries, lists, or custom objects is tracked separately.

### Example: Handling Complex Data Structures

```python
import torch
import leapp
from leapp import annotate

@annotate.method(export_with="jit", node_name="process_robot_state")
def process_robot_state(state_dict):
    """Process complex robot state dictionary."""
    # LEAPP tracks each tensor independently
    processed_state = {
        'position': state_dict['position'] * 2.0,
        'velocity': state_dict['velocity'] + 1.0,
        'sensors': {
            'lidar': state_dict['sensors']['lidar'].mean(dim=1),
            'camera': state_dict['sensors']['camera'].flatten()
        }
    }
    return processed_state

def main():
    leapp.start(name="nested_data_example")
    
    # Complex nested input
    robot_state = {
        'position': torch.tensor([1.0, 2.0, 3.0]),
        'velocity': torch.tensor([0.5, 0.5, 0.0]),
        'sensors': {
            'lidar': torch.randn(360, 3),
            'camera': torch.randn(3, 224, 224)
        }
    }
    
    processed = process_robot_state(robot_state)
    
    # LEAPP creates connections for each tensor path:
    # - robot_state['position'] -> processed_state['position']
    # - robot_state['velocity'] -> processed_state['velocity']
    # - robot_state['sensors']['lidar'] -> processed_state['sensors']['lidar']
    # - robot_state['sensors']['camera'] -> processed_state['sensors']['camera']
    
    p = annotate.input_tensors("decision_maker", {"processed": processed})

    # You can access nested structures naturally
    position_factor = p['position'].norm()
    velocity_factor = p['velocity'].sum()
    sensor_confidence = p['sensors']['lidar'].std()

    action = torch.stack([position_factor, velocity_factor, sensor_confidence])
    annotate.output_tensors("decision_maker", {"action": action}, export_with="jit")
    
    leapp.stop()
    leapp.compile_graph()
```

### How LEAPP Handles Nested Structures

When LEAPP detects a complex nested data structure (dicts, lists, tuples) as an input or output, it automatically:

1. **Flattens the structure**: Each individual tensor within the nested structure is extracted and tracked separately. For example, `state_dict['sensors']['lidar']` becomes a distinct input named `state_dict_sensors_lidar`.

2. **Generates an auto-interface**: LEAPP automatically generates wrapper code that:
   - **On input**: Accepts flat individual tensors and reconstructs them into the nested structure that your original code expects
   - **On output**: Takes the nested structure returned by your code and unpacks it into flat individual tensors

3. **Tracks connections at tensor level**: This flattening enables LEAPP to track data flow connections between nodes at the individual tensor level, not just at the parameter level.

**The result**: All exported nodes have simple, flat tensor interfaces (no complex nested structures), while your original code continues to work with nested structures naturally. This guarantees:
- Consistent tensor-level connection tracking across all nodes
- Compatibility with deployment frameworks that expect flat tensor inputs/outputs
- Clear visibility into exactly which tensors flow between which nodes

For example, in the code above, the `process_robot_state` node's exported model will have 4 separate tensor inputs (`state_dict_position`, `state_dict_velocity`, `state_dict_sensors_lidar`, `state_dict_sensors_camera`) rather than a single complex dictionary input.

## Summary

Diving deeper into LEAPP reveals its flexibility in handling complex situations:

- **Distributed traced inputs** let one node collect data from multiple call sites
- **Explicit state APIs** handle recurrent feedback cleanly
- **`annotate.module()`** tracks reassigned module buffers automatically
- **`annotate.register_buffer()`** handles fixed-location tensors and traced in-place writes
- **Nested data structures** are automatically tracked

Remember: LEAPP's goal is to capture your computational graph accurately. Being explicit about data dependencies and maintaining naming consistency will help avoid most issues. 