# Advanced Node Configuration with LEAPP

This guide dives deeper into specific scenarios you might encounter when using LEAPP for complex nodes. Unlike the getting started guide, this focuses on particular situations rather than end-to-end examples.

## Declaring Explicit Return Values

Sometimes you need to export a function that modifies internal state but doesn't explicitly return those values in its signature. LEAPP allows you to declare return values using the `outputs` parameter in `@annotate.method()` or `annotate.block()`, even when those values aren't part of the original function's return statement.

### Example: Exporting Internal State Variables

Consider a method that processes actions and stores them in an internal buffer, but doesn't return anything:

```python
import torch
import leapp
from leapp import annotate

class ActionProcessor:
    def __init__(self, scale: float, offset: float):
        self._scale = torch.tensor(scale)
        self._offset = torch.tensor(offset)
        self._raw_actions = torch.zeros(10)
        self._processed_actions = torch.zeros(10)
        self._clip = torch.tensor([[[-1.0, 1.0]]])
        self.cfg = type('Config', (), {'clip': True})()
    
    @annotate.method(outputs=["self._processed_actions"], export_with="jit-trace")
    def process_actions(self, actions: torch.Tensor):
        # Store the raw actions
        self._raw_actions[:] = actions
        # Apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # Clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # Note: No explicit return statement!
```

In this example:
- The original `process_actions()` method has no return statement
- By specifying `outputs=["self._processed_actions"]`, LEAPP will **internally modify** the function to create a return statement that returns `self._processed_actions`
- The exported graph node will have `self._processed_actions` as an output that can be connected to other nodes

### Use Cases

This feature is particularly useful for:

- **Stateful Classes**: Methods that update internal state variables without returning them
- **In-Place Operations**: Functions that modify tensors in-place and store results internally
- **Multi-Stage Processing**: Breaking down complex processing into methods that store intermediate results

**⚠️ Important Notes:**

- Output variables must be **assigned** within the annotated method/block
- For class methods, `self.*` variables are automatically accessible (no need to declare in `environment_constants`)
- The specified outputs must exist and be populated when the method completes execution
- Output order in the list determines the return order in the exported function

## Environment Constants: Referencing External Data

Sometimes you need to reference data from outside your annotated code block, such as pre-trained models, configuration objects, or constants. LEAPP's `environment_constants` parameter lets you explicitly declare these external dependencies.

### Example: Using an external model as a constant variable

```python
import torch
import leapp
from leapp import annotate

def process_with_external_model():
    # Load a pre-trained TorchScript model
    pretrained_model = torch.jit.load("path/to/model.pt")
    
    leapp.start(name="external_model_pipeline")
    
    # Create some input data
    sensor_input = torch.randn(10, 128)
    
    # Use environment constants to reference external variables
    with annotate.block("model_inference",
                        inputs=["sensor_input"],
                        outputs=["predictions"],
                        environment_constants=["pretrained_model"],
                        export_with="jit"):
        # LEAPP captures the external model and makes it available
        predictions = pretrained_model(sensor_input)
    
    leapp.stop()
    leapp.compile_graph()
    return predictions
```

**⚠️ Important: Environment Variable Accessibility**

Environment constants **must be accessible** in the local or global frame when the annotation block is invoked. They need to be specified if they're used anywhere in your call stack:

### Special Case: Class Member Variables

When working within a class, LEAPP automatically makes all member variables (self.*) available:

```python
class RobotProcessor:
    def __init__(self):
        self.pretrained_model = torch.jit.load("path/to/model.pt")
        self.scaling_factor = 2.5
    
    @annotate.method(export_with="jit")
    def process_data(self, input_data):
        # self.* variables are automatically available - no need to declare them!
        scaled = input_data * self.scaling_factor
        return self.pretrained_model(scaled)
```

### Freezing Loop Variables: Capturing Changing Values

Use `environment_constants` to **freeze variables that change over time** but should be treated as constants for each individual node. This is common when creating multiple nodes in a loop.

**The Problem:** Without marking them as constants, tracing would capture the wrong (later) value of the variable.

**The Solution:** Mark changing variables as `environment_constants` to freeze their value at node creation time.

```python
# Example: Splitting action tensor with changing index
idx = 0
for term_name, term in self.terms.items():
    with annotate.block(
        node_name=f"{term_name}_split",
        inputs=["action"],
        outputs=["term_actions"],
        environment_constants=['idx'],  # Freeze idx at current iteration value
        export_with="jit-trace",
    ):
        term_actions = action[:, idx : idx + term.action_dim]
    
    idx += term.action_dim  # idx changes, but each node keeps its frozen value
```

**Also works with class members that change:**

```python
class DataSlicer:
    def __init__(self):
        self.idx = 0
        self.stride = 3
    
    @annotate.method(export_with="jit", environment_constants=['self.idx', 'self.stride'])
    def get_subset(self, inputA: torch.Tensor):
        retval = inputA[self.idx:self.idx+self.stride]
        self.idx += self.stride  # Changes after trace, but traced value is frozen
        return retval
```

**Key Points:**
- Environment constants are captured at trace/export time and embedded in the exported model
- **Variables must exist in scope BEFORE the annotation block is invoked** - LEAPP cannot capture variables that don't exist yet
- **Two main use cases:**
  1. **External dependencies**: Pre-trained models, configuration objects, constants that need to be captured
  2. **Freezing changing values**: Loop variables, counters, or any value that changes but should be constant for a specific node
- **Declare ALL external variables used in your call stack** - including those used in nested function calls
- Use `environment_constants` for external variables not in self
- **Class members (self.*) are automatically available** - no declaration needed (but can be explicitly listed if you want to freeze their changing values)
- The values must be serializable for the export format you're using

## Distributed input_tensors: Marking a node with a diverse source

The `input_tensors` and `output_tensors` API provides a high degree of freedom when marking node bounds. While each node can only have 1 call to `output_tensors`, there can be as many calls as needed to `input_tensors`. This is crucial for capturing a node with inputs from multiple sources spanning different functions or files.

```python
import torch
import leapp
from leapp import annotate

def get_lidar_data(env):
    # imagine this function is bound to some object that data from the environment
    # some environment setup this is not tracable because it involves the environment
    lidar_data = env.get('lidar_data')
    #reference the same node name
    lidar_data = annotate.input_tensors('sensor_fusion', {'lidar_data': lidar_data}) 
    return lidar_data

def get_camera_features(env):
    # imagine this function is bound to some object that data from the environment
    # some environment setup this is not tracable because it involves the environment
    camera_features = env.get('camera_features')
    #reference the same node name
    camera_features = annotate.input_tensors('sensor_fusion', {'camera_features': camera_features})
    return camera_features

def run_pipeline():
    leapp.start(name="distributed_inputs_example")
    model_inputs = []
    for feature in preconfigured_features:
        # pulls data from get_lidar_data and get_camera_data
        # possibly does some other processing before returning
        model_inputs.append(feature.get(feature)) 
    
    # all operatons for all the nodes in features are captured
    concatenated = torch.cat(features)
    # Single output call closes the node. both inputs are now part of the same node
    annotate.output_tensors('sensor_fusion', {'model_input': concatenated}, export_with="jit")
    
    output = model(concatenated)
    
    leapp.stop()
    leapp.compile_graph()
```

Both `input_tensors` calls reference the same node name (`'sensor_fusion'`), so they're combined into a single node with two inputs despite being in seperate locations.

## Register Buffers: Persistent State in Modules

Register buffers are persistent tensors that are part of a module's state but are not parameters (they don't require gradients). They're useful for maintaining running statistics, normalization factors, or any persistent state across forward passes.

### Example: Maintaining Running Statistics

```python
import torch
import torch.nn as nn
import leapp
from leapp import annotate

class RobotController:
    def __init__(self):
        # Initialize buffers for running statistics
        self.running_mean = torch.zeros(3)
        self.running_count = torch.tensor(0)
        self.action_history = torch.zeros(10, 3)
    
    def process_sensors(self, sensor_data):
        leapp.start(name="stateful_controller")
        
        with annotate.block("update_statistics",
                            inputs=["sensor_data"],
                            outputs=["normalized_data"],
                            register_buffers=["self.running_mean", 
                                            "self.running_count",
                                            "self.action_history"],
                            export_with="jit"):
            # Update running mean
            self.running_count += 1
            alpha = 1.0 / self.running_count
            self.running_mean = (1 - alpha) * self.running_mean + alpha * sensor_data.mean(dim=0)
            
            # Normalize using running statistics
            normalized_data = (sensor_data - self.running_mean) / (self.running_mean.std() + 1e-6)
            
            # Shift action history
            self.action_history = torch.roll(self.action_history, -1, dims=0)
            self.action_history[-1] = normalized_data[:3]
        
        leapp.stop()
        leapp.compile_graph()
        return normalized_data
```

**Important:** Register buffers uses PyTorch's `register_buffer()` method. For more details on PyTorch buffers, see [the official documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer).

**Key Differences:**
- `environment_constants`: Values frozen at export time, won't change
- `register_buffers`: Mutable state that persists across calls, will be updated

### Traced Tensor Pattern: `annotate.register_buffer()`

The `register_buffers` parameter shown above works with decorators and context managers. For traced tensor nodes (`input_tensors` / `output_tensors`), use `annotate.register_buffer()` instead to make a pre-existing tensor participate in tracing. This is necessary when you need **in-place assignment** (`tensor[:] = ...`) on a tensor that was not returned by `input_tensors()`.

```python
import torch
import leapp
from leapp import annotate

class Module:
    def __init__(self):
        self.values = torch.tensor([1.0, 2.0, 3.0])

    def run(self, traced_input):
        # Register the buffer — wraps self.values as a TracedTensor
        buffers = annotate.register_buffer('my_node', {'values': self.values})
        self.values = buffers['values']

        # In-place assignment is now traced
        self.values[:] = traced_input
        # Subsequent operations are also traced
        return self.values * 100.0

module = Module()

leapp.start(name="buffer_example")

input_tensor = torch.tensor([4.0, 5.0, 6.0])
traced_input = annotate.input_tensors('my_node', {'input': input_tensor})

result = module.run(traced_input)

annotate.output_tensors('my_node', {'result': result}, export_with="jit")

leapp.stop()
leapp.compile_graph()
```

**Key rules:**
- `input_tensors()` **must** be called first to create the node before calling `register_buffer()`.
- The returned dict contains traced wrappers — you **must** reassign them back (e.g. `self.values = buffers['values']`) for subsequent operations to be recorded.
- The buffer tensors must be **raw `torch.Tensor`** values, not already-traced tensors.

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
    
    with annotate.block("decision_maker",
                        inputs=["processed"],
                        outputs=["action"],
                        export_with="jit"):
        # You can access nested structures naturally
        position_factor = processed['position'].norm()
        velocity_factor = processed['velocity'].sum()
        sensor_confidence = processed['sensors']['lidar'].std()
        
        action = torch.tensor([position_factor, velocity_factor, sensor_confidence])
    
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

## IO Reconciliation Pitfalls

One of the most common issues in LEAPP is name mismatches during IO reconciliation. Downstream graph generation libraries sometimes require exact name matches to connect outputs to inputs. LEAPP will automatically change connected i/o to the same name if their names don't match. 

This process can cause problems and not all can be automatically detected by LEAPP. 

### Understanding the Problem

IO reconciliation happens when LEAPP tries to connect nodes in your graph. Names are generated from:
1. **Function parameters** for decorated methods
2. **Return Values** in your Python code
3. **Declared inputs/outputs** in annotation blocks

if these names don't match, during graph detection, LEAPP automatically changes selects one of them and makes them the same.

### Common Pitfall Examples

```python
@annotate.method()
def funcA(input: torch.Tensor):
    detections = torch.zeros(input.shape)
    #some processing
    return detections

@annotate.method()
def funcB(detections):
    retval = torch.tensor([])
    #some processing
    return retval
@annotate.method()
def funcC(input, detections):
    retval = torch.tensor([])
    return retval

leapp.start(name = "failed example")
detections = funcA(data)
funcBreturn = funcB(detections)
funcCreturn = funcC(detections, data)
leapp.stop()
leapp.compile_graph() #failure on this line
```
#### Why does this fail?
this example passes the output of funcA `detections` to `funcB` and `funcC`. `funcC` function signature register detections as input while also containing a detections field. During io reconciliation, LEAPP tries to update the `funcC` input name to detections which causes an unresolvable conflict with the other input. 

### Best Practices to Avoid Issues
The best way to avoid these issues is to avoid io reconciliation altogether. For that we should try to use clear and consistent naming throughout. Reconciliation is for cases where that is not possible and as a last resort.
```python
# GOOD: Consistent naming throughout the pipeline
```python
@annotate.method()
def funcA(input: torch.Tensor):
    detections = torch.zeros(input.shape)
    #some processing
    return detections

@annotate.method()
def funcB(detections):
    retval = torch.tensor([])
    #some processing
    return functionB_retval
@annotate.method()
def funcC(detections, data)
    retval = torch.tensor([])
    return retval

leapp.start(name = "failed example")
detections = funcA(data)
funcBreturn = funcB(detections)
funcCreturn = funcC(detections, data)
leapp.stop()
leapp.compile_graph() #failure on this line

```


### Debugging IO Reconciliation

When you see the warning about IO names being changed:

1. **Check the generated YAML** to see what names LEAPP assigned
2. **Look at the error message** - it often shows the generated forward method signature
3. **Ensure consistency** between:
   - Function parameter names
   - Declared input/output names in blocks
   - Actual variable usage in your code

   
## Summary

Diving deeper into LEAPP reveals its flexibility in handling complex situations:

- **Environment constants** for external dependencies
- **Register buffers** for persistent mutable state
- **Nested data structures** are automatically tracked
- **IO reconciliation** requires careful naming consistency
- **Pre-compiled models** can be integrated using `export_with=None`

Remember: LEAPP's goal is to capture your computational graph accurately. Being explicit about data dependencies and maintaining naming consistency will help avoid most issues. 