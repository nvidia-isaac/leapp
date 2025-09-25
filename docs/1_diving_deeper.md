# Diving Deeper with LEAPP

This guide dives deeper into specific scenarios you might encounter when using LEAPP for complex computational graphs. Unlike the getting started guide, this focuses on particular situations rather than end-to-end examples.

## Environment Constants: Referencing External Data

Sometimes you need to reference data from outside your annotated code block, such as pre-trained models, configuration objects, or constants. LEAPP's `environment_constants` parameter lets you explicitly declare these external dependencies.

### Example: Using an external model as a constant variable

```python
import torch
from leapp import annotate

def process_with_external_model():
    # Load a pre-trained TorchScript model
    pretrained_model = torch.jit.load("path/to/model.pt")
    
    annotate.start(name="external_model_pipeline")
    
    # Create some input data
    sensor_input = torch.randn(10, 128)
    
    # Use environment constants to reference external variables
    with annotate.block("model_inference",
                        inputs=["sensor_input"],
                        outputs=["predictions"],
                        environment_constants=["pretrained_model"],
                        export_with="torch"):
        # LEAPP captures the external model and makes it available
        predictions = pretrained_model(sensor_input)
    
    annotate.stop()
    annotate.compile_graph()
    return predictions
```

**⚠️ Important: Environment Variable Accessibility**

Environment constants **must be accessible** in the local or global frame when the annotation block is invoked. They need to be specified if they're used anywhere in your call stack:

### Special Case: Class Member Variables

When working within a class, LEAPP automatically makes all member variables (self.*) available as constants:

```python
import torch
from leapp import annotate

class RobotProcessor:
    def __init__(self):
        # These member variables are automatically available in annotated blocks
        self.pretrained_model = torch.jit.load("path/to/model.pt")
        self.config_threshold = 0.7
        self.scaling_factor = 2.5
    
    @annotate.method(export_with="torch", node_name="process_with_members")
    def process_data(self, input_data):
        """Process using class member variables."""
        # All self.* variables are automatically available - no need to declare them!
        scaled = input_data * self.scaling_factor
        predictions = self.pretrained_model(scaled)
        
        # Apply threshold from class member
        return torch.where(predictions > self.config_threshold,
                          predictions,
                          torch.zeros_like(predictions))
    
    def process_with_block(self, sensor_input):
        with annotate.block("member_variable_processing",
                            inputs=["sensor_input"],
                            outputs=["result"],
                            export_with="torch"):
            # self.* variables are automatically available in blocks too!
            # No need to declare them as environment_constants
            normalized = sensor_input / self.scaling_factor
            result = self.pretrained_model(normalized)
        return result
```

**Key Points:**
- Environment constants are captured at trace time and embedded in the exported model
- **Variables must exist in scope BEFORE the annotation block is invoked** - LEAPP cannot capture variables that don't exist yet
- **Declare ALL external variables used in your call stack** - including those used in nested function calls
- Use `environment_constants` for external variables not in self
- **Class members (self.*) are automatically available** - no declaration needed
- The values must be serializable for the export format you're using

## Register Buffers: Persistent State in Modules

Register buffers are persistent tensors that are part of a module's state but are not parameters (they don't require gradients). They're useful for maintaining running statistics, normalization factors, or any persistent state across forward passes.

### Example: Maintaining Running Statistics

```python
import torch
import torch.nn as nn
from leapp import annotate

class RobotController:
    def __init__(self):
        # Initialize buffers for running statistics
        self.running_mean = torch.zeros(3)
        self.running_count = torch.tensor(0)
        self.action_history = torch.zeros(10, 3)
    
    def process_sensors(self, sensor_data):
        annotate.start(name="stateful_controller")
        
        with annotate.block("update_statistics",
                            inputs=["sensor_data"],
                            outputs=["normalized_data"],
                            register_buffers=["self.running_mean", 
                                            "self.running_count",
                                            "self.action_history"],
                            export_with="torch"):
            # Update running mean
            self.running_count += 1
            alpha = 1.0 / self.running_count
            self.running_mean = (1 - alpha) * self.running_mean + alpha * sensor_data.mean(dim=0)
            
            # Normalize using running statistics
            normalized_data = (sensor_data - self.running_mean) / (self.running_mean.std() + 1e-6)
            
            # Shift action history
            self.action_history = torch.roll(self.action_history, -1, dims=0)
            self.action_history[-1] = normalized_data[:3]
        
        annotate.stop()
        annotate.compile_graph()
        return normalized_data
```

**Important:** Register buffers behave similarly to PyTorch's `register_buffer()` method. For more details on PyTorch buffers, see [the official documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer).

**Key Differences:**
- `environment_constants`: Values frozen at export time, won't change
- `register_buffers`: Mutable state that persists across calls, will be updated

## Nested Data Connections

LEAPP can track data connections through complex nested structures. Each individual tensor within nested dictionaries, lists, or custom objects is tracked separately.

### Example: Handling Complex Data Structures

```python
import torch
from leapp import annotate

@annotate.method(export_with="torch", node_name="process_robot_state")
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
    annotate.start(name="nested_data_example")
    
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
                        export_with="torch"):
        # You can access nested structures naturally
        position_factor = processed['position'].norm()
        velocity_factor = processed['velocity'].sum()
        sensor_confidence = processed['sensors']['lidar'].std()
        
        action = torch.tensor([position_factor, velocity_factor, sensor_confidence])
    
    annotate.stop()
    annotate.compile_graph()
```

**Key Points:**
- LEAPP automatically detects and tracks individual tensors in nested structures
- Each tensor gets its own node in the computational graph
- The structure is preserved in the exported format

## IO Reconciliation Pitfalls

One of the most common issues in LEAPP is name mismatches during IO reconciliation. Downstream graph generation libraries require exact name matches to connect outputs to inputs. LEAPP will automatically change connected i/o to the same name if their names don't match. 

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
    retval = troch.tensor([])
    #some processing
    return retval
@annotate.method()
def funcC(input, detections)
    retval = torch.tensor([])
    return retval

annotate.start(name = "failed example")
detections = funcA(data)
funcBreturn = funcB(detections)
funcCreturn = funcC(detections, data)
annotate.stop()
annotate.compile_graph() #failure on this line
```
#### Why does this fail?
this example passes the output of funcA `detections` to `funcB` and `funcC`. `funcC` function signature register detections as input while also containing a detections field. During io reconciliation, LEAPP tries to update the `funcC` input name to detections which causes an unresolvable confict with the other input. 

### Best Practices to Avoid Issues
The best way to avoid these issues is to avoid io reconsiliation alltogether. For that we should try to use clear and consistent naming throuout. Reconsiliation is for cases where that is not possible and as a last resort.
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
    retval = troch.tensor([])
    #some processing
    return functionB_retval
@annotate.method()
def funcC(detections, data)
    retval = torch.tensor([])
    return retval

annotate.start(name = "failed example")
detections = funcA(data)
funcBreturn = funcB(detections)
funcCreturn = funcC(detections, data)
annotate.stop()
annotate.compile_graph() #failure on this line

```

### Debugging IO Reconciliation

When you see the warning about IO names being changed:

1. **Check the generated YAML** to see what names LEAPP assigned
2. **Look at the error message** - it often shows the generated forward method signature
3. **Ensure consistency** between:
   - Function parameter names
   - Declared input/output names in blocks
   - Actual variable usage in your code

```python
# The generated forward method will show you what LEAPP expects:
# def forward(self, sensor_data: Tensor, movement_speed: Tensor) -> Tensor:
#     # Your code must use these exact parameter names
```

## Summary

Diving deeper into LEAPP reveals its flexibility in handling complex situations:

- **Environment constants** for external dependencies
- **Register buffers** for persistent mutable state
- **Nested data structures** are automatically tracked
- **IO reconciliation** requires careful naming consistency

Remember: LEAPP's goal is to capture your computational graph accurately. Being explicit about data dependencies and maintaining naming consistency will help avoid most issues. 