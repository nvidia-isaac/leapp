# Getting Started with LEAPP

Welcome to LEAPP - Lightweight Export Annotations for Policy Pipelines! This guide will walk you through the basics of using LEAPP to trace and export computational graphs from your PyTorch code.

## What You'll Learn

In this guide, you'll learn how to:
- Use function decorators to annotate methods
- Use context managers to annotate code blocks  
- Create computational graphs that connect multiple processing stages
- Export your pipeline for deployment

## Simple Example: Robot Sensor Processing Pipeline

Let's create a simple robot sensor processing pipeline that demonstrates LEAPP's core features. Our pipeline will:
1. Process raw sensor data using a decorated function
2. Calculate movement speed using another decorated function
3. Make control decisions using an annotation block that combines outputs from both functions

```python
import torch
from leapp import annotate

# First function: Process sensor data
@annotate.method(export_with="torch", node_name="sensor_processor")
def process_sensor_data(raw_readings):
    """Process raw sensor readings and normalize them."""
    processed = torch.clamp(raw_readings, min=0.0, max=1.0)
    normalized = (processed - 0.5) * 2.0
    return normalized

# Second function: Calculate movement speed
@annotate.method(export_with="torch", node_name="speed_calculator")
def calculate_movement_speed(sensor_data):
    """Calculate safe movement speed based on sensor data."""
    obstacle_distance = torch.mean(torch.abs(sensor_data))
    safe_speed = torch.clamp(obstacle_distance, min=0.1, max=1.0)
    return safe_speed

def main():
    # Start tracing our computational graph
    annotate.start(name="sample_robot_pipeline")
    
    # Create some sample sensor data
    raw_sensor_data = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2])
    
    # Process sensor data using first decorated function
    clean_data = process_sensor_data(raw_sensor_data)
    
    # Calculate movement speed using second decorated function
    movement_speed = calculate_movement_speed(clean_data)
    
    # Annotation block: Control decisions (uses outputs from both functions)
    with annotate.block("control_decision",
                         inputs=["clean_data", "movement_speed"],
                         outputs=["robot_action"],
                         export_with="torch"):
        sensor_confidence = 1.0 - torch.std(clean_data)
        
        if sensor_confidence > 0.8:
            robot_action = torch.tensor([movement_speed.item(), 0.0, 0.0])
        else:
            robot_action = torch.tensor([0.0, 0.0, movement_speed.item() * 0.5])
    
    # Stop tracing and compile the graph
    annotate.stop()
    annotate.compile_graph()
    
    print(f"Raw sensor data: {raw_sensor_data}")
    print(f"Processed data: {clean_data}")
    print(f"Movement speed: {movement_speed}")
    print(f"Robot action: {robot_action}")
    print(f"Sensor confidence: {sensor_confidence}")

if __name__ == "__main__":
    main()
```

## Understanding the Example

### 1. Function Decorator (`@annotate.method`)

```python
@annotate.method(export_with="torch", node_name="sensor_processor")
def process_sensor_data(raw_readings):
    # Your processing logic here
    return processed_data
```

The `@annotate.method` decorator marks an entire function as a graph node. Key parameters:
- `export_with="torch"`: Export this node as TorchScript
- `node_name`: Custom name for the node in the graph

### 2. Context Manager Blocks (`annotate.block`)

```python
with annotate.block("control_decision",
                     inputs=["clean_data", "movement_speed"],
                     outputs=["robot_action"],
                     export_with="torch"):
    # Your processing logic here
    robot_action = combine_inputs(clean_data, movement_speed)
```

Context managers let you annotate specific code blocks as graph nodes. Key parameters:
- First parameter: Node name
- `inputs`: List of input variable names (can take outputs from multiple functions)
- `outputs`: List of output variable names  
- `export_with`: Export format

### 3. Graph Flow

The example demonstrates how data flows through the computational graph:

```
raw_sensor_data → [sensor_processor] → clean_data
                                          ↓
                                     ┌────┴────┐
                                     ↓         ↓
                        [speed_calculator]     │
                                     ↓         ↓
                           movement_speed → [control_decision] ← clean_data
                                                   ↓
                                              robot_action
```

### 4. Tracing Lifecycle

```python
# 1. Start tracing
annotate.start(name="sample_robot_pipeline")

# 2. Run your annotated code
# ... your pipeline code ...

# 3. Stop tracing  
annotate.stop()

# 4. Compile and export
annotate.compile_graph()
```

## Generated Output Files

After running `compile_graph()`, LEAPP generates:

- **`sample_robot_pipeline.yaml`** - Complete graph specification with metadata
- **`sample_robot_pipeline_graph.png`** - Visual diagram of your computational graph  
- **Individual model files** - Exported models for each annotated function/block

## Try It Yourself

1. Save the example code to a file (e.g., `simple_pipeline.py`)
2. Run it: `python simple_pipeline.py`
3. Check the generated files to see your exported pipeline!

## Understanding the Generated Output

When you run the example, LEAPP generates several files that help you understand and deploy your computational graph.

### Visual Graph Representation

First, let's look at the generated graph visualization:

![Sample Robot Pipeline Graph](images/sample_robot_pipeline.png)

This automatically generated diagram shows your entire computational pipeline at a glance. You can see:
- **Function nodes** (sensor_processor, speed_calculator) represented as rectangles
- **Block nodes** (control_decision) also shown as processing steps  
- **Data flow connections** showing how outputs from functions feed into blocks
- **Input/output tensors** with their names and shapes

This visual representation is invaluable for **verifying that LEAPP detected your intended graph structure correctly**. You can quickly spot if connections are missing, if nodes aren't being captured, or if the data flow doesn't match your expectations.

### Graph Specification File

LEAPP also generates a complete specification of your pipeline in `sample_robot_pipeline/sample_robot_pipeline.yaml`:

```yaml
graph_name: sample_robot_pipeline
nodes:
  sensor_processor:
    type: method
    inputs:
      - name: raw_readings
        dtype: float32
        shape: [5]
    outputs:
      - name: normalized
        dtype: float32
        shape: [5]
  
  speed_calculator:
    type: method
    inputs:
      - name: sensor_data
        dtype: float32
        shape: [5]
    outputs:
      - name: safe_speed
        dtype: float32
        shape: []
  
  control_decision:
    type: block
    inputs:
      - name: clean_data
        dtype: float32
        shape: [5]
      - name: movement_speed
        dtype: float32
        shape: []
    outputs:
      - name: robot_action
        dtype: float32
        shape: [3]
```

This YAML file contains:
- **Complete node specifications** with input/output tensor descriptions
- **Shape and data type information** for all tensors
- **Graph structure metadata** ready for deployment systems
- **Export configuration** showing how each node should be compiled

Additionally, you'll find the individual exported model files (`sensor_processor.pt`, `speed_calculator.pt`, `control_decision.pt`) ready for deployment in production environments.

## Next Steps

Now that you understand the basics, you can:
- Explore more complex pipelines in the `examples/` directory
- Learn about advanced features like environment constants and custom backends
- Integrate LEAPP into your existing robotics or AI pipelines

The key insight is that LEAPP makes it easy to take your existing PyTorch code and turn it into exportable, deployable computational graphs with minimal code changes!
