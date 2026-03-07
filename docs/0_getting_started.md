# Getting Started with LEAPP

Welcome to LEAPP - Lightweight Export Annotations for Policy Pipelines! This guide will walk you through the basics of using LEAPP to trace and export computational graphs from your PyTorch code.

## What You'll Learn

In this guide, you'll learn how to:
- Use function decorators to annotate methods
- Use traced tensors to annotate input and output tensors
- Create computational graphs that connect multiple processing stages
- Export your pipeline for deployment

## Simple Example: Robot Sensor Processing Pipeline

Let's create a simple robot sensor processing pipeline that demonstrates LEAPP's two annotation methods. Our pipeline will:
1. Process raw sensor data using a **decorated function** (`@annotate.method`)
2. Extract navigation features using **traced tensors** (`annotate.input_tensors` / `annotate.output_tensors`)

```python
import torch
import leapp
from leapp import annotate

# Method node: Process and normalize sensor data
@annotate.method(export_with="jit")
def process_sensor_data(raw_readings):
    """Process raw sensor readings and normalize them."""
    processed = torch.clamp(raw_readings, min=0.0, max=1.0)
    normalized = (processed - 0.5) * 2.0
    return normalized

# Helper function for feature extraction (called within traced tensor context)
def compute_features(sensor_data):
    """Compute features from sensor data."""
    distance = torch.mean(torch.abs(sensor_data))
    variance = torch.var(sensor_data)
    return distance, variance

def main():
    # Start tracing our computational graph
    leapp.start(name="sample_pipeline")

    # Create some sample sensor data
    raw_sensor_data = torch.tensor([0.1, 0.8, 0.3, 0.9, 0.2])

    # ===== NODE 1: Method decorator =====
    # Process sensor data using decorated function
    clean_data = process_sensor_data(raw_sensor_data)

    # ===== NODE 2: Traced tensors =====
    # Extract features using traced tensors
    # This allows us to trace operations across function calls
    sensor_input = annotate.input_tensors('feature_extractor', {
        'sensor_data': clean_data
    })

    # Operations are automatically traced - even through helper functions!
    distance, variance = compute_features(sensor_input)

    # Additional inline operations are also traced
    safe_speed = torch.clamp(distance, min=0.1, max=1.0)
    confidence = 1.0 / (1.0 + variance)

    # Mark outputs to finalize the traced node
    annotate.output_tensors('feature_extractor', {
        'safe_speed': safe_speed,
        'confidence': confidence
    }, export_with="jit")

    # Stop tracing and compile the graph
    leapp.stop()
    leapp.compile_graph()

    print(f"Raw sensor data: {raw_sensor_data}")
    print(f"Processed data: {clean_data}")
    print(f"Safe speed: {safe_speed}")
    print(f"Confidence: {confidence}")

if __name__ == "__main__":
    main()
```

## Understanding the Example

### 1. Function Decorator (`@annotate.method`)

```python
@annotate.method(export_with="jit")
def process_sensor_data(raw_readings):
    # Your processing logic here
    return processed_data
```

The `@annotate.method` decorator marks an entire function as a graph node. Key parameters:
- `export_with`: Export format ("jit" for TorchScript, "onnx" for ONNX)
- `node_name`: Optional custom name for the node (defaults to function name)

### 2. Traced Tensors (`annotate.input_tensors` / `annotate.output_tensors`)

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

Traced tensors provide the most flexible approach for capturing operations:
- **Spans function calls**: Operations through helper functions are automatically traced
- **Inline operations**: Mix function calls with inline tensor operations
- **Programmatic control**: Define nodes dynamically without decorators

### 3. Graph Flow

The example demonstrates how data flows through the computational graph:

```
raw_sensor_data → [process_sensor_data] → clean_data
                                              ↓
                                    [feature_extractor]
                                       ↓          ↓
                                 safe_speed   confidence
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

- **`sample_pipeline.yaml`** - Complete graph specification with metadata
- **`sample_pipeline.png`** - Visual diagram of your computational graph
- **Individual model files** - Exported models for each annotated function

## Try It Yourself

1. Save the example code to a file (e.g., `simple_pipeline.py`)
2. Run it: `python simple_pipeline.py`
3. Check the generated files to see your exported pipeline!

## Understanding the Generated Output

When you run the example, LEAPP generates several files that help you understand and deploy your computational graph.

### Visual Graph Representation

First, let's look at the generated graph visualization:

![Sample Robot Pipeline Graph](images/robot_pipeline.png)

This automatically generated diagram shows your entire computational pipeline at a glance. You can see:
- **Method nodes** (process_sensor_data) created from decorated functions
- **Traced tensor nodes** (feature_extractor) capturing programmatic tensor operations
- **Data flow connections** showing how outputs from nodes feed into subsequent nodes
- **Input/output tensors** with their names and shapes

This visual representation is invaluable for **verifying that LEAPP detected your intended graph structure correctly**. You can quickly spot if connections are missing, if nodes aren't being captured, or if the data flow doesn't match your expectations.

### Graph Specification File

LEAPP also generates a complete specification of your pipeline in `sample_pipeline/sample_pipeline.yaml`:

```yaml
models:
  feature_extractor:
    inputs:
    - dtype: float32
      name: sensor_data
      shape: [5]
      type: tensor
    outputs:
    - dtype: float32
      name: safe_speed
      shape: []
      type: tensor
    - dtype: float32
      name: confidence
      shape: []
      type: tensor
    parameters:
      backend: jit
      model_path: feature_extractor.pt

pipeline:
  inputs:
    process_sensor_data: [raw_readings]
  outputs:
    feature_extractor: [safe_speed, confidence]
  data_flow:
    process_sensor_data/sensor_data: [feature_extractor/sensor_data]
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
- Learn about advanced features in [1_advanced_nodes.md](1_advanced_nodes.md) and [3_advanced_graph.md](3_advanced_graph.md)
- Get detailed explanation on the api at [api.md](api.md)
- Integrate LEAPP into your existing pipelines
