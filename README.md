# LEAPP - Lightweight Export Annotations for Policy Pipelines

A Python package for tracing and exporting computational graphs from PyTorch code. Convert your multi-step computations into exportable models with automatic graph generation. LEAPP is specifically designed for robotics and autonomous agent applications where efficient policy export is crucial.

## What is LEAPP?

**LEAPP** stands for **Lightweight Export Annotations for Policy Pipelines**. It's designed to make it easy to export and deploy complex policy pipelines for robotics, autonomous vehicles, and other embodied AI systems. LEAPP excels at capturing entire computational graphs composed of multiple interconnected policies and models.

## Features

- 🎯 **Export to Multiple Formats**: Torchscript, ONNX, and more
- 🔧 **BYOM (Bring Your Own Model)**: Works with preexported models - simply load and integrate your existing trained models
- 📊 **Automatic Graph Visualization**: Generate graph diagrams
- 📝 **YAML Specifications**: Complete graph metadata for deployment
- 🧩 **Flexible API**: Annotate functions, code snippets, or even individual tensors
- 🤖 **Policy Pipelines**: Designed for complex multi-component policies that conain pre and post processing
- ⚡ **Lightweight**: Minimal code insertions to capture entire compute graphs

## Installation

### Pip Command
pip install leapp --index-url https://__token__:<your_personal_token>@gitlab-master.nvidia.com/api/v4/projects/202237/packages/pypi/simple
You will need a gitlab personal access token.

## Documentation

For more detailed documentation, examples, and guides, see the `docs/` directory in this repository.

## Quick Start

The recommended way to explore this package is by running the included examples:

```bash
python examples/wbc_obj.py
```

This demonstrates a complete robotics control pipeline that processes sensor data, runs neural network inference, and exports the entire computational graph.

## Usage

LEAPP provides three methods for marking nodes in your computational graph:
- **Decorators**: Use `@annotate.method()` to mark entire functions/methods as nodes
- **Context Managers**: Use `with annotate.block()` to mark code blocks as nodes
- **Traced Tensors**: Use `annotate.input_tensors()` and `annotate.output_tensors()` to programmatically define nodes by tracking tensor operations

### Method 1: Decorator Pattern

```python
import torch
from leapp import annotate

@annotate.method(export_with="jit")
def process_data(input_tensor):
    # Your computation here
    result = input_tensor * 2 + 1
    return result

# Start tracing
annotate.start(name="my_graph")
# Run your functions
output = process_data(torch.randn(10))
# Stop tracing and compile graph
annotate.stop()
annotate.compile_graph()
```

### Method 2: Context Manager Pattern

```python
import torch
from leapp import annotate

annotate.start(name="my_graph")

# Example data
raw_data = torch.randn(100, 10)
model = torch.nn.Linear(64, 3)

with annotate.block("preprocessing",
                     inputs=["raw_data"],
                     outputs=["processed_data"],
                     export_with="jit"):
    processed_data = raw_data.normalize()
    processed_data = processed_data.reshape(-1, 64)

with annotate.block("inference",
                     inputs=["processed_data"],
                     outputs=["predictions"],
                     export_with="jit"):
    predictions = model(processed_data)

annotate.stop()
annotate.compile_graph()
```

### Method 3: Traced Tensors Pattern

TracedTensors provide the most flexible approach, allowing you to programmatically capture tensor operations without decorators or context managers. This is especially useful for dynamic workflows or when integrating with existing code.

```python
import torch
from leapp import annotate

annotate.start(name="my_graph")

# Create traced inputs - returns TracedTensor objects that record operations
joint_pos, joint_vel = annotate.input_tensors({
    'joint_pos': torch.randn(12),
    'joint_vel': torch.randn(12)
}, 'preprocessing')

# Perform operations - all ops are automatically traced
normalized_pos = joint_pos / 3.14
normalized_vel = joint_vel / 10.0
combined = torch.cat([normalized_pos, normalized_vel])

# Mark outputs and specify export format
annotate.output_tensors('preprocessing', {
    'features': combined
}, export_with="jit")

# Chain to another node - traced tensors automatically connect nodes
features = annotate.input_tensors({'features': combined}, 'inference')
predictions = features @ torch.randn(24, 3)  # Simple linear transform
annotate.output_tensors('inference', {'predictions': predictions}, export_with="onnx")

annotate.stop()
annotate.compile_graph()
```

**Key features of Traced Tensors:**
- Supports complex nested inputs (dicts, lists, tuples of tensors)
- Automatically prunes unused inputs from the exported model
- Works seamlessly with `@annotate.method()` and `annotate.block()` in the same graph
- Supports both PyTorch and ONNX export backends

## API Reference

### ExportManager

#### import
```python
from leapp import annotate  # Singleton, export manager
```

#### Flow Control methods
- `start(name, save_path=".", verbose = False)`: Begin tracing mode with graph name
- `stop()`: End tracing mode
- `compile_graph(visualize=True)`: Generate final graph and exports, set visualize to false to skip graph generation

#### Annotations
- `method(**params)`: Decorator for functions/methods
- `block(name, **params)`: Context manager for code blocks
- `input_tensors(tensors_dict, node_name)`: Create traced tensor inputs for a node
- `output_tensors(tensors_dict, node_name, **params)`: Mark traced tensor outputs and finalize a node

#### Annotations Parameters
- `node_name`: name of the node to generate
- `export_with`: Export format ("torch", "onnx")
- `inputs`: Input variable names (for context manager)
- `outputs`: Output variable names (for context manager)
- `environment_constants`: Variables to treat as constants
- `register_buffers` : mutable variables to register to the buffer.

## Output Files

Running `compile_graph()` generates:
- **`{graph_name}.yaml`**: Complete graph specification
- **`{graph_name}_graph.png`**: Visual graph representation
- **Individual model files**: Exported models for each function

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.5.0
- PyYAML ≥ 6.0
- matplotlib ≥ 3.5.0
- networkx ≥ 2.6
- onnx
- onnxscript

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This software is proprietary to NVIDIA and is protected by copyright and other intellectual property rights.
