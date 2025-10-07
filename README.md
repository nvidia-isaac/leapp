# LEAPP - Lightweight Export Annotations for Policy Pipelines

A Python package for tracing and exporting computational graphs from PyTorch code. Convert your multi-step computations into exportable models with automatic graph generation. LEAPP is specifically designed for robotics and autonomous agent applications where efficient policy export is crucial.

## What is LEAPP?

**LEAPP** stands for **Lightweight Export Annotations for Policy Pipelines**. It's designed to make it easy to export and deploy complex policy pipelines for robotics, autonomous vehicles, and other embodied AI systems. LEAPP excels at capturing entire computational graphs composed of multiple interconnected policies and models.

## Features

- 🎯 **Export to Multiple Formats**: Torchscript, ONNX, and more
- 🔧 **BYOM (Bring Your Own Model)**: Works with preexported models - simply load and integrate your existing trained models
- 📊 **Automatic Graph Visualization**: Generate graph diagrams
- 📝 **YAML Specifications**: Complete graph metadata for deployment
- 🧩 **Flexible API**: Use decorators or context managers
- 🤖 **Policy Pipelines**: Designed for complex multi-component policies that conain pre and post processing
- ⚡ **Lightweight**: Minimal code insertions to capture entire compute graphs

## Installation

### Pip Command
pip install leapp --index-url https://__token__:<your_personal_token>@gitlab-master.nvidia.com/api/v4/projects/202237/packages/pypi/simple

**You will need a gitlab personal access token.**

## Documentation

For more detailed documentation, examples, and guides, see the `docs/` directory in this repository.

## Quick Start

The recommended way to explore this package is by running the included examples:

```bash
python examples/wbc_obj.py
```

This demonstrates a complete robotics control pipeline that processes sensor data, runs neural network inference, and exports the entire computational graph.

## Usage

LEAPP provides two clear methods for marking nodes in your computational graph:
- **Decorators**: Use `@annotate.method()` to mark entire functions/methods as nodes
- **Context Managers**: Use `with annotate.block()` to mark code blocks as nodes

### Method 1: Decorator Pattern

```python
import torch
from leapp import annotate

@annotate.method(export_with="torch")
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
                     export_with="torch"):
    processed_data = raw_data.normalize()
    processed_data = processed_data.reshape(-1, 64)

with annotate.block("inference",
                     inputs=["processed_data"],
                     outputs=["predictions"],
                     export_with="torch"):
    predictions = model(processed_data)

annotate.stop()
annotate.compile_graph()
```

## API Reference

### ExportManager

#### import
```python
from leapp import annotate  # Singleton, export manager
```

#### Flow Control methods
- `start(name, save_path=".", tag_io=True)`: Begin tracing mode with graph name
- `stop()`: End tracing mode
- `compile_graph(visualize=True)`: Generate final graph and exports, set visualize to false to skip graph generation

#### Annotations
- `method(**params)`: Decorator for functions/methods
- `block(name, **params)`: Context manager for code blocks

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

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This software is proprietary to NVIDIA and is protected by copyright and other intellectual property rights.
