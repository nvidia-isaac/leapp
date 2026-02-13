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


### Method 1: Traced Tensors Pattern

TracedTensors provide the most flexible approach, allowing you to programmatically capture tensor operations without decorators or context managers. This is especially useful for dynamic workflows or when integrating with existing code.

```python
import torch
from leapp import annotate

annotate.start(name="my_graph")

# Create traced inputs - returns TracedTensor objects that record operations
joint_pos, joint_vel = annotate.input_tensors('preprocessing', {
    'joint_pos': torch.randn(12),
    'joint_vel': torch.randn(12)
})

# Perform operations - all ops are automatically traced
normalized_pos = joint_pos / 3.14
normalized_vel = joint_vel / 10.0
combined = torch.cat([normalized_pos, normalized_vel])

# Mark outputs and specify export format
annotate.output_tensors('preprocessing', {
    'features': combined
}, export_with="jit")

# Chain to another node - traced tensors automatically connect nodes
features = annotate.input_tensors('inference', {'features': combined})
predictions = features @ torch.randn(24, 3)  # Simple linear transform
annotate.output_tensors('inference', {'predictions': predictions}, export_with="onnx")

annotate.stop()
annotate.compile_graph()
```

### Method 2: Decorator Pattern

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

### Method 3: Context Manager Pattern

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


#### State Tensors

State tensors are traced tensors that are **both inputs AND outputs** - useful for history buffers, running statistics, or recurrent states.

```python
# After input_tensors, add state tensors to the same node
obs = annotate.input_tensors("policy", {"observation": torch.randn(4)})

# Single state returns TracedTensor directly (multiple returns tuple)
running_mean = annotate.state_tensors("policy", {"running_mean": torch.zeros(4)})

# Update state (if not called, state passes through unchanged)
new_mean = 0.9 * running_mean + 0.1 * obs
annotate.update_state("policy", {"running_mean": new_mean})

# State outputs (running_mean_out) added automatically
annotate.output_tensors("policy", {"action": obs - new_mean}, export_with="jit")
```

State outputs use `{name}_out` suffix for ONNX SSA compliance. State inputs and outputs are tagged with the same `leapp_tag`, enabling automatic feedback connection detection in the graph.

#### Auto Buffer Tracking (Stateful Neural Nets)

For models that store hidden state as registered buffers (GRU, LSTM, custom RNNs), LEAPP can **auto-detect** which buffers are mutated during forward — no annotations needed inside the model:

```python
import torch.nn as nn

class GRUPolicy(nn.Module):
    """Standard PyTorch model — no LEAPP imports."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(16, 32, num_layers=1, batch_first=False)
        self.mlp = nn.Linear(32, 8)
        self.register_buffer("h_state", torch.zeros(1, 1, 32))

    def forward(self, obs):
        gru_out, h_out = self.gru(obs.unsqueeze(0), self.h_state)
        self.h_state = h_out  # reassignment — detected by LEAPP
        return self.mlp(gru_out.squeeze(0))
```

Export with `annotate.module()`:

```python
model = GRUPolicy()
model.eval()

annotate.start("my_graph", save_path=".")
obs_traced = annotate.input_tensors("policy", {"obs": obs})

annotate.module("policy", model)
action = model(obs_traced)

annotate.output_tensors("policy", {"action": action}, export_with="onnx-torchscript")
annotate.stop()
annotate.compile_graph()
```

**How it works**: `annotate.module()` replaces registered buffers with TracedTensor inputs. When `output_tensors()` compiles the graph, it auto-detects which buffers were reassigned (mutated) during forward. Mutated buffers become state outputs with automatic feedback connections. Non-mutated buffers (e.g. normalizer mean/var) are baked as constants in the exported model, preserving their trained values.

**Requirements**:
- Hidden states must be registered as buffers via `register_buffer()` in the model's `__init__`.
- The forward pass must use *reassignment* (`self.h = h_out`) to update state. In-place mutation (`self.h.copy_(h_out)`) is not detected — use the explicit `state_tensors()`/`update_state()` API for in-place patterns.
- Use `export_with="onnx-torchscript"` for models containing `nn.GRU` or `nn.LSTM`.
- Optionally pass `buffer_names=["h_state"]` to `module()` to track only specific buffers.

See `examples/stateful_gru_export.py` for a complete runnable example.

## API Reference

### ExportManager

#### import
```python
from leapp import annotate  # Singleton, export manager
```

#### Flow Control methods
- `start(name, save_path=".", verbose=False, dry_run=False, patch_numpy=True)`: Begin tracing mode with graph name
  - `dry_run` (bool): If True, skips model compilation and export. This is used to verify graph structure and i/o. Defaults to False.
  - `patch_numpy` (bool): If True, patches torch numpy functions for TracedTensor compatibility. Defaults to True but under some cases this is known to cause errors. If not needed try setting this to false.
- `stop()`: End tracing mode


- `compile_graph(visualize=True, verbose=None, merge_nodes=MergeCfgEnum.NO_MERGE, validate=True, rtol=1e-3, atol=1e-5, strict=True)`: Generate final graph and exports
  - `visualize` (bool): Generate graph visualization. Defaults to True.
  - `verbose` (bool | None): Override verbose logging. Defaults to None (unchanged).
  - `merge_nodes` (MergeCfgEnum): Node merging strategy — `NO_MERGE` (default), `MERGE_ALL`, or `MERGE_SEQUENTIAL`.
  - `validate` (bool): If True, validates exported models against captured outputs. Defaults to True.
  - `rtol` (float): Relative tolerance for validation. Defaults to 1e-3.
  - `atol` (float): Absolute tolerance for validation. Defaults to 1e-5.
  - `strict` (bool): If True, raises an exception when validation fails. Defaults to True.

#### Annotations
- `method(**params)`: Decorator for functions/methods
- `block(name, **params)`: Context manager for code blocks
- `input_tensors(tensors_dict, node_name)`: Create traced tensor inputs for a node
- `output_tensors(node_name, tensors_dict, **params)`: Mark traced tensor outputs and finalize a node
- `state_tensors(node_name, tensors_dict)`: Create state tensors (both input and output) for a node
- `update_state(node_name, tensors_dict)`: Set output values for state tensors
- `module(node_name, model, buffer_names=None)`: Register a module for automatic stateful buffer tracking

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
