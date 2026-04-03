# LEAPP - Lightweight Export Annotations for Policy Pipelines

A Python package for tracing and exporting multi-step PyTorch computational graphs. Annotate your existing code with lightweight markers, and LEAPP captures the graph structure, exports each stage as an individual model, and generates a deployment-ready YAML specification.

## What is LEAPP?

**LEAPP** stands for **Lightweight Export Annotations for Policy Pipelines**. It is designed for pipelines that chain multiple PyTorch models or processing stages together — where you need to export the whole pipeline, not just a single model.

LEAPP works by tracing real execution of your code. It records which tensors flow between stages, exports each stage independently, and writes a YAML that describes how to wire them back together at inference time. Unlike other solutions, LEAPP is designed to be non-invasive and eliminates the need to have another export-specific implementation of your processing.

## Features

- **Export to Multiple Formats**: TorchScript and ONNX
- **BYOM (Bring Your Own Model)**: Integrate pre-compiled models into the graph without recompiling
- **YAML Specification**: Complete metadata for deployment and downstream frameworks
- **Flexible Structuring**: Easily define complex node boundaries or multi-node graph structures
- **Lightweight**: Minimal insertions — no rewrites of existing model code required, annotations safely noop if not exporting
- **Automatic Graph Visualization**: Generate a diagram of your pipeline

## Installation

pip: You will need a GitLab personal access token.
```
pip install leapp --index-url https://__token__:<your_personal_token>@gitlab-master.nvidia.com/api/v4/projects/202237/packages/pypi/simple
```
from source:
```
# clone this repository
cd <repository location> && pip install .
```

## Documentation

For detailed guides and API reference, see the `docs/` directory.

## Usage



LEAPP operates using traced datatypes. Traced datatypes extend the original datatype and silently add tracing capabilities to them. The result the ability to define node boundaries explicitly, including across helper functions or when inputs come from multiple call sites.

```python
import torch
import leapp
import env # mock environment setup
from leapp import annotate

def get_obs1_from_env():
    obs1 = env.get_from_env('obs')
    obs1 = annotate.input_tensors("preprocess", {"obs1_name": obs1}) #inserted annotation
    # **some processing**
    return obs1

def get_obs2_from_env():
    obs2 = env.get_from_env('obs2')
    obs2 = annotate.input_tensors("preprocess", {"obs2_name": obs2}) #inserted annotation
    # **some processing**
    return obs2


leapp.start(name="my_pipeline")

obs1 = get_obs1_from_env()
obs2 =  get_obs2_from_env()

# Simple feature construction from the two observation tensors.
obs1_centered = obs1 - obs1.mean()
obs2_scaled = obs2 / (obs2.abs().max() + 1e-6)
features = torch.cat([obs1_centered, obs2_scaled], dim=-1)

annotate.output_tensors("preprocess", {"processed_features": features}, export_with="jit")  #inserted annotation

featues = annotate.input_tensors("predict", {"features": features}) #inserted annotation
output = torch.relu(featues @ torch.randn(16, 4))
annotate.output_tensors("predict", {"output": output}, export_with="onnx") #inserted annotation

leapp.stop()
leapp.compile_graph()
```

On top of simply tracing inputs and outputs, leapp provides various annotations to accomodate a variety of policies. Please refer to `docs/api.md` for more details. 

## Output Files

Running `compile_graph()` creates a directory named after your graph containing:

- **`{graph_name}.yaml`** — complete pipeline specification (node I/O, model paths, data flow)
- **`{graph_name}.png`** — visual diagram of the pipeline
- **Individual model files** — one exported model per node (`.pt`, `.onnx`, etc...)

## API Reference

```python
import leapp
from leapp import annotate
```

**Lifecycle**

- `leapp.start(name, save_path=".", verbose=False, dry_run=False, non_traced=None, max_cached_io=5, global_patching=True)`
- `leapp.stop()`
- `leapp.compile_graph(visualize=True, verbose=None, validate=True, dry_run=False, rtol=1e-3, atol=1e-5, strict=True)`

**Annotations**

- `annotate.input_tensors(node_name, tensors)` — begin a traced tensor node
- `annotate.output_tensors(node_name, tensors, export_with, ...)` — finalize a traced tensor node
- `annotate.method(export_with, node_name, ...)` — decorator, shorthand for annotating i/o with function structure
- `annotate.state_tensors(node_name, tensors)` — declare recurrent/state inputs
- `annotate.update_state(node_name, tensors)` — update recurrent state outputs
- `annotate.register_buffer(node_name, tensors)` — register traced persistent buffers for in-place writes
- `annotate.module(node_name, model, ...)` — trace an `nn.Module` with buffer tracking
- `annotate.mirror_leapp_tags(source, target)` — copy tracing tags between tensors

See `docs/api.md` for the full reference and `docs/` for advanced features.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.6.0
- PyYAML ≥ 6.0
- matplotlib ≥ 3.5.0
- networkx ≥ 2.6
- onnx ≥ 1.19.0
- onnxruntime ≥ 1.20.0
- onnxscript ≥ 0.1.0
- safetensors ≥ 0.5.0

## License

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0.
