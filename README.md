# LEAPP - Lightweight Export Annotations for Policy Pipelines

A Python package for tracing and exporting multi-step PyTorch computational graphs. Annotate your existing code with lightweight markers, and LEAPP captures the graph structure, exports each stage as an individual model, and generates a deployment-ready YAML specification.

## What is LEAPP?

**LEAPP** stands for **Lightweight Export Annotations for Policy Pipelines**. It is designed for pipelines that chain multiple PyTorch models or processing stages together — where you need to export the whole pipeline, not just a single model.

LEAPP works by tracing one real execution of your code. It records which tensors flow between stages, exports each stage independently, and writes a YAML that describes how to wire them back together at inference time.

## Features

- **Export to Multiple Formats**: TorchScript and ONNX
- **BYOM (Bring Your Own Model)**: Integrate pre-compiled models into the graph without recompiling
- **Automatic Graph Visualization**: Generate a diagram of your pipeline
- **YAML Specification**: Complete graph metadata for deployment and downstream frameworks
- **Flexible Annotation API**: Annotate at the function, code block, or tensor level
- **Lightweight**: Minimal insertions — no rewrites of existing model code required

## Installation

```
pip install leapp --index-url https://__token__:<your_personal_token>@gitlab-master.nvidia.com/api/v4/projects/202237/packages/pypi/simple
```

You will need a GitLab personal access token.

## Documentation

For detailed guides and API reference, see the `docs/` directory.

## Usage

LEAPP provides two ways to mark stages in your pipeline:

- **Decorator** — `@annotate.method()`: wrap an entire function as a node
- **Traced tensors** — `annotate.input_tensors()` / `annotate.output_tensors()`: explicitly mark tensor boundaries, including across multiple function calls

Both produce the same result: a named node in the exported graph with defined inputs, outputs, and an exported model file.

### Decorator

```python
import torch
import leapp
from leapp import annotate

@annotate.method(export_with="jit")
def preprocess(raw):
    return (raw - raw.mean()) / raw.std()

@annotate.method(export_with="jit")
def predict(features):
    return torch.relu(features @ torch.randn(16, 4))

leapp.start(name="my_pipeline")
features = preprocess(torch.randn(16))
output = predict(features)
leapp.stop()
leapp.compile_graph()
```

### Traced Tensors

Traced tensors are the most flexible option. They let you define node boundaries explicitly, including across helper functions or when inputs come from multiple call sites.

```python
import torch
import leapp
from leapp import annotate

leapp.start(name="my_pipeline")

raw = torch.randn(16)
x = annotate.input_tensors("preprocess", {"raw": raw})
features = (x - x.mean()) / x.std()
annotate.output_tensors("preprocess", {"features": features}, export_with="jit")

f = annotate.input_tensors("predict", {"features": features})
output = torch.relu(f @ torch.randn(16, 4))
annotate.output_tensors("predict", {"output": output}, export_with="jit")

leapp.stop()
leapp.compile_graph()
```

## Output Files

Running `compile_graph()` creates a directory named after your graph containing:

- **`{graph_name}.yaml`** — complete pipeline specification (node I/O, model paths, data flow)
- **`{graph_name}.png`** — visual diagram of the pipeline
- **Individual model files** — one exported model per node (`.pt` or `.onnx`)

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

- `@annotate.method(export_with, node_name, ...)` — decorator
- `annotate.input_tensors(node_name, tensors)` — begin a traced tensor node
- `annotate.output_tensors(node_name, tensors, export_with, ...)` — finalize a traced tensor node

**Common annotation parameters**

| Parameter | Description |
|---|---|
| `export_with` | Export backend: `"jit"`, `"onnx"`, `"onnx-torchscript"`, or `None` (pre-compiled) |
| `node_name` | Name of the node in the graph |
| `environment_constants` | External variables to bake in as constants |

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
- safetensors ≥ 0.4.0

## License

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0.
