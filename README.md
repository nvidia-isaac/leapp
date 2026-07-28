<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP — Lightweight Export Annotations for Policy Pipelines

LEAPP is a Python package for tracing and exporting multi-step PyTorch computational graphs. Annotate your existing code with lightweight markers, and LEAPP captures the graph structure, exports each stage as an individual model, and generates a deployment-ready YAML specification.

LEAPP is designed for pipelines that chain multiple PyTorch models or processing stages together — where you need to export the whole pipeline, not just a single model. It traces real execution of your code, records which tensors flow between stages, exports each stage independently, and writes a YAML that describes how to wire everything back together at inference time.

## Installation

```bash
pip install leapp
```

## Quick example

```python
import torch
import leapp
from leapp import annotate

leapp.start(name="my_pipeline")

x = annotate.input_tensors("preprocess", {"obs": torch.randn(8)})
features = (x - x.mean()) / (x.std() + 1e-6)
annotate.output_tensors("preprocess", {"features": features}, export_with="jit")

features = annotate.input_tensors("policy", {"features": features})
action = torch.tanh(features @ torch.randn(8, 4))
annotate.output_tensors("policy", {"action": action}, export_with="onnx")

leapp.stop()
leapp.compile_graph()
```

Running this produces a `my_pipeline/` directory containing the exported models, a YAML pipeline spec, and, on Python 3.11 or later, a PNG graph visualization. On Python 3.10, LEAPP warns and skips visualization while completing the rest of the export.

## Documentation

Full guides and API reference live at **[https://nvidia-isaac.github.io/leapp/](https://nvidia-isaac.github.io/leapp/)**:

- [Getting started](https://nvidia-isaac.github.io/leapp/getting_started.html) — install, first pipeline, generated output
- [Guides](https://nvidia-isaac.github.io/leapp/guides/index.html) — node patterns, export options, feedback graphs, semantic data, runtime validation
- [API reference](https://nvidia-isaac.github.io/leapp/api/index.html)

## License

Copyright (c) 2026, NVIDIA Corporation & affiliates. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [thirdparty.txt](thirdparty.txt).
