<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# leapp-visualization

Static layered graph layout and PNG rendering for pipeline-style directed graphs.

This package is intentionally independent from LEAPP runtime/export internals,
PyTorch, and ONNX. LEAPP consumes it through a thin adapter and ships the
modules as part of the `leapp` distribution; this tree keeps its own layout and
`pyproject.toml` so the visualization stack can still be developed or installed
on its own.

## Install

```bash
# Python 3.11+
pip install -e ./packages/leapp-visualization
```

Requires `fast-sugiyama` and `Pillow`.

## Usage

```python
from leapp_visualization import VisualGraph, render_graph

png_path = render_graph(graph, ".", "pipeline")
```

## Layout

- `src/leapp_visualization/` — visual model, layout, geometry, PNG rendering
- `tests/` — package tests (run via `pytest packages/leapp-visualization/tests`)
