# leapp-visualization Package Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the reusable graph layout/rendering code into a sibling `leapp-visualization` Python package while keeping `leapp.compile_graph(visualize=True)` working by default.

**Architecture:** `leapp_visualization` owns generic graph dataclasses, layered layout, geometry, SVG, PNG, and `render_graph`. LEAPP keeps an adapter that converts LEAPP nodes and tensor descriptors into `leapp_visualization.VisualGraph`, then calls the generic renderer.

**Tech Stack:** Python 3.11+, setuptools, uv, `fast-sugiyama`, Pillow, pytest.

## Global Constraints

- The root `leapp` package depends on `leapp-visualization` by default.
- `leapp-visualization` must not depend on PyTorch, ONNX, ONNX Runtime, or LEAPP internals.
- `compile_graph(visualize=True)` continues to emit both `<graph_name>.svg` and `<graph_name>.png`.
- No new graph layout dependency beyond the existing `fast-sugiyama`.
- Visualization errors continue to fail graph compilation when `visualize=True`.

---

### Task 1: Create the sibling visualization package

**Files:**
- Create: `packages/leapp-visualization/pyproject.toml`
- Create: `packages/leapp-visualization/README.md`
- Create: `packages/leapp-visualization/src/leapp_visualization/__init__.py`
- Move from: `leapp/leapp_graph/visualization/model.py`
- Move from: `leapp/leapp_graph/visualization/layout.py`
- Move from: `leapp/leapp_graph/visualization/geometry.py`
- Move from: `leapp/leapp_graph/visualization/svg_renderer.py`
- Move from: `leapp/leapp_graph/visualization/png_renderer.py`
- Move from: `leapp/leapp_graph/visualization/visualize.py`

**Interfaces:**
- Produces: `leapp_visualization.VisualGraph`, `VisualNode`, `VisualPort`, `VisualTerminal`, `VisualEdge`, `render_graph(graph: VisualGraph, save_path: str, graph_name: str) -> tuple[str, str]`
- Consumes: existing generic renderer implementation from `leapp/leapp_graph/visualization`

- [ ] **Step 1: Create package metadata**

Create `packages/leapp-visualization/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "leapp-visualization"
version = "0.5.2"
description = "Reusable static layered graph visualization for LEAPP and similar pipeline graphs"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.11"
dependencies = [
    "fast-sugiyama>=0.5.3",
    "Pillow>=10.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["leapp_visualization*"]
```

- [ ] **Step 2: Move generic modules**

Move the generic modules into `packages/leapp-visualization/src/leapp_visualization/`. Rename `visualize.py` to `render.py` and rename its public function from `visualize_graph` to `render_graph`.

- [ ] **Step 3: Remove LEAPP logging from the generic renderer**

In `packages/leapp-visualization/src/leapp_visualization/render.py`, keep only generic rendering:

```python
from __future__ import annotations

import os

from .geometry import resolve_geometry
from .layout import compute_layered_layout
from .model import VisualGraph
from .png_renderer import write_png
from .svg_renderer import write_svg


def render_graph(graph: VisualGraph, save_path: str, graph_name: str) -> tuple[str, str]:
    layout = compute_layered_layout(graph)
    geometry = resolve_geometry(graph, layout, graph_name)

    svg_path = os.path.join(save_path, f"{graph_name}.svg")
    png_path = os.path.join(save_path, f"{graph_name}.png")
    write_svg(svg_path, geometry)
    write_png(png_path, geometry)
    return svg_path, png_path
```

- [ ] **Step 4: Export the public API**

In `packages/leapp-visualization/src/leapp_visualization/__init__.py`, export:

```python
from .model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal
from .render import render_graph

__all__ = [
    "VisualEdge",
    "VisualGraph",
    "VisualNode",
    "VisualPort",
    "VisualTerminal",
    "render_graph",
]
```

- [ ] **Step 5: Verify the package imports**

Run: `PYTHONPATH=packages/leapp-visualization/src uv run python -c "from leapp_visualization import VisualGraph, render_graph; print(VisualGraph, render_graph)"`

Expected: command exits with code 0 and prints the class/function objects.

### Task 2: Add the LEAPP adapter and dependency wiring

**Files:**
- Create: `leapp/leapp_graph/visualization_adapter.py`
- Modify: `leapp/leapp_graph/graph_gui.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `leapp_visualization.render_graph(graph, save_path, graph_name) -> tuple[str, str]`
- Produces: `leapp.leapp_graph.graph_gui.visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name) -> None`

- [ ] **Step 1: Move LEAPP-specific builder logic into the adapter**

Create `leapp/leapp_graph/visualization_adapter.py` with the existing `build_visual_graph(...)` implementation, but import models from `leapp_visualization`:

```python
from leapp_visualization import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal
```

Keep helper behavior for `_shape_to_tuple`, `_kind_to_string`, `_find_port_id`, and `_node_sort_key`.

- [ ] **Step 2: Keep graph_gui as LEAPP's compatibility entrypoint**

Update `leapp/leapp_graph/graph_gui.py` to:

```python
from leapp.utils.logging import _get_logger
from leapp_visualization import render_graph

from .visualization_adapter import build_visual_graph


def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    graph = build_visual_graph(nodes, connections, feedback_connections, inputs, outputs)
    svg_path, png_path = render_graph(graph, save_path, graph_name)
    _get_logger().info(f"Graph visualization saved as: {svg_path}")
    _get_logger().info(f"Graph visualization saved as: {png_path}")
```

- [ ] **Step 3: Make LEAPP depend on the sibling package by default**

Update root `pyproject.toml`:

```toml
dependencies = [
    "leapp-visualization==0.5.2",
    "torch>=2.6.0",
    "PyYAML>=6.0",
    "onnx>=1.19.0",
    "onnxruntime>=1.20.0",
    "onnxscript>=0.1.0",
    "safetensors>=0.5.0",
]

[tool.uv.sources]
leapp-visualization = { path = "packages/leapp-visualization" }
```

Remove `fast-sugiyama` and `Pillow` from root dependencies because they belong to `leapp-visualization`.

- [ ] **Step 4: Refresh the lockfile**

Run: `uv lock`

Expected: lockfile includes a local source entry for `leapp-visualization`.

### Task 3: Update tests to use package boundaries

**Files:**
- Modify: `tests/unit_tests/test_graph_visualization_imports.py`
- Modify: `tests/unit_tests/test_graph_visualization_layout.py`
- Modify: `tests/unit_tests/test_graph_visualization_geometry.py`
- Modify: `tests/unit_tests/test_graph_visualization_svg.py`
- Modify: `tests/unit_tests/test_graph_visualization_png.py`
- Modify: `tests/unit_tests/test_graph_visualization_builder.py`
- Modify: `tests/unit_tests/test_graph_visualization_integration.py`

**Interfaces:**
- Consumes: `leapp_visualization` public and module-level APIs
- Produces: tests that verify generic renderer code does not import through `leapp.leapp_graph.visualization`

- [ ] **Step 1: Update generic renderer imports**

Replace imports like:

```python
from leapp.leapp_graph.visualization.model import VisualGraph
```

with:

```python
from leapp_visualization.model import VisualGraph
```

For public import tests, assert:

```python
from leapp_visualization import VisualGraph, render_graph
```

- [ ] **Step 2: Update builder imports**

In builder/adapter tests, import:

```python
from leapp.leapp_graph.visualization_adapter import build_visual_graph
```

- [ ] **Step 3: Update visualization failure monkeypatch**

In the integration failure test, monkeypatch:

```python
import leapp.leapp_graph.graph_gui as graph_gui
monkeypatch.setattr(graph_gui, "render_graph", explode)
```

where `explode(graph, save_path, graph_name)` raises `RuntimeError("visualization exploded")`.

- [ ] **Step 4: Run focused visualization tests**

Run: `uv run pytest tests/unit_tests/test_graph_visualization_*.py -q`

Expected: all graph visualization unit tests pass.

### Task 4: Remove old package path and validate distributions

**Files:**
- Delete: `leapp/leapp_graph/visualization/__init__.py`
- Delete: `leapp/leapp_graph/visualization/model.py`
- Delete: `leapp/leapp_graph/visualization/layout.py`
- Delete: `leapp/leapp_graph/visualization/geometry.py`
- Delete: `leapp/leapp_graph/visualization/svg_renderer.py`
- Delete: `leapp/leapp_graph/visualization/png_renderer.py`
- Delete: `leapp/leapp_graph/visualization/visualize.py`
- Modify: `docs/design/graph-visualization-autolayout.md`
- Modify: `docs/adr/0003-static-layered-graph-visualization.md`

**Interfaces:**
- Consumes: updated imports from Tasks 1-3
- Produces: a root package that no longer vendors the generic visualization code under `leapp.leapp_graph.visualization`

- [ ] **Step 1: Remove the old visualization package**

Delete `leapp/leapp_graph/visualization/` after all imports point to `leapp_visualization` or `visualization_adapter`.

- [ ] **Step 2: Update design docs**

Update docs to say the static renderer lives in the sibling `leapp-visualization` package while LEAPP provides the adapter.

- [ ] **Step 3: Run the approved verification command**

Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_*.py \
  tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback \
  tests/test_examples/test_getting_started.py::TestGettingStarted::test_getting_started_execution \
  tests/test_examples/test_feedback_example.py::TestFeedbackExample::test_feedback_example_execution -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Build both distributions**

Run:

```bash
uv build packages/leapp-visualization
uv build
```

Expected: both builds exit with code 0 and produce wheels/sdists under their `dist/` directories.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add pyproject.toml uv.lock packages/leapp-visualization leapp/leapp_graph tests docs
git commit --no-gpg-sign -m "Split graph visualization into leapp-visualization package"
```

