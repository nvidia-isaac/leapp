# Graph Visualization Auto-Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LEAPP's interactive Matplotlib/NetworkX graph visualization with a deterministic, static, automatically laid out visualization that emits both `<graph_name>.svg` and `<graph_name>.png` from `leapp.compile_graph(visualize=True)`.

**Architecture:** Keep the existing `LeappGraph.visualize()` call boundary, replace `graph_gui.py` with a thin compatibility wrapper, and move rendering into `leapp/leapp_graph/visualization/` as five internal layers: visual model builder, `fast-sugiyama` layout adapter, geometry resolver, SVG renderer, and Pillow PNG renderer.

**Tech Stack:** Python 3.11+, `fast-sugiyama>=0.5.3` for layered graph layout, `Pillow>=10.0.0` for PNG drawing, standard-library dataclasses/typing/XML escaping, pytest for regression coverage.

## Global Constraints

- Keep `leapp.compile_graph(visualize=True)` as the user-facing API.
- Always emit both SVG and PNG when visualization is enabled.
- Make SVG the primary artifact and PNG a raster companion generated from the same resolved geometry.
- Static only: no JavaScript, browser automation, Qt, Graphviz binary, Node, Java, or Cairo dependency.
- Render graph inputs, graph outputs, LEAPP nodes, forward edges, and feedback edges.
- Render ports inside each LEAPP node:
  - input ports on the left
  - output ports on the right
  - label includes tensor name, shape, dtype, and semantic `kind` when present
- Feedback edges must be visually distinct and excluded from forward rank assignment.
- Preserve current YAML/model export behavior and pipeline schema.
- Use ASCII in new source files.
- Keep intermediate commits testable: add the new dependency path first, replace the old visualizer only after the new renderers have unit coverage.
- Use `git commit --no-gpg-sign` because this repository has `commit.gpgsign=true` configured locally.

---

## Task 1: Add Dependency Direction And Visualization Package Shell

**Files:**

- `pyproject.toml`
- `leapp/leapp_graph/visualization/__init__.py`
- `leapp/leapp_graph/visualization/model.py`
- `tests/unit_tests/test_graph_visualization_imports.py`

**Intent:** Establish the Python 3.11 dependency floor and create importable visualization package boundaries without changing the active renderer yet.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_imports.py`:

```python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

def test_visualization_package_exports_public_entrypoint():
    from leapp.leapp_graph.visualization import visualize_graph

    assert callable(visualize_graph)


def test_visual_model_types_are_importable():
    from leapp.leapp_graph.visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal

    assert VisualGraph.__name__ == "VisualGraph"
    assert VisualNode.__name__ == "VisualNode"
    assert VisualPort.__name__ == "VisualPort"
    assert VisualTerminal.__name__ == "VisualTerminal"
    assert VisualEdge.__name__ == "VisualEdge"
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_imports.py -q
```

- [ ] Expected before implementation: fail with `ModuleNotFoundError` or `ImportError` for `leapp.leapp_graph.visualization`.

**Implementation Steps:**

- [ ] Update `pyproject.toml`:
  - `requires-python = ">=3.11"`
  - classifiers keep only Python 3.11 and 3.12 from the currently listed version-specific Python classifiers.
  - add dependencies:

```toml
    "fast-sugiyama>=0.5.3",
    "Pillow>=10.0.0",
```

  - keep `matplotlib` and `networkx` for this task so existing `graph_gui.py` still imports before the replacement is wired.
- [ ] Add `leapp/leapp_graph/visualization/model.py` with these public dataclasses and helpers:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PortSide = Literal["input", "output"]
TerminalKind = Literal["graph_input", "graph_output"]
EdgeKind = Literal["forward", "feedback", "graph_input", "graph_output"]


@dataclass(frozen=True)
class VisualPort:
    id: str
    node_id: str
    side: PortSide
    name: str
    shape: tuple[str, ...]
    dtype: str
    kind: str | None = None


@dataclass(frozen=True)
class VisualNode:
    id: str
    title: str
    backend: str | None
    inputs: tuple[VisualPort, ...] = field(default_factory=tuple)
    outputs: tuple[VisualPort, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualTerminal:
    id: str
    kind: TerminalKind
    title: str
    node_id: str
    port_id: str


@dataclass(frozen=True)
class VisualEdge:
    id: str
    kind: EdgeKind
    source_id: str
    target_id: str
    source_port_id: str | None = None
    target_port_id: str | None = None
    label: str = ""


@dataclass(frozen=True)
class VisualGraph:
    nodes: tuple[VisualNode, ...]
    terminals: tuple[VisualTerminal, ...]
    edges: tuple[VisualEdge, ...]

    def visual_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes) + tuple(terminal.id for terminal in self.terminals)


def visual_id(*parts: object) -> str:
    normalized = [str(part).replace("/", ":").replace(" ", "_") for part in parts]
    return ":".join(normalized)
```

- [ ] Add `leapp/leapp_graph/visualization/__init__.py`:

```python
from .model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


def visualize_graph(*args, **kwargs):
    from .visualize import visualize_graph as _visualize_graph

    return _visualize_graph(*args, **kwargs)


__all__ = [
    "VisualEdge",
    "VisualGraph",
    "VisualNode",
    "VisualPort",
    "VisualTerminal",
    "visualize_graph",
]
```

- [ ] Add `leapp/leapp_graph/visualization/visualize.py` with a temporary entrypoint that raises a clear implementation error only if called:

```python
def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    raise RuntimeError("LEAPP graph visualization renderer is not wired yet")
```

  This module is intentionally not imported by `leapp/leapp_graph/graph_gui.py` yet.

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_imports.py -q
```

- [ ] Expected after implementation: `2 passed`.
- [ ] Run:

```bash
uv run python - <<'PY'
import matplotlib
import networkx
import PIL
import fast_sugiyama
print("visualization deps import")
PY
```

- [ ] Expected after `uv` syncs dependencies: prints `visualization deps import`.

**Commit:**

- [ ] Commit:

```bash
git add pyproject.toml leapp/leapp_graph/visualization tests/unit_tests/test_graph_visualization_imports.py
git commit --no-gpg-sign -m "Add graph visualization package shell"
```

---

## Task 2: Build The LEAPP Visual Model

**Files:**

- `leapp/leapp_graph/visualization/builder.py`
- `tests/unit_tests/test_graph_visualization_builder.py`

**Intent:** Convert LEAPP's current in-memory `nodes`, `connections`, `feedback_connections`, `graph_inputs`, and `graph_outputs` into a deterministic, port-aware `VisualGraph`.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_builder.py`:

```python
from dataclasses import dataclass

from leapp.leapp_graph.visualization.builder import build_visual_graph


@dataclass
class FakeTensorDescription:
    name_str: str
    dtype: str
    shape: tuple[int, ...]
    tag: str | None = None
    semantics: dict | None = None

    def get_semantics(self):
        return self.semantics or {}


@dataclass
class FakeNode:
    name: str
    inputs: list[FakeTensorDescription]
    outputs: list[FakeTensorDescription]
    backend: str | None = "jit-script"
    node_index: int = 0


def test_build_visual_graph_preserves_ports_semantics_and_external_io():
    policy = FakeNode(
        name="policy",
        inputs=[FakeTensorDescription("obs", "float32", (1, 12), semantics={"kind": "state"})],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), tag="policy/action", semantics={"kind": "command"})],
        backend="onnx-dynamo",
        node_index=0,
    )
    clamp = FakeNode(
        name="clamp",
        inputs=[FakeTensorDescription("raw_action", "float32", (1, 4), tag="policy/action")],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), tag="clamp/action")],
        backend="jit-script",
        node_index=1,
    )
    connections = [
        {
            "source": {"node": policy, "idx": 0},
            "targets": [{"node": clamp, "idx": 0}],
        }
    ]

    graph = build_visual_graph(
        nodes={"policy": policy, "clamp": clamp},
        connections=connections,
        feedback_connections=[],
        graph_inputs=["policy/obs"],
        graph_outputs=["clamp/action"],
    )

    node_by_id = {node.id: node for node in graph.nodes}
    assert tuple(node_by_id) == ("node:policy", "node:clamp")
    assert node_by_id["node:policy"].inputs[0].name == "obs"
    assert node_by_id["node:policy"].inputs[0].shape == ("1", "12")
    assert node_by_id["node:policy"].inputs[0].dtype == "float32"
    assert node_by_id["node:policy"].inputs[0].kind == "state"
    assert node_by_id["node:policy"].outputs[0].kind == "command"
    assert node_by_id["node:policy"].backend == "onnx-dynamo"

    terminal_by_id = {terminal.id: terminal for terminal in graph.terminals}
    assert terminal_by_id["terminal:input:policy:obs"].kind == "graph_input"
    assert terminal_by_id["terminal:output:clamp:action"].kind == "graph_output"

    edge_kinds = [edge.kind for edge in graph.edges]
    assert edge_kinds == ["forward", "graph_input", "graph_output"]
    assert graph.edges[0].source_port_id == "port:policy:output:action"
    assert graph.edges[0].target_port_id == "port:clamp:input:raw_action"
    assert graph.edges[0].label == "action"


def test_build_visual_graph_marks_feedback_edges_without_adding_them_to_forward_flow():
    first = FakeNode(
        name="first",
        inputs=[FakeTensorDescription("state", "float32", (2,), tag="second/state_next")],
        outputs=[FakeTensorDescription("hidden", "float32", (2,), tag="first/hidden")],
        node_index=0,
    )
    second = FakeNode(
        name="second",
        inputs=[FakeTensorDescription("hidden", "float32", (2,), tag="first/hidden")],
        outputs=[FakeTensorDescription("state_next", "float32", (2,), tag="second/state_next")],
        node_index=1,
    )

    graph = build_visual_graph(
        nodes={"first": first, "second": second},
        connections=[{"source": {"node": first, "idx": 0}, "targets": [{"node": second, "idx": 0}]}],
        feedback_connections=[{"source": {"node": second, "idx": 0}, "targets": [{"node": first, "idx": 0}]}],
        graph_inputs=[],
        graph_outputs=[],
    )

    assert [edge.kind for edge in graph.edges] == ["forward", "feedback"]
    assert graph.edges[1].source_id == "node:second"
    assert graph.edges[1].target_id == "node:first"
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_builder.py -q
```

- [ ] Expected before implementation: fail importing `leapp.leapp_graph.visualization.builder`.

**Implementation Steps:**

- [ ] Add `leapp/leapp_graph/visualization/builder.py` with these imports and public entrypoint:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal, visual_id
```

```text
build_visual_graph(nodes: Mapping[str, Any], connections: Sequence[dict], feedback_connections: Sequence[dict], graph_inputs: Sequence[str], graph_outputs: Sequence[str]) -> VisualGraph
```

- [ ] The `build_visual_graph` body must build all `VisualNode` objects first, then append forward edges, feedback edges, graph input terminals/edges, and graph output terminals/edges in that order.
- [ ] Implement concrete helper functions:
  - `_build_node(node: Any) -> VisualNode`
  - `_build_port(node_name: str, side: str, desc: Any) -> VisualPort`
  - `_shape_to_tuple(shape: Any) -> tuple[str, ...]`
  - `_kind_to_string(desc: Any) -> str | None`
  - `_find_port_id(node: Any, side: str, port_name: str) -> str`
  - `_node_sort_key(item: tuple[str, Any]) -> tuple[int, str]`
- [ ] Use `node.node_index` as the primary node order and node name as the secondary order.
- [ ] Extract backend from `node.get_backend()` when present, otherwise from `node.backend`.
- [ ] Convert enum semantic kinds by reading `.value` when available, otherwise `str(kind)`.
- [ ] Build one `VisualEdge` for each source-target pair, not one edge per node pair.
- [ ] Use stable IDs:
  - node: `visual_id("node", node.name)`
  - input port: `visual_id("port", node.name, "input", desc.name_str)`
  - output port: `visual_id("port", node.name, "output", desc.name_str)`
  - graph input terminal: `visual_id("terminal", "input", node_name, input_name)`
  - graph output terminal: `visual_id("terminal", "output", node_name, output_name)`
  - edge: `visual_id("edge", edge_kind, source_id, source_port_name, target_id, target_port_name, sequence_index)`
- [ ] Return graph edges ordered as:
  1. forward internal edges
  2. feedback internal edges
  3. graph input terminal edges
  4. graph output terminal edges

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_builder.py tests/unit_tests/test_graph_visualization_imports.py -q
```

- [ ] Expected after implementation: all listed tests pass.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/builder.py tests/unit_tests/test_graph_visualization_builder.py
git commit --no-gpg-sign -m "Build port-aware graph visualization model"
```

---

## Task 3: Add Layered Layout Adapter

**Files:**

- `leapp/leapp_graph/visualization/layout.py`
- `tests/unit_tests/test_graph_visualization_layout.py`

**Intent:** Use `fast_sugiyama.from_edges()` to compute deterministic left-to-right rank/order centers for visual nodes and graph terminals. Feedback edges are excluded from the layout edge set.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_layout.py`:

```python
from leapp.leapp_graph.visualization.layout import compute_layered_layout
from leapp.leapp_graph.visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


def _node(name):
    return VisualNode(id=f"node:{name}", title=name, backend="jit-script")


def test_layered_layout_places_pipeline_left_to_right():
    graph = VisualGraph(
        nodes=(_node("a"), _node("b")),
        terminals=(
            VisualTerminal("terminal:input:a:x", "graph_input", "x", "node:a", "port:a:input:x"),
            VisualTerminal("terminal:output:b:y", "graph_output", "y", "node:b", "port:b:output:y"),
        ),
        edges=(
            VisualEdge("e0", "graph_input", "terminal:input:a:x", "node:a", None, "port:a:input:x", "x"),
            VisualEdge("e1", "forward", "node:a", "node:b", "port:a:output:y", "port:b:input:y", "y"),
            VisualEdge("e2", "graph_output", "node:b", "terminal:output:b:y", "port:b:output:y", None, "y"),
        ),
    )

    layout = compute_layered_layout(graph)

    assert layout.centers["terminal:input:a:x"].x < layout.centers["node:a"].x
    assert layout.centers["node:a"].x < layout.centers["node:b"].x
    assert layout.centers["node:b"].x < layout.centers["terminal:output:b:y"].x
    assert all(edge_id != "e_feedback" for edge_id in layout.forward_edge_ids)


def test_layered_layout_is_deterministic_for_same_graph():
    graph = VisualGraph(
        nodes=(_node("a"), _node("b"), _node("c")),
        terminals=(),
        edges=(
            VisualEdge("e0", "forward", "node:a", "node:c"),
            VisualEdge("e1", "forward", "node:b", "node:c"),
            VisualEdge("e_feedback", "feedback", "node:c", "node:a"),
        ),
    )

    first = compute_layered_layout(graph)
    second = compute_layered_layout(graph)

    assert first == second
    assert first.centers["node:a"].x < first.centers["node:c"].x
    assert first.centers["node:b"].x < first.centers["node:c"].x
    assert "e_feedback" not in first.forward_edge_ids


def test_layered_layout_handles_edgeless_graph_with_stable_grid():
    graph = VisualGraph(nodes=(_node("a"), _node("b")), terminals=(), edges=())

    layout = compute_layered_layout(graph)

    assert tuple(layout.centers) == ("node:a", "node:b")
    assert layout.centers["node:a"].x < layout.centers["node:b"].x
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_layout.py -q
```

- [ ] Expected before implementation: fail importing `compute_layered_layout`.

**Implementation Steps:**

- [ ] Add `leapp/leapp_graph/visualization/layout.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

from fast_sugiyama import from_edges

from .model import VisualGraph


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class LayoutResult:
    centers: dict[str, Point]
    forward_edge_ids: tuple[str, ...]
```

- [ ] Implement `compute_layered_layout(graph: VisualGraph) -> LayoutResult`.
- [ ] Use only edges with kind `forward`, `graph_input`, and `graph_output` as Sugiyama edges.
- [ ] Sort visual IDs, encode them to contiguous integers, pass integer edge tuples to `from_edges()`, and map returned integer positions back to visual IDs. This avoids Python string hash randomization inside the dependency's internal set construction.
- [ ] Sort edge tuples before encoding so the same visual graph produces the same integer edge list.
- [ ] Call:

```python
layouts = from_edges(
    edge_pairs,
    vertex_spacing=96,
    dummy_vertices=False,
    crossing_minimization="median",
    check_layout=True,
).dot_layout(spacing=96)
```

- [ ] Convert the raw top-down Sugiyama coordinates into LEAPP left-to-right coordinates:
  - `raw_max_y = max(raw_y values)`
  - `layout_x = (raw_max_y - raw_y) * 2.6`
  - `layout_y = raw_x * 1.8`
- [ ] Normalize all coordinates so the minimum `x` and `y` are `0.0`.
- [ ] Add visual IDs missing from Sugiyama output, such as isolated nodes and terminals, with a deterministic grid:
  - next available `x` column after the current maximum
  - row based on sorted missing ID order
  - horizontal step `260.0`
  - vertical step `140.0`
- [ ] For edge-less graphs, skip `from_edges()` and return the deterministic grid directly.

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_layout.py tests/unit_tests/test_graph_visualization_builder.py -q
```

- [ ] Expected after implementation: all listed tests pass.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/layout.py tests/unit_tests/test_graph_visualization_layout.py
git commit --no-gpg-sign -m "Add layered graph visualization layout"
```

---

## Task 4: Resolve Concrete Geometry And Edge Routes

**Files:**

- `leapp/leapp_graph/visualization/geometry.py`
- `tests/unit_tests/test_graph_visualization_geometry.py`

**Intent:** Convert visual model plus abstract layout centers into concrete rectangles, port anchors, terminal anchors, canvas bounds, and routed edge paths shared by SVG and PNG renderers.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_geometry.py`:

```python
from leapp.leapp_graph.visualization.geometry import resolve_geometry
from leapp.leapp_graph.visualization.layout import LayoutResult, Point
from leapp.leapp_graph.visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


def test_geometry_places_input_and_output_ports_on_node_edges():
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend="onnx-dynamo",
        inputs=(VisualPort("port:policy:input:obs", "node:policy", "input", "obs", ("1", "12"), "float32", "state"),),
        outputs=(VisualPort("port:policy:output:action", "node:policy", "output", "action", ("1", "4"), "float32", "command"),),
    )
    graph = VisualGraph(nodes=(node,), terminals=(), edges=())
    layout = LayoutResult(centers={"node:policy": Point(0.0, 0.0)}, forward_edge_ids=())

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    node_geometry = geometry.nodes["node:policy"]
    input_anchor = geometry.ports["port:policy:input:obs"].anchor
    output_anchor = geometry.ports["port:policy:output:action"].anchor

    assert input_anchor.x == node_geometry.rect.x
    assert output_anchor.x == node_geometry.rect.x + node_geometry.rect.width
    assert input_anchor.y < output_anchor.y
    assert geometry.width > node_geometry.rect.width
    assert geometry.height > node_geometry.rect.height


def test_geometry_routes_forward_and_feedback_edges_differently():
    a = VisualNode(
        id="node:a",
        title="a",
        backend=None,
        inputs=(VisualPort("port:a:input:state", "node:a", "input", "state", ("2",), "float32", None),),
        outputs=(VisualPort("port:a:output:y", "node:a", "output", "y", ("2",), "float32", None),),
    )
    b = VisualNode(
        id="node:b",
        title="b",
        backend=None,
        inputs=(VisualPort("port:b:input:y", "node:b", "input", "y", ("2",), "float32", None),),
        outputs=(VisualPort("port:b:output:state", "node:b", "output", "state", ("2",), "float32", None),),
    )
    graph = VisualGraph(
        nodes=(a, b),
        terminals=(),
        edges=(
            VisualEdge("edge:forward", "forward", "node:a", "node:b", "port:a:output:y", "port:b:input:y", "y"),
            VisualEdge("edge:feedback", "feedback", "node:b", "node:a", "port:b:output:state", "port:a:input:state", "state"),
        ),
    )
    layout = LayoutResult(centers={"node:a": Point(0.0, 0.0), "node:b": Point(300.0, 0.0)}, forward_edge_ids=("edge:forward",))

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    forward = geometry.edges["edge:forward"]
    feedback = geometry.edges["edge:feedback"]
    assert forward.kind == "forward"
    assert feedback.kind == "feedback"
    assert min(point.y for point in feedback.points) < geometry.content_bounds.y
    assert forward.points[0].x < forward.points[-1].x
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_geometry.py -q
```

- [ ] Expected before implementation: fail importing `resolve_geometry`.

**Implementation Steps:**

- [ ] Add `leapp/leapp_graph/visualization/geometry.py` with these dataclasses:

```python
from __future__ import annotations

from dataclasses import dataclass

from .layout import LayoutResult, Point
from .model import EdgeKind, VisualGraph


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class NodeGeometry:
    id: str
    title: str
    backend: str | None
    rect: Rect
    header_rect: Rect


@dataclass(frozen=True)
class TerminalGeometry:
    id: str
    title: str
    kind: str
    rect: Rect
    anchor: Point


@dataclass(frozen=True)
class PortGeometry:
    id: str
    node_id: str
    side: str
    name: str
    detail: str
    kind: str | None
    rect: Rect
    anchor: Point
    full_label: str


@dataclass(frozen=True)
class EdgeGeometry:
    id: str
    kind: EdgeKind
    label: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class GraphGeometry:
    graph_name: str
    width: float
    height: float
    content_bounds: Rect
    nodes: dict[str, NodeGeometry]
    terminals: dict[str, TerminalGeometry]
    ports: dict[str, PortGeometry]
    edges: dict[str, EdgeGeometry]
```

- [ ] Implement `resolve_geometry(graph: VisualGraph, layout: LayoutResult, graph_name: str) -> GraphGeometry`.
- [ ] Use deterministic text measurement rather than renderer-specific font metrics:
  - title width: `len(text) * 8.0`
  - port primary width: `len(text) * 7.0`
  - port detail width: `len(text) * 6.2`
  - minimum node width `220.0`
  - maximum node width `420.0`
  - terminal width between `100.0` and `220.0`
- [ ] Use fixed vertical tokens:
  - canvas margin `48.0`
  - header height `34.0`
  - port row height `46.0`
  - node internal padding `14.0`
  - gap between input and output port groups `10.0`
- [ ] Compute node height as:

```python
header_height + padding_top + len(inputs) * port_row_height + port_group_gap + len(outputs) * port_row_height + padding_bottom
```

- [ ] If a node has no inputs or no outputs, still reserve an empty group gap so output rows remain visually separated from input rows.
- [ ] Create port labels:
  - primary line: port name
  - detail line: `[shape] dtype`
  - kind line only in `full_label` and renderer text when kind exists
  - shape text uses `[]` for scalars and `[1, 12]` for tuple shapes
- [ ] Truncate visible name and detail strings in geometry with a helper:

```python
def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."
```

- [ ] Preserve the full label in `PortGeometry.full_label`.
- [ ] Position visual items from `LayoutResult.centers`, treating each point as the item center before global canvas normalization.
- [ ] After item placement, compute content bounds, then shift all rectangles and points by the canvas margin plus any feedback-lane offset.
- [ ] Forward edge route:
  - start at source output anchor or source terminal anchor
  - end at target input anchor or target terminal anchor
  - points are `(start, control1, control2, end)` with horizontal controls halfway between start and end
- [ ] Feedback edge route:
  - use source and target anchors
  - route through an outside lane above `content_bounds.y`
  - assign lane by stable sorted feedback edge order
  - points are `(start, up_from_source, lane_mid_left_or_right, down_to_target, end)`

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_geometry.py tests/unit_tests/test_graph_visualization_layout.py -q
```

- [ ] Expected after implementation: all listed tests pass.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/geometry.py tests/unit_tests/test_graph_visualization_geometry.py
git commit --no-gpg-sign -m "Resolve graph visualization geometry"
```

---

## Task 5: Render Static SVG

**Files:**

- `leapp/leapp_graph/visualization/svg_renderer.py`
- `tests/unit_tests/test_graph_visualization_svg.py`

**Intent:** Render a self-contained static SVG string from `GraphGeometry`.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_svg.py`:

```python
import xml.etree.ElementTree as ET

from leapp.leapp_graph.visualization.geometry import EdgeGeometry, GraphGeometry, NodeGeometry, PortGeometry, Rect, TerminalGeometry
from leapp.leapp_graph.visualization.layout import Point
from leapp.leapp_graph.visualization.svg_renderer import render_svg


def _geometry():
    return GraphGeometry(
        graph_name="demo",
        width=500.0,
        height=260.0,
        content_bounds=Rect(48.0, 70.0, 404.0, 142.0),
        nodes={
            "node:policy": NodeGeometry("node:policy", "policy", "onnx-dynamo", Rect(140.0, 90.0, 240.0, 120.0), Rect(140.0, 90.0, 240.0, 34.0)),
        },
        terminals={
            "terminal:input:policy:obs": TerminalGeometry("terminal:input:policy:obs", "obs", "graph_input", Rect(48.0, 126.0, 80.0, 28.0), Point(128.0, 140.0)),
        },
        ports={
            "port:policy:input:obs": PortGeometry("port:policy:input:obs", "node:policy", "input", "obs", "[1, 12] float32", "state", Rect(140.0, 130.0, 110.0, 46.0), Point(140.0, 153.0), "obs\n[1, 12] float32\nstate"),
        },
        edges={
            "edge:input": EdgeGeometry("edge:input", "graph_input", "obs", (Point(128.0, 140.0), Point(134.0, 140.0), Point(134.0, 153.0), Point(140.0, 153.0))),
        },
    )


def test_render_svg_is_parseable_static_and_contains_port_details():
    svg = render_svg(_geometry())

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 500 260"
    assert "<script" not in svg.lower()
    assert "policy" in svg
    assert "obs" in svg
    assert "[1, 12] float32" in svg
    assert "state" in svg
    assert "marker-end" in svg
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py -q
```

- [ ] Expected before implementation: fail importing `render_svg`.

**Implementation Steps:**

- [ ] Add `leapp/leapp_graph/visualization/svg_renderer.py` with these imports and public functions:

```python
from __future__ import annotations

from html import escape

from .geometry import EdgeGeometry, GraphGeometry, Point, Rect
```

- [ ] Implement `render_svg(geometry: GraphGeometry) -> str`.
- [ ] Implement `write_svg(path: str, geometry: GraphGeometry) -> None` as an UTF-8 text write of `render_svg(geometry)`.
- [ ] Use these visual tokens:

```python
COLORS = {
    "background": "#F7F8FA",
    "node": "#FFFFFF",
    "node_border": "#C9D2DE",
    "header": "#EDF2F7",
    "text": "#18212F",
    "secondary_text": "#5E6B7A",
    "forward_edge": "#566273",
    "feedback_edge": "#B42318",
    "graph_input": "#2F855A",
    "graph_output": "#C2410C",
    "torch": "#2563EB",
    "warp": "#76B900",
    "unknown_backend": "#667085",
}
```

- [ ] Use font family:

```text
Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif
```

- [ ] Emit:
  - `<svg xmlns="http://www.w3.org/2000/svg" width="{int(geometry.width)}" height="{int(geometry.height)}" viewBox="0 0 {int(geometry.width)} {int(geometry.height)}">`
  - `<defs>` with arrow markers for forward/input/output/feedback edges
  - one background `<rect>`
  - edges before nodes
  - terminals
  - nodes and ports
- [ ] Use `<title>` inside each port group with `PortGeometry.full_label`.
- [ ] Render paths:
  - four-point edge as cubic `M start C control1 control2 end`
  - five-point feedback edge as `M x0 y0 L x1 y1 L x2 y2 L x3 y3 L x4 y4`
- [ ] Escape all text with `html.escape`.
- [ ] Do not include scripts, external stylesheets, external images, or remote fonts.

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py tests/unit_tests/test_graph_visualization_geometry.py -q
```

- [ ] Expected after implementation: all listed tests pass.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/svg_renderer.py tests/unit_tests/test_graph_visualization_svg.py
git commit --no-gpg-sign -m "Render static graph visualization SVG"
```

---

## Task 6: Render PNG With Pillow From The Same Geometry

**Files:**

- `leapp/leapp_graph/visualization/png_renderer.py`
- `tests/unit_tests/test_graph_visualization_png.py`

**Intent:** Render a PNG from `GraphGeometry` without converting SVG through external tooling.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_png.py`:

```python
from pathlib import Path

from PIL import Image

from leapp.leapp_graph.visualization.geometry import EdgeGeometry, GraphGeometry, NodeGeometry, PortGeometry, Rect
from leapp.leapp_graph.visualization.layout import Point
from leapp.leapp_graph.visualization.png_renderer import write_png


def test_write_png_creates_nonempty_raster(tmp_path: Path):
    path = tmp_path / "graph.png"
    geometry = GraphGeometry(
        graph_name="demo",
        width=360.0,
        height=220.0,
        content_bounds=Rect(48.0, 48.0, 264.0, 124.0),
        nodes={
            "node:policy": NodeGeometry("node:policy", "policy", "jit-script", Rect(90.0, 60.0, 220.0, 120.0), Rect(90.0, 60.0, 220.0, 34.0)),
        },
        terminals={},
        ports={
            "port:policy:output:action": PortGeometry("port:policy:output:action", "node:policy", "output", "action", "[1, 4] float32", "command", Rect(200.0, 118.0, 110.0, 46.0), Point(310.0, 141.0), "action\n[1, 4] float32\ncommand"),
        },
        edges={
            "edge:self": EdgeGeometry("edge:self", "forward", "action", (Point(310.0, 141.0), Point(330.0, 141.0), Point(330.0, 170.0), Point(310.0, 170.0))),
        },
    )

    write_png(str(path), geometry)

    image = Image.open(path)
    assert image.size == (360, 220)
    assert image.getbbox() is not None
```

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_png.py -q
```

- [ ] Expected before implementation: fail importing `write_png`.

**Implementation Steps:**

- [ ] Add `leapp/leapp_graph/visualization/png_renderer.py` with these imports and public function:

```python
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .geometry import EdgeGeometry, GraphGeometry, Point, Rect
from .svg_renderer import COLORS
```

- [ ] Implement `write_png(path: str, geometry: GraphGeometry) -> None`.
- [ ] Render at scale factor `2`, then downsample with `Image.Resampling.LANCZOS`.
- [ ] Load fonts in this order:
  1. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
  2. `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`
  3. `ImageFont.load_default()`
- [ ] Draw background, edges, terminals, nodes, and ports in the same ordering as SVG.
- [ ] Draw cubic forward edges by sampling 24 points from the cubic Bezier:

```python
def sample_cubic(points: tuple[Point, Point, Point, Point], samples: int = 24) -> list[Point]
```

- [ ] Draw feedback edges as line segments through all route points.
- [ ] Draw arrowheads using vector math from the last two sampled points.
- [ ] Use `ImageDraw.rounded_rectangle` with each geometry `Rect` and `radius=6` for nodes and terminals.
- [ ] Keep card radius at or below 8px.

**Verify:**

- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_png.py tests/unit_tests/test_graph_visualization_svg.py -q
```

- [ ] Expected after implementation: all listed tests pass.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/png_renderer.py tests/unit_tests/test_graph_visualization_png.py
git commit --no-gpg-sign -m "Render graph visualization PNG with Pillow"
```

---

## Task 7: Wire `compile_graph(visualize=True)` To Emit SVG And PNG

**Files:**

- `leapp/leapp_graph/visualization/visualize.py`
- `leapp/leapp_graph/graph_gui.py`
- `leapp/leapp.py`
- `tests/functional_tests/test_annotate.py`
- `tests/unit_tests/test_graph_visualization_integration.py`

**Intent:** Replace the active Matplotlib/NetworkX visualizer with the new static renderer while keeping the existing `visualize_graph(...)` import path alive.

**Test First:**

- [ ] Add `tests/unit_tests/test_graph_visualization_integration.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from leapp.leapp_graph.visualization.visualize import visualize_graph


@dataclass
class FakeTensorDescription:
    name_str: str
    dtype: str
    shape: tuple[int, ...]
    tag: str | None = None
    semantics: dict | None = None

    def get_semantics(self):
        return self.semantics or {}


@dataclass
class FakeNode:
    name: str
    inputs: list[FakeTensorDescription]
    outputs: list[FakeTensorDescription]
    backend: str | None = "jit-script"
    node_index: int = 0


def test_visualize_graph_writes_svg_and_png(tmp_path: Path):
    node = FakeNode(
        name="policy",
        inputs=[FakeTensorDescription("obs", "float32", (1, 12))],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), semantics={"kind": "command"})],
        backend="jit-script",
    )

    visualize_graph(
        nodes={"policy": node},
        connections=[],
        feedback_connections=[],
        inputs=["policy/obs"],
        outputs=["policy/action"],
        save_path=str(tmp_path),
        graph_name="demo",
    )

    svg_path = tmp_path / "demo.svg"
    png_path = tmp_path / "demo.png"
    assert svg_path.exists()
    assert png_path.exists()
    assert "policy" in svg_path.read_text(encoding="utf-8")
    assert "command" in svg_path.read_text(encoding="utf-8")
    assert Image.open(png_path).size[0] > 0
```

- [ ] Update `tests/functional_tests/test_annotate.py::test_annotate_traced_tensors_diamond_with_feedback` after `leapp.compile_graph(visualize=True)` to assert both files exist:

```python
assert os.path.exists(os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.svg"))
assert os.path.exists(os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.png"))
```

  If this test class does not import `os`, add the import at the top of the file.
- [ ] Run:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_integration.py tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback -q
```

- [ ] Expected before implementation: integration test fails with the temporary `RuntimeError`; functional test still uses the old GUI path and may block or fail in headless mode.

**Implementation Steps:**

- [ ] Replace `leapp/leapp_graph/visualization/visualize.py` with:

```python
from __future__ import annotations

import os

from leapp.utils.logging import _get_logger

from .builder import build_visual_graph
from .geometry import resolve_geometry
from .layout import compute_layered_layout
from .png_renderer import write_png
from .svg_renderer import write_svg


def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    visual_graph = build_visual_graph(nodes, connections, feedback_connections, inputs, outputs)
    layout = compute_layered_layout(visual_graph)
    geometry = resolve_geometry(visual_graph, layout, graph_name)

    svg_path = os.path.join(save_path, f"{graph_name}.svg")
    png_path = os.path.join(save_path, f"{graph_name}.png")
    write_svg(svg_path, geometry)
    write_png(png_path, geometry)

    _get_logger().info(f"Graph visualization saved as: {svg_path}")
    _get_logger().info(f"Graph visualization saved as: {png_path}")
```

- [ ] Replace `leapp/leapp_graph/graph_gui.py` with a compatibility wrapper:

```python
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from .visualization import visualize_graph

__all__ = ["visualize_graph"]
```

- [ ] Update `leapp/leapp.py` docstrings and output artifact comments so visualization is described as `(<graph_name>.svg`, `<graph_name>.png`)`.
- [ ] Make sure no code path imports `matplotlib` or `networkx` after replacing `graph_gui.py`.

**Verify:**

- [ ] Run:

```bash
uv run pytest \
  tests/unit_tests/test_graph_visualization_integration.py \
  tests/unit_tests/test_graph_visualization_builder.py \
  tests/unit_tests/test_graph_visualization_layout.py \
  tests/unit_tests/test_graph_visualization_geometry.py \
  tests/unit_tests/test_graph_visualization_svg.py \
  tests/unit_tests/test_graph_visualization_png.py \
  tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback \
  -q
```

- [ ] Expected after implementation: all listed tests pass.
- [ ] Run:

```bash
rg -n "matplotlib|networkx|InteractiveGraphVisualizer|spring_layout|plt\\." leapp
```

- [ ] Expected after implementation: no matches.

**Commit:**

- [ ] Commit:

```bash
git add leapp/leapp_graph/visualization/visualize.py leapp/leapp_graph/graph_gui.py leapp/leapp.py tests/functional_tests/test_annotate.py tests/unit_tests/test_graph_visualization_integration.py
git commit --no-gpg-sign -m "Wire static graph visualization export"
```

---

## Task 8: Remove Old Visualization Dependencies And Update Docs

**Files:**

- `pyproject.toml`
- `thirdparty.txt`
- `README.md`
- `docs/source/api/index.rst`
- `docs/source/getting_started.rst`
- `docs/source/guides/debugging.rst`
- `docs/source/spelling_wordlist.txt`
- `tests/test_examples/test_getting_started.py`
- `tests/test_examples/test_feedback_example.py`
- `tests/test_examples/test_wbc_plain.py`
- `tests/test_examples/test_wbc_obj.py`

**Intent:** Finish the user-facing cleanup: remove Matplotlib/NetworkX dependency requirements, document SVG+PNG output, and update example artifact expectations.

**Test First:**

- [ ] Update example tests to expect both SVG and PNG:
  - `tests/test_examples/test_getting_started.py`: add `"sample_pipeline.svg"`.
  - `tests/test_examples/test_feedback_example.py`: add `"sample_feedback_graph.svg"`.
  - `tests/test_examples/test_wbc_plain.py`: add `"sample_wbc_graph.svg"`.
  - `tests/test_examples/test_wbc_obj.py`: add `"sample_wbc_obj.svg"`.
- [ ] Run:

```bash
uv run pytest \
  tests/test_examples/test_getting_started.py::TestGettingStarted::test_getting_started_execution \
  tests/test_examples/test_feedback_example.py::TestFeedbackExample::test_feedback_example_execution \
  -q
```

- [ ] Expected before docs/dependency cleanup but after Task 7: tests pass if SVGs are emitted; if example cleanup is the first edit in this task, they should pass after the expected-file lists are updated.

**Implementation Steps:**

- [ ] Remove from `pyproject.toml` dependencies:

```toml
    "matplotlib>=3.5.0",
    "networkx>=2.6",
```

- [ ] Keep:

```toml
    "fast-sugiyama>=0.5.3",
    "Pillow>=10.0.0",
```

- [ ] Update `thirdparty.txt`:
  - remove Matplotlib notice section
  - remove NetworkX notice section
  - add fast-sugiyama MIT notice with project URL `https://pypi.org/project/fast-sugiyama/`
  - add Pillow HPND-style license notice with project URL `https://python-pillow.org/`
- [ ] Update `README.md` sentence near the quickstart to say LEAPP writes a YAML pipeline spec plus SVG and PNG graph visualizations.
- [ ] Update `docs/source/api/index.rst`:
  - generated artifact step says `Generate {name}.svg and {name}.png if visualize=True`.
  - output file list includes both SVG and PNG.
  - tree example includes both `complete_pipeline.svg` and `complete_pipeline.png`.
- [ ] Update `docs/source/getting_started.rst`:
  - graph visualization section says SVG is the primary artifact and PNG is emitted alongside it.
  - keep existing image directive pointing at the PNG docs image unless the docs build already supports embedding the generated SVG.
- [ ] Update `docs/source/guides/debugging.rst` visualization sentence to mention SVG and PNG.
- [ ] Remove `matplotlib` from `docs/source/spelling_wordlist.txt` if there are no docs references left.

**Verify:**

- [ ] Run:

```bash
rg -n "matplotlib|networkx|NetworkX|Matplotlib|spring layout|manual|drag|\\.png`|\\.png\\)" README.md docs/source leapp pyproject.toml tests
```

- [ ] Expected after cleanup:
  - no Matplotlib/NetworkX/spring/manual-drag references in runtime docs or code
  - remaining `.png` references are either paired with `.svg` or refer to static docs images/icons
- [ ] Run:

```bash
uv run pytest \
  tests/unit_tests/test_graph_visualization_imports.py \
  tests/unit_tests/test_graph_visualization_builder.py \
  tests/unit_tests/test_graph_visualization_layout.py \
  tests/unit_tests/test_graph_visualization_geometry.py \
  tests/unit_tests/test_graph_visualization_svg.py \
  tests/unit_tests/test_graph_visualization_png.py \
  tests/unit_tests/test_graph_visualization_integration.py \
  tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback \
  tests/test_examples/test_getting_started.py::TestGettingStarted::test_getting_started_execution \
  tests/test_examples/test_feedback_example.py::TestFeedbackExample::test_feedback_example_execution \
  -q
```

- [ ] Expected after cleanup: all listed tests pass.
- [ ] Optional broad smoke run if time permits:

```bash
uv run pytest tests/unit_tests tests/functional_tests/test_annotate.py -q
```

**Commit:**

- [ ] Commit:

```bash
git add pyproject.toml thirdparty.txt README.md docs/source/api/index.rst docs/source/getting_started.rst docs/source/guides/debugging.rst docs/source/spelling_wordlist.txt tests/test_examples/test_getting_started.py tests/test_examples/test_feedback_example.py tests/test_examples/test_wbc_plain.py tests/test_examples/test_wbc_obj.py
git commit --no-gpg-sign -m "Document static SVG graph visualization"
```

---

## Final Verification

- [ ] Run focused visualization suite:

```bash
uv run pytest \
  tests/unit_tests/test_graph_visualization_imports.py \
  tests/unit_tests/test_graph_visualization_builder.py \
  tests/unit_tests/test_graph_visualization_layout.py \
  tests/unit_tests/test_graph_visualization_geometry.py \
  tests/unit_tests/test_graph_visualization_svg.py \
  tests/unit_tests/test_graph_visualization_png.py \
  tests/unit_tests/test_graph_visualization_integration.py \
  tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback \
  -q
```

- [ ] Expected: all tests pass.
- [ ] Run dependency/code cleanup check:

```bash
rg -n "matplotlib|networkx|InteractiveGraphVisualizer|spring_layout|plt\\." leapp pyproject.toml
```

- [ ] Expected: no matches.
- [ ] Run artifact smoke:

```bash
uv run python - <<'PY'
import os
import torch
import leapp
from leapp import annotate, TensorSemantics, InputKindEnum, OutputKindEnum

leapp.start("viz_smoke", save_path=".")
obs = annotate.input_tensors(
    "policy",
    [TensorSemantics("joint_pos", torch.randn(1, 12), kind=InputKindEnum.JOINT_POSITION)],
)
action = obs * 0.5
annotate.output_tensors(
    "policy",
    [TensorSemantics("torques", action, kind=OutputKindEnum.JOINT_TORQUES)],
    export_with="jit",
)
leapp.stop()
leapp.compile_graph(visualize=True, validate=True)

assert os.path.exists("viz_smoke/viz_smoke.svg")
assert os.path.exists("viz_smoke/viz_smoke.png")
print("viz_smoke svg+png emitted")
PY
```

- [ ] Expected: prints `viz_smoke svg+png emitted`.
- [ ] Inspect generated SVG and PNG manually enough to confirm:
  - graph input appears left of node
  - graph output appears right of node
  - node contains input/output port rows
  - port labels include name, shape, dtype, and kind
  - no GUI opens
- [ ] Remove the smoke output directory if it was created in the repository root:

```bash
rm -rf viz_smoke
```

- [ ] Final git status:

```bash
git status --short
```

- [ ] Expected: clean worktree after final commit, or only intentionally uncommitted generated artifacts if the user asked to inspect them.
