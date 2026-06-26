# LEAPP graph visualization uses static layered SVG and PNG artifacts

LEAPP will replace the current interactive Matplotlib graph visualizer with a static
layered visualization that emits both SVG and PNG from `compile_graph(visualize=True)`.

## Decision

Use `fast-sugiyama` as the layout engine for directed graph layer/order placement, then use
LEAPP-owned rendering code to draw a static visualization. The SVG is the primary artifact.
The PNG is generated from the same resolved geometry as a companion raster artifact.

The visualization renders graph inputs, graph outputs, leapp nodes, and data-flow edges.
Each leapp node renders visualization ports for its tensor inputs and outputs. Port labels
show tensor name, shape, dtype, and semantic `kind` when present. Feedback edges are routed
and styled separately from forward data-flow edges.

The old Matplotlib window, NetworkX spring layout, and manual node dragging workflow are not
kept as a fallback.

## Why

LEAPP graph exports are pipeline graphs: direction matters. A Sugiyama-style layered layout
matches that structure better than a force-directed spring layout. rqt_graph achieves this
with Graphviz `dot`; Netron achieves a similar effect with a Dagre-style layout. LEAPP should
use the same family of layout ideas while keeping installation simple for Python users.

`fast-sugiyama` is uv/pip-installable, MIT licensed, and avoids a system Graphviz binary.
LEAPP can own rendering because the project needs LEAPP-specific concepts such as graph
inputs, graph outputs, feedback edges, node-kind/backend hints, and semantic tensor `kind`.

SVG provides a modern, docs-friendly primary artifact. PNG remains useful for reports,
README previews, and tools that do not embed SVG well.

## Consequences

The package can require Python 3.11+ and add `fast-sugiyama` plus Pillow. Matplotlib can be
removed if no other runtime path needs it.

Rendering becomes deterministic and headless. Users no longer need to manually position
nodes or close a GUI window to save the graph image.

The renderer must implement node sizing, port placement, text truncation, edge routing, SVG
generation, and PNG drawing. This is more local rendering code than delegating everything to
Graphviz, but avoids non-Python system dependencies and gives LEAPP control over its visual
language.
