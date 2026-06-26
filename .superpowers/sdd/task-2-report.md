## Task 2 Report: Build The LEAPP Visual Model

### What I implemented

Implemented `leapp/leapp_graph/visualization/builder.py` with `build_visual_graph(...)` as the public entrypoint.

The builder now:

- sorts nodes by `node.node_index` and then by name
- builds `VisualNode` objects first
- preserves port metadata, including shape, dtype, and semantic `kind`
- converts semantic enum kinds through `.value` when present
- resolves backend from `node.get_backend()` when available, otherwise `node.backend`
- builds one `VisualEdge` per source-target pair
- emits edges in the required order:
  1. forward
  2. feedback
  3. graph input
  4. graph output
- creates stable IDs using the brief’s `visual_id(...)` patterns
- creates `VisualTerminal` objects for graph inputs and outputs

I also added `tests/unit_tests/test_graph_visualization_builder.py` with the two requested cases:

- port preservation, semantics preservation, and external I/O handling
- feedback edge separation from forward flow

### Test results

RED step:

- `uv run pytest tests/unit_tests/test_graph_visualization_builder.py -q`
- Result before implementation: import failure with `ModuleNotFoundError: No module named 'leapp.leapp_graph.visualization.builder'`

GREEN step:

- `uv run pytest tests/unit_tests/test_graph_visualization_builder.py -q`
- Result after implementation: `2 passed`

Verification step:

- `uv run pytest tests/unit_tests/test_graph_visualization_builder.py tests/unit_tests/test_graph_visualization_imports.py -q`
- Result: `4 passed`

### Files changed

- `leapp/leapp_graph/visualization/builder.py`
- `tests/unit_tests/test_graph_visualization_builder.py`

### Self-review findings

- The implementation stays within the requested surface area and does not touch `graph_gui.py` or geometry/rendering code.
- Node ordering, edge grouping, and stable ID generation match the brief.
- The helper layer accepts both the fake test objects and the real LEAPP node/description shape.

### Concerns

- None identified from the requested verification scope.
