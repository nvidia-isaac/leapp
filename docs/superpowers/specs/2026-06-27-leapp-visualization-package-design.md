# leapp-visualization package split design

## Goal

Split the general graph layout and rendering code into a separate Python distribution named `leapp-visualization`, while keeping `leapp.compile_graph(visualize=True)` working by default for LEAPP users.

The split should let other projects install and use the renderer without depending on PyTorch, ONNX, or LEAPP internals.

## Chosen approach

Use a sibling package in this repository:

```text
packages/
  leapp-visualization/
    pyproject.toml
    src/
      leapp_visualization/
        __init__.py
        model.py
        layout.py
        geometry.py
        svg_renderer.py
        png_renderer.py
        render.py
```

The root `leapp` package will depend on `leapp-visualization` by default. This preserves the existing user experience: a normal `pip install leapp` includes visualization support.

## Package responsibilities

`leapp-visualization` owns the reusable pieces:

- generic graph dataclasses: nodes, ports, terminals, and edges
- layered layout using `fast-sugiyama`
- geometry resolution, including node sizing, port ordering, and terminal alignment
- SVG rendering
- PNG rendering through Pillow
- a high-level `render_graph(graph, output_path_base, title)` API that emits both `.svg` and `.png`

`leapp` owns only LEAPP-specific adaptation:

- converting LEAPP graph nodes, tensor descriptors, graph inputs, graph outputs, and feedback connections into `leapp_visualization.VisualGraph`
- logging the generated artifact paths through LEAPP logging
- invoking the adapter and renderer directly from `LeappGraph.visualize(...)`

## Public API

The reusable package should expose a small stable API:

```python
from leapp_visualization import (
    VisualEdge,
    VisualGraph,
    VisualNode,
    VisualPort,
    VisualTerminal,
    render_graph,
)
```

`render_graph(...)` should take a `VisualGraph`, output directory or base path, and title. It should return the generated SVG and PNG paths so callers can log or attach them.

The LEAPP package should not expose renderer internals as part of its own public API. Existing internal imports can remain temporarily with thin forwarding modules if needed for compatibility inside the current branch, but tests and new code should import generic renderer code from `leapp_visualization`.

## Dependencies

Move visualization-only dependencies to `packages/leapp-visualization/pyproject.toml`:

- `fast-sugiyama>=0.5.3`
- `Pillow>=10.0.0`

Keep the root `leapp` package dependency list focused on LEAPP runtime/export concerns, and add a default dependency on the sibling package:

```toml
dependencies = [
    "leapp-visualization==0.5.2",
    ...
]
```

For local development with uv, the lockfile should resolve the dependency from the sibling path. If the repository does not already use uv workspaces, add the smallest viable workspace configuration rather than restructuring the whole project.

## Data flow

`leapp.compile_graph(visualize=True)` will continue to call LEAPP's graph visualization adapter.

The adapter will:

1. receive LEAPP nodes, connections, feedback connections, graph inputs, and graph outputs
2. build a `leapp_visualization.VisualGraph`
3. call `leapp_visualization.render_graph(...)`
4. log the returned SVG and PNG artifact paths

Other projects can skip the adapter and construct `VisualGraph` directly.

## Error handling

Visualization errors should continue to fail graph compilation when `visualize=True`. Silent fallback would hide broken artifacts and make debugging harder.

The generic package should raise normal Python exceptions with actionable context. LEAPP should not wrap those exceptions unless it can add useful LEAPP-specific context.

## Testing

Move renderer unit tests to use `leapp_visualization` imports:

- layout tests
- geometry tests
- SVG renderer tests
- PNG renderer tests
- generic import/API tests

Keep LEAPP integration tests in the root package:

- adapter builds the expected `VisualGraph` from LEAPP-like nodes
- `compile_graph(visualize=True)` emits both SVG and PNG
- visualization failures still propagate

Verification should include:

```bash
uv run pytest tests/visualization_tests/ \
  tests/functional_tests/test_annotate.py::TestAnnotateTensor::test_annotate_traced_tensors_diamond_with_feedback \
  tests/test_examples/test_getting_started.py::TestGettingStarted::test_getting_started_execution \
  tests/test_examples/test_feedback_example.py::TestFeedbackExample::test_feedback_example_execution -q
```

Also build both distributions locally:

```bash
uv build
uv build packages/leapp-visualization
```

## Non-goals

- No runtime web UI.
- No interactive graph editing.
- No new layout dependency beyond the existing `fast-sugiyama`.
- No change to the LEAPP visualization artifact contract: `compile_graph(visualize=True)` still emits both SVG and PNG.
