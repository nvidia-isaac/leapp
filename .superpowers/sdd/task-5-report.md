## What I Implemented

- Added `leapp/leapp_graph/visualization/svg_renderer.py`.
- Implemented `render_svg(geometry: GraphGeometry) -> str` as a self-contained static SVG renderer.
- Implemented `write_svg(path: str, geometry: GraphGeometry) -> None` using UTF-8 text output.
- Exported `COLORS` for downstream renderer reuse.
- Rendered edges before terminals and nodes.
- Added SVG support for:
  - graph background rect
  - inline `<defs>` markers for forward, feedback, graph input, and graph output edges
  - cubic paths for 4-point edges
  - polyline-style `L` segments for 5-point feedback edges
  - terminals
  - nodes with headers and backend label
  - ports with `<title>` populated from `PortGeometry.full_label`
- Escaped all dynamic text with `html.escape`.
- Kept the SVG static and self-contained: no scripts, external stylesheets, external images, or remote fonts.

## Test Results

### RED

Command:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py -q
```

Observed failure before implementation:

```text
E   ModuleNotFoundError: No module named 'leapp.leapp_graph.visualization.svg_renderer'
```

I then expanded the SVG test coverage to also check:

- `<title>` contains `PortGeometry.full_label`
- edges are rendered before nodes
- `write_svg()` writes UTF-8 output
- `COLORS` is exported

### GREEN

Command:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py -q
```

Result:

```text
3 passed in 1.51s
```

### Verification

Command:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py tests/unit_tests/test_graph_visualization_geometry.py -q
```

Result:

```text
5 passed in 1.53s
```

## Files Changed

- `leapp/leapp_graph/visualization/svg_renderer.py`
- `tests/unit_tests/test_graph_visualization_svg.py`
- `.superpowers/sdd/task-5-report.md`

## Self-Review Findings

- The implementation is scoped to Task 5 only.
- The renderer is pure string generation with no changes to PNG rendering, visualize wiring, or GUI code.
- The output is static SVG only and uses inline attributes instead of external dependencies.
- Port tooltip content is emitted as `<title>` using the required `full_label`.
- Edge path generation matches the required 4-point cubic and 5-point feedback forms.

## Concerns

- None.

## Review Fixes

### Changes

- Updated `leapp/leapp_graph/visualization/svg_renderer.py` so output-side port name and detail text use the right-side inset and `text-anchor="end"`, matching the existing right-aligned output-side accent and kind label.
- Expanded `tests/unit_tests/test_graph_visualization_svg.py` coverage for:
  - output-port right alignment
  - 5-point feedback-edge SVG path serialization
  - graph-output terminal rendering

### Test Output

Command:

```bash
uv run pytest tests/unit_tests/test_graph_visualization_svg.py tests/unit_tests/test_graph_visualization_geometry.py -q
```

Result:

```text
.......                                                                  [100%]
7 passed in 1.51s
```
