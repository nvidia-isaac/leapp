# leapp-visualization

`leapp-visualization` provides static layered graph layout and rendering for pipeline-style directed graphs.

It emits SVG as the primary artifact and PNG as a companion artifact. The package is intentionally independent from LEAPP runtime/export internals, PyTorch, and ONNX.

```python
from leapp_visualization import VisualGraph, render_graph

svg_path, png_path = render_graph(graph, ".", "pipeline")
```
