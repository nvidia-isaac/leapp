# leapp-visualization

`leapp-visualization` provides static layered graph layout and rendering for pipeline-style directed graphs.

It emits a PNG artifact. The package is intentionally independent from LEAPP runtime/export internals, PyTorch, and ONNX.

```python
from leapp_visualization import VisualGraph, render_graph

png_path = render_graph(graph, ".", "pipeline")
```
