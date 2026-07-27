from __future__ import annotations

import os

from .geometry import resolve_geometry
from .layout import compute_layered_layout
from .model import VisualGraph
from .png_renderer import write_png


def render_graph(graph: VisualGraph, save_path: str, graph_name: str) -> str:
    layout = compute_layered_layout(graph)
    geometry = resolve_geometry(graph, layout, graph_name)

    png_path = os.path.join(save_path, f"{graph_name}.png")
    write_png(png_path, geometry)
    return png_path
