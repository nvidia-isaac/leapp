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
