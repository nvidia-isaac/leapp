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
