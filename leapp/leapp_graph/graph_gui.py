#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from leapp.utils.logging import _get_logger
from leapp_visualization import render_graph

from .visualization_adapter import build_visual_graph


def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    graph = build_visual_graph(nodes, connections, feedback_connections, inputs, outputs)
    svg_path, png_path = render_graph(graph, save_path, graph_name)
    _get_logger().info(f"Graph visualization saved as: {svg_path}")
    _get_logger().info(f"Graph visualization saved as: {png_path}")


__all__ = ["visualize_graph"]
