#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

def test_visualization_package_exports_public_entrypoint():
    from leapp.leapp_graph.visualization import visualize_graph

    assert callable(visualize_graph)


def test_visual_model_types_are_importable():
    from leapp.leapp_graph.visualization.model import (
        VisualEdge,
        VisualGraph,
        VisualNode,
        VisualPort,
        VisualTerminal,
    )

    assert VisualGraph.__name__ == "VisualGraph"
    assert VisualNode.__name__ == "VisualNode"
    assert VisualPort.__name__ == "VisualPort"
    assert VisualTerminal.__name__ == "VisualTerminal"
    assert VisualEdge.__name__ == "VisualEdge"
