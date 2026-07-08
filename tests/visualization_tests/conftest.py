#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import sys


if sys.version_info < (3, 11):
    collect_ignore_glob = [
        "test_graph_visualization_builder.py",
        "test_graph_visualization_geometry.py",
        "test_graph_visualization_imports.py",
        "test_graph_visualization_integration.py",
        "test_graph_visualization_layout.py",
        "test_graph_visualization_png.py",
        "test_graph_visualization_svg.py",
    ]
