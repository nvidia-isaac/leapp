#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

def test_visualization_package_exports_public_entrypoint():
    from leapp_visualization import render_graph

    assert callable(render_graph)


def test_visual_model_types_are_importable():
    from leapp_visualization import (
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
