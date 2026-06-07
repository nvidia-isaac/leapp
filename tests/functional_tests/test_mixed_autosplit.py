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
import pytest
torch = pytest.importorskip("torch")
import leapp
from leapp import annotate


def test_start_installs_bridge_when_warp_available(tmp_path):
    pytest.importorskip("warp")
    leapp.start("g", save_path=str(tmp_path))
    import warp as wp
    assert wp.launch.__name__ == "patched_launch"
    leapp.stop()
    assert wp.launch.__name__ != "patched_launch"


def test_torch_only_region_keeps_bare_name(tmp_path):
    from leapp.export_manager import ExportManager
    leapp.start("g", save_path=str(tmp_path))
    x = torch.ones(4)
    xt = annotate.input_tensors("policy", {"x": x})
    y = xt * 2.0
    annotate.output_tensors("policy", {"y": y}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)
    nodes = ExportManager().get_nodes()
    assert "policy" in nodes
    assert not any(n.startswith("policy.") for n in nodes)
    # reset singleton so the next test sees a clean state (no leapp.start() will be called before it)
    ExportManager().reset_nodes()
