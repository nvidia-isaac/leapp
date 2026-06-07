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
from leapp.warp_bridge import RegionSegmenter


class FakeTorchNode:
    def __init__(self, name):
        self.name = name
        self.compiled = None
    def compile_trace(self, tensors, backend=None, **kw):
        # mimic create_output tagging: stamp each finalized output's leapp_tag
        self.compiled = (dict(tensors), backend)
        for n, t in tensors.items():
            t.leapp_tag = f"{self.name}/{n}/"


class FakeManager:
    def __init__(self):
        self.nodes = {}
        self.renamed = []
        self.indices = []
    def _rename_node(self, old, new):
        self.nodes[new] = self.nodes.pop(old)
        self.nodes[new].name = new
        self.renamed.append((old, new))
    def _assign_index(self, node):
        self.indices.append(node.name)
    def _default_torch_backend(self):
        return "onnx-torchscript"


def test_from_torch_splits_segment():
    mgr = FakeManager()
    seg0 = FakeTorchNode("policy")
    mgr.nodes["policy"] = seg0
    seg = RegionSegmenter(mgr, region="policy", first_node=seg0)

    h = torch.ones(3)
    seg.on_from_torch_input(h, out_name="out0", warp_dtype="vec3f")

    assert ("policy", "policy.01_torch") in mgr.renamed
    assert seg0.compiled[0]["out0"] is h
    assert seg0.compiled[1] == "onnx-torchscript"
    assert h.leapp_tag == "policy.01_torch/out0/"
    assert seg.open_kind == "warp"
    assert seg.open_node.name == "policy.02_warp"


def test_second_from_torch_while_warp_open_fails_loud():
    mgr = FakeManager()
    seg0 = FakeTorchNode("policy")
    mgr.nodes["policy"] = seg0
    seg = RegionSegmenter(mgr, region="policy", first_node=seg0)
    seg.on_from_torch_input(torch.ones(3), out_name="out0", warp_dtype="vec3f")
    with pytest.raises(RuntimeError, match="linear"):
        seg.on_from_torch_input(torch.ones(3), out_name="out1", warp_dtype="vec3f")
