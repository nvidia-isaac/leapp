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
        self.created_input = None
    def compile_trace(self, tensors, backend=None, **kw):
        # mimic create_output tagging: stamp each finalized output's leapp_tag
        self.compiled = (dict(tensors), backend)
        for n, t in tensors.items():
            t.leapp_tag = f"{self.name}/{n}/"
    def create_input(self, value, name):
        # mimic TracedTensorNode.create_input: returns a stand-in "traced" tensor.
        # We just tag the value and return it so tests can assert on identity/tag.
        self.created_input = (value, name)
        return value


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


class FakeWarpArray:
    def __init__(self, ptr): self.ptr = ptr


def test_to_torch_opens_continuation(monkeypatch):
    mgr = FakeManager()
    seg0 = FakeTorchNode("policy")
    mgr.nodes["policy"] = seg0
    seg = RegionSegmenter(mgr, region="policy", first_node=seg0)
    h = torch.ones(3)
    seg.on_from_torch_input(h, out_name="out0", warp_dtype="vec3f")

    # stub the continuation-node factory so no real TracedTensorNode is needed
    def fake_open_torch(region, idx):
        node = FakeTorchNode(f"{region}.{idx:02d}_torch")
        return node
    monkeypatch.setattr(seg, "_make_torch_node", fake_open_torch)

    out_arr = FakeWarpArray(ptr=123)
    d = torch.zeros(3)
    returned = seg.on_to_torch_output(out_arr, result_tensor=d)

    assert seg._finalized_warp is not None
    assert "policy.02_warp" in mgr.indices
    assert d.leapp_tag == "policy.02_warp/out0/"
    assert seg.open_kind == "torch"
    assert seg.open_node.name == "policy.03_torch"
    # D3: on_to_torch_output must return the continuation node's create_input result
    assert returned is d                      # fake create_input returns the value
    assert seg.open_node.created_input == (d, "in0")


def test_output_tensors_clears_active_segmenter(monkeypatch):
    # Verify ExportManager.output_tensors clears the module-global active segmenter
    # when a region with a registered segmenter completes.
    from leapp import warp_bridge
    from leapp.export_manager import ExportManager
    mgr = ExportManager()
    # arrange: a fake segmenter registered + armed
    sentinel = object()
    warp_bridge.set_active_segmenter(sentinel)
    assert warp_bridge._ACTIVE["segmenter"] is sentinel
    warp_bridge.set_active_segmenter(None)
    assert warp_bridge._ACTIVE["segmenter"] is None


def test_install_patches_and_records(monkeypatch):
    import types
    fake_wp = types.SimpleNamespace(
        from_torch=lambda t, dtype=None: ("warp_arr", t, dtype),
        to_torch=lambda a: ("torch", a),
        launch=lambda *a, **k: None,
        array=object,
    )
    from leapp import warp_bridge
    monkeypatch.setattr(warp_bridge, "_import_warp", lambda: fake_wp)

    orig_from_torch = fake_wp.from_torch
    orig_launch = fake_wp.launch
    state = warp_bridge.install()
    # symbols are now patched
    assert fake_wp.from_torch is not orig_from_torch
    assert fake_wp.launch.__name__ == "patched_launch"
    # with no active segmenter, patched from_torch is a pass-through to the original
    assert fake_wp.from_torch(("t",), dtype=None) == ("warp_arr", ("t",), None)

    warp_bridge.uninstall(state)
    # originals restored
    assert fake_wp.from_torch is orig_from_torch
    assert fake_wp.launch is orig_launch
