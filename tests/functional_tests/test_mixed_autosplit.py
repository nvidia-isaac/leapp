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


def test_fork_fails_loud():
    # A second wp.from_torch while a warp segment is open == a non-linear/forked region.
    # v1 must fail loudly (ADR-0002). Tested at the segmenter level with fakes.
    from leapp.warp_bridge import RegionSegmenter

    class FakeNode:
        def __init__(self, n):
            self.name = n
        def compile_trace(self, t, backend=None, **k):
            for nm, tt in t.items():
                tt.leapp_tag = f"{self.name}/{nm}/"

    class Mgr:
        def __init__(self):
            self.nodes = {}
        def _rename_node(self, o, n):
            self.nodes[n] = self.nodes.pop(o); self.nodes[n].name = n
        def _assign_index(self, node):
            pass
        def _default_torch_backend(self):
            return "onnx-torchscript"

    mgr = Mgr()
    f = FakeNode("r")
    mgr.nodes["r"] = f
    seg = RegionSegmenter(mgr, "r", f)
    h = torch.ones(3)
    seg.on_from_torch_input(h, "out0", "vec3f")
    with pytest.raises(RuntimeError, match="linear"):
        seg.on_from_torch_input(torch.ones(3), "out1", "vec3f")


def test_nontraced_from_torch_is_constant(tmp_path):
    # A plain torch.Tensor (not a TracedData/traced type) crossing wp.from_torch while a
    # segmenter IS active must NOT trigger a split — it is a baked constant, not a graph edge.
    # This exercises the `is_traced_type(t) and t.is_tracing` guard in patched_from_torch.
    pytest.importorskip("warp")
    import warp as wp
    wp.init()
    from leapp import warp_bridge

    split_called = []

    class SpySeg:
        open_kind = "torch"
        _bridge_counter = 0
        def on_from_torch_input(self, *a, **k):
            split_called.append(True)

    # Install the bridge patches and activate a spy segmenter
    state = warp_bridge.install()
    warp_bridge.set_active_segmenter(SpySeg())
    try:
        const = torch.ones(3)              # plain torch.Tensor, NOT a TracedData instance
        wp.from_torch(const.reshape(-1, 3), dtype=wp.vec3f)   # should be a pass-through
    finally:
        warp_bridge.uninstall(state)

    assert split_called == [], (
        "wp.from_torch of an untraced torch.Tensor must not create a node split"
    )
