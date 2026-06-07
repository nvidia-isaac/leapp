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
"""Bridge interception + the per-region linear segmenter.

v1 bridge = wp.from_torch / wp.to_torch ONLY. The segmenter turns each crossing into a node
boundary. Linear chains only (ADR-0002): a forked tensor across a bridge fails loudly.
"""
from leapp.leapp_graph.datatypes import is_traced_type
from leapp.leapp_graph.warp_region_node import WarpRegionNode


def _seg_name(region, idx, kind):
    return f"{region}.{idx:02d}_{kind}"


class RegionSegmenter:
    """One per active marked region. Tracks the currently-open segment node and splits at
    bridges. The first segment starts as the bare `region` node and is renamed to
    `<region>.01_torch` the first time a split actually happens."""

    def __init__(self, manager, region, first_node):
        self.mgr = manager
        self.region = region
        self.open_node = first_node
        self.open_kind = "torch"
        self._seg_idx = 1
        self._split_happened = False
        self._bridge_counter = 0  # incremented on each traced wp.from_torch crossing
        self._pending_warp_inputs = {}
        self._finalized_warp = None

    def _ensure_first_renamed(self):
        if not self._split_happened:
            new = _seg_name(self.region, 1, "torch")
            self.mgr._rename_node(self.open_node.name, new)
            self._split_happened = True

    def on_from_torch_input(self, torch_tensor, out_name, warp_dtype):
        if self.open_kind != "torch":
            raise RuntimeError(
                f"warp_bridge: wp.from_torch reached while a warp segment is open in region "
                f"'{self.region}'. v1 supports only linear torch<->warp chains (ADR-0002); "
                "express this as explicit manual nodes.")
        self._ensure_first_renamed()
        torch_node = self.open_node
        torch_node.compile_trace({out_name: torch_tensor},
                                 backend=self.mgr._default_torch_backend())
        self.mgr._assign_index(torch_node)
        self._seg_idx += 1
        warp_node = WarpRegionNode(_seg_name(self.region, self._seg_idx, "warp"))
        self.mgr.nodes[warp_node.name] = warp_node
        warp_node._max_cached_io = getattr(self.mgr, "_max_cached_io", 0)
        self.open_node = warp_node
        self.open_kind = "warp"
        return torch_tensor

    def _make_torch_node(self, region, idx):
        # real impl: create a TracedTensorNode via the manager so input_tensors/output_tensors
        # route correctly. Overridden in unit tests.
        from leapp.leapp_graph.traced_node import TracedTensorNode
        node = TracedTensorNode(_seg_name(region, idx, "torch"))
        node._max_cached_io = getattr(self.mgr, "_max_cached_io", 0)
        self.mgr.nodes[node.name] = node
        return node

    def on_to_torch_output(self, warp_array, result_tensor):
        if self.open_kind != "warp":
            raise RuntimeError(
                f"warp_bridge: wp.to_torch reached with no open warp segment in region "
                f"'{self.region}'.")
        warp_node = self.open_node
        warp_node._pending_outputs = getattr(warp_node, "_pending_outputs", {})
        out_name = f"out{len(warp_node._pending_outputs)}"
        # Store (warp_array, torch_tensor) so _finalize_warp_node has the real output values
        # for validation without needing to call wp.to_torch() through the patched bridge again.
        warp_node._pending_outputs[out_name] = (warp_array, result_tensor)
        result_tensor.leapp_tag = f"{warp_node.name}/{out_name}/"
        self._finalize_warp_node(warp_node)
        self.mgr._assign_index(warp_node)
        self._finalized_warp = warp_node
        self._seg_idx += 1
        cont = self._make_torch_node(self.region, self._seg_idx)
        self.open_node = cont
        self.open_kind = "torch"
        return cont.create_input(result_tensor, "in0")

    def record_launch(self, args, kwargs):
        self.open_node._records.append({"args": args, "kwargs": kwargs})

    def bind_warp_input(self, name, warp_array, src_tensor):
        self.open_node._wp_inputs[name] = warp_array
        self._pending_warp_inputs[name] = src_tensor

    def _finalize_warp_node(self, warp_node):
        # Assemble bridged I/O onto the WarpRegionNode. `_wp_inputs` (the warp arrays) was filled
        # by bind_warp_input; `_pending_warp_inputs` holds the producing torch tensors (their
        # leapp_tag forms the incoming edge); `_pending_outputs` was filled in on_to_torch_output
        # as (warp_array, torch_tensor) tuples.
        inputs = {n: (warp_node._wp_inputs[n], self._pending_warp_inputs[n])
                  for n in warp_node._wp_inputs}
        # Unpack (warp_array, torch_tensor) pairs; set_io receives {name: warp_array}
        # and uses the torch_tensor for the output placeholder (validation reference).
        outputs = dict(getattr(warp_node, "_pending_outputs", {}))
        warp_node.set_save_dir(self.mgr.get_save_path())
        warp_node.set_io(warp_node._records, inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Module-level bridge patches installed by leapp.start()
# ---------------------------------------------------------------------------

_ACTIVE = {"segmenter": None}


def _import_warp():
    import warp as wp
    return wp


def set_active_segmenter(segmenter):
    _ACTIVE["segmenter"] = segmenter


def install():
    """Patch wp.from_torch/to_torch/launch so the active RegionSegmenter sees every bridge
    crossing and warp launch. Returns state to pass to uninstall(). Patches are pass-throughs
    when no segmenter is active or the crossing tensor is untraced (an untraced tensor across
    wp.from_torch is a baked constant, not a graph edge)."""
    wp = _import_warp()
    orig = {"from_torch": wp.from_torch, "to_torch": wp.to_torch, "launch": wp.launch}

    def patched_from_torch(t, dtype=None, *a, **k):
        arr = orig["from_torch"](t, dtype=dtype, **k)
        seg = _ACTIVE["segmenter"]
        # A live TracedTensor (actively being traced) crossing into warp creates an edge/split.
        # A raw torch.Tensor (untraced) is a baked constant input to the .wrp (no split).
        if seg is not None and is_traced_type(t) and getattr(t, 'is_tracing', False):
            from leapp.backends import warp_dtypes as wd
            if dtype is not None:
                wdstr = wd.warp_dtype_to_str(dtype)
            else:
                wdstr = wd.torch_dtype_to_warp_str(t.dtype)
            name = f"out{seg._bridge_counter}"
            seg._bridge_counter += 1
            seg.on_from_torch_input(t, out_name=name, warp_dtype=wdstr)
            seg.bind_warp_input(name, arr, t)
        return arr

    def patched_to_torch(a, *args, **k):
        out = orig["to_torch"](a, *args, **k)
        seg = _ACTIVE["segmenter"]
        if seg is not None and seg.open_kind == "warp":
            return seg.on_to_torch_output(a, result_tensor=out)
        return out

    def patched_launch(*a, **k):
        seg = _ACTIVE["segmenter"]
        if seg is not None and seg.open_kind == "warp":
            seg.record_launch(a, k)
        return orig["launch"](*a, **k)
    patched_launch.__name__ = "patched_launch"

    wp.from_torch, wp.to_torch, wp.launch = patched_from_torch, patched_to_torch, patched_launch
    return {"wp": wp, "orig": orig}


def uninstall(state):
    wp, orig = state["wp"], state["orig"]
    wp.from_torch, wp.to_torch, wp.launch = orig["from_torch"], orig["to_torch"], orig["launch"]
    _ACTIVE["segmenter"] = None
