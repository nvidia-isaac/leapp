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
"""A graph-registered warp node: a contiguous warp segment captured as one native .wrp.

Built by the bridge segmenter (leapp.warp_bridge). Inputs/outputs are declared by the bridge
crossings (wp.from_torch / wp.to_torch). Each input carries the leapp_tag of the torch tensor
that produced it, so the cross-kind edge forms by tag-matching like a torch->torch edge.
"""
import os
import torch
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.backends.warp_capture_core import replay_into_apic_capture, resolve_device
from leapp.backends.warp_export_backend import save_warp_node
from leapp.backends import warp_dtypes as wd


class WarpRegionNode(LeappNode):
    def __init__(self, name, device=None, dry_run=False):
        super().__init__(name, dry_run=dry_run)
        self.device = str(device) if device else None
        self._records = []
        self._wp_inputs = {}     # port name -> wp.array
        self._wp_outputs = {}    # port name -> wp.array
        self._save_dir = None

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir

    def set_io(self, records, inputs, outputs):
        """Declare node I/O and tag edges.

        Args:
            records: List of wp.launch record dicts (as captured by the bridge or constructed
                directly). Used by replay_into_apic_capture() during compile_model().
            inputs: Mapping of port name -> (wp.array, source_torch_tensor).
                The source torch tensor must carry a ``leapp_tag`` attribute so that the
                cross-kind edge is formed by tag-matching in the LEAPP graph.
            outputs: Mapping of port name -> wp.array. A placeholder torch tensor is created
                and tagged ``"<node_name>/<port_name>/"`` so downstream nodes can connect.
        """
        import warp as wp
        self._records = records
        if self.device is None:
            self.device = resolve_device(wp, records, None)
        self._wp_inputs = {n: arr for n, (arr, _src) in inputs.items()}
        self._wp_outputs = dict(outputs)
        for n, (arr, src) in inputs.items():
            # add_input pulls src.leapp_tag into the input TensorDescription.tag
            self.add_input(n, n, src)
        for n, arr in outputs.items():
            wdstr = wd.warp_dtype_to_str(arr.dtype)
            view_shape = tuple(arr.shape) + wd.trailing_shape(wdstr)
            placeholder = torch.zeros(
                view_shape,
                dtype=getattr(torch, wd.scalar_base_str(wdstr)),
                device=self.device,
            )
            # tag_data sets placeholder.leapp_tag = "<self.name>/<n>/"
            self.tag_data(placeholder, n)
            self.add_output(n, n, placeholder)

    def compile_model(self):
        """Replay the recorded warp launches inside an APIC capture and wire the backend.

        After this call the node has a live WarpExportBackend with a compiled_model callable.
        The .wrp artifact is written to self._save_dir at this point; save_model() relocates
        it idempotently and records model_path / md5sum / sha256sum.
        """
        import warp as wp
        if self._save_dir is None:
            raise RuntimeError(
                f"WarpRegionNode '{self.name}': call set_save_dir() before compile_model()")
        graph = replay_into_apic_capture(wp, self._records, self.device)
        node_dict = save_warp_node(
            graph, self._save_dir, self.name,
            inputs=self._wp_inputs, outputs=self._wp_outputs,
        )
        # save_warp_node returns model_path as a basename; we need the full absolute path
        # so that WarpExportBackend.save() can locate and relocate it correctly.
        wrp_path = os.path.join(self._save_dir, self.name + ".wrp")
        params = dict(node_dict["parameters"])
        params["model_path"] = wrp_path
        self.setup_backend("warp", params)
        self.export_backend.load(params["model_path"], params["sha256sum"])
        self._model_captured = True
