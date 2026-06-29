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

from typing import TYPE_CHECKING
from leapp.utils.logging import _get_logger

if TYPE_CHECKING:
    from leapp.leapp_graph.traced_node import TracedTensorNode

from leapp.leapp_graph.datatypes import WarpLeappCallDetector, wp
from leapp.leapp_graph.datatypes.warp import WarpSegment

if wp is None or WarpLeappCallDetector is None:
    WarpOp = None
else:
    class WarpOp:
        def __init__(self, node_ref: "TracedTensorNode", device: str = "cuda:0"):
            self.node_ref = node_ref
            self.node_name = node_ref.name
            self.node_graph = node_ref.graph

            # scoped capture variables
            self._scope = None
            self._capture = None
            self._segment = None
            self._detector = None
            self.device = device

        def __enter__(self):
            self._segment = WarpSegment(
                node_name=self.node_name,
                device=self.device,
            )
            self._detector = WarpLeappCallDetector.instance()
            self._detector.push_segment(self._segment)
            self._scope = wp.ScopedCapture(
                device=self.device,
                force_module_load=True,
                apic=True,
            )
            self._capture = self._scope.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            scope_result = False
            scope_result = self._scope.__exit__(exc_type, exc_value, traceback)
            try:
                if exc_type is None:
                    graph = self._capture.graph
                    # ScopedCapture only records the kernels; replay the graph here
                    # so the real buffers advance at trace time. Suppress detection
                    # during replay so the patched ``wp.capture_launch`` /
                    # ``wp.synchronize`` calls do not append spurious events to the
                    # still-active segment.
                    with self._detector.paused():
                        # still need to execute the graph to get outputs
                        wp.capture_launch(graph)
                        wp.synchronize()
                    if self._segment is not None:
                        self._segment.apic_graph = graph
                        self._segment.add_event({"kind": "scoped_capture"})
                        self.node_ref.insert_warp_marker(self._segment)
            finally:
                if self._detector is not None:
                    self._detector.pop_segment(self._segment)

            return scope_result
