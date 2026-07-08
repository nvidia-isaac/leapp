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
import inspect
import os
import tempfile
from pathlib import Path

from leapp.leapp_graph.custom_operator_registry import warp_operator
from leapp.leapp_graph.custom_operator_registry.warp_operator.bundle import (
    WRP_FILENAME,
    pack_bundle,
)
from leapp.utils.tensor_description import warp_dtype_to_torch_name
from leapp.utils.logging import _get_logger

if TYPE_CHECKING:
    from leapp.leapp_graph.traced_node import TracedTensorNode
    from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend

try:
    import warp as wp
    from leapp.leapp_graph.datatypes.warp import TracedWpArray, WarpSegment
except ImportError:
    wp = None
    WarpOp = None
else:
    def _save_warp_bundle(node_name: str, segment: WarpSegment) -> bytes:
        """Persist a captured Warp APIC graph and pack it as a WRPB archive.

        Returns archive bytes for immediate FX embedding.
        """
        if segment.apic_graph is None:
            raise RuntimeError(
                f"[{node_name}] Warp segment '{segment.proxy_name or segment.node_name}' "
                "has no APIC graph to save."
            )

        bundle_dir = tempfile.mkdtemp(prefix=f"leapp_warp_{node_name}_")
        save_basename = os.path.join(bundle_dir, Path(WRP_FILENAME).stem)

        def _as_raw_warp_array(array):
            return getattr(array, "data", array)

        capture_inputs = {
            ref.name: _as_raw_warp_array(ref.array)
            for ref in segment.input_refs.values()
            if ref.proxy is not None and ref.array is not None
        }
        capture_outputs = {
            ref.name: _as_raw_warp_array(ref.array)
            for ref in segment.output_refs.values()
            if ref.array is not None
        }

        wp.capture_save(
            segment.apic_graph,
            save_basename,
            inputs=capture_inputs,
            outputs=capture_outputs,
        )
        wrp_path = f"{save_basename}.wrp"
        archive = pack_bundle(Path(wrp_path))

        _get_logger().debug(
            f"[{node_name}] Saved and packed Warp APIC bundle for "
            f"'{segment.proxy_name}' ({len(archive)} bytes)"
        )
        return archive


    def _update_output_ref_proxy(
        node_ref: "TracedTensorNode", ref, proxy
    ) -> None:
        if isinstance(ref.array, TracedWpArray):
            ref.array.rebind_tracing_proxy(ref.name, node_ref, proxy)


    def _insert_warp_marker(
        node_ref: "TracedTensorNode",
        segment: WarpSegment,
        *,
        wrp_archive: bytes,
    ) -> WarpSegment:
        """Build Warp runtime metadata and ask the node to emit FX proxies."""
        if segment.status == "closed":
            return segment
        if segment.status == "invalid":
            raise RuntimeError("Cannot close invalid Warp segment.")
        if segment.node_name != node_ref.name:
            raise ValueError(
                f"Warp segment belongs to node '{segment.node_name}', not "
                f"'{node_ref.name}'."
            )

        if segment.is_empty:
            segment.status = "closed"
            return segment

        input_refs = [
            ref for ref in segment.input_refs.values() if ref.proxy is not None
        ]
        input_proxies = [ref.proxy for ref in input_refs]
        output_refs = list(segment.output_refs.values())

        # ``output_mask`` starts all-True; the post-prune pass rewrites it to
        # mark only surviving outputs and zeroes unused shapes.
        output_shapes: list[list[int]] = []
        output_dtypes: list[str] = []
        for ref in output_refs:
            if ref.shape is None:
                raise RuntimeError(
                    f"Warp segment output '{ref.name}' has no observed shape; "
                    f"cannot emit {warp_operator.QUALIFIED_NAME}."
                )
            output_shapes.append([int(dim) for dim in ref.shape])
            output_dtypes.append(
                warp_dtype_to_torch_name(
                    getattr(ref.array, "dtype", None), text=ref.dtype
                )
            )

        output_mask = [True] * len(output_refs)
        runtime_metadata = warp_operator.build_runtime_metadata(
            segment=segment,
            input_refs=input_refs,
            output_refs=output_refs,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
            output_mask=output_mask,
        )
        encoded_metadata = warp_operator.encode_runtime_metadata(runtime_metadata)

        # this step inserts the FX proxy and updates all the outputs.
        warp_runner, output_proxies = node_ref.create_warp_proxy(
            encoded_metadata,
            input_proxies,
            wrp_archive,
            len(output_refs),
        )

        segment.marker_proxy = warp_runner
        segment.proxy_name = str(warp_runner.node.name)
        warp_runner.node.meta["leapp_warp_segment"] = segment

        for ref, proxy in zip(output_refs, output_proxies):
            ref.proxy = proxy
            ref.context = node_ref
            segment.output_proxies[ref.name] = proxy
            proxy.node.meta["leapp_warp_segment"] = segment
            proxy.node.meta["leapp_warp_output_ref"] = ref
            _update_output_ref_proxy(node_ref, ref, proxy)

        segment.status = "closed"
        return segment


    class WarpOp:
        def __init__(
            self,
            node_ref: "TracedTensorNode",
            *,
            warp_backend: "WarpPatchBackend",
            device: str = "cuda:0",
        ):
            self.node_ref = node_ref
            self.node_name = node_ref.name
            self.node_graph = node_ref.graph
            self._warp_backend = warp_backend

            # scoped capture variables
            self._scope = None
            self._capture = None
            self._segment = None
            self.device = device

        def __enter__(self):
            scoped_capture_params = inspect.signature(wp.ScopedCapture).parameters
            if "apic" not in scoped_capture_params or not hasattr(wp, "capture_save"):
                raise RuntimeError(
                    "annotate.warp_op requires Warp APIC support "
                    "(ScopedCapture(..., apic=True) and wp.capture_save). "
                    f"Installed Warp version: {getattr(wp, '__version__', 'unknown')}."
                )
            self._segment = WarpSegment(
                node_name=self.node_name,
                device=self.device,
            )
            self._warp_backend.push_segment(self._segment)
            self._scope = wp.ScopedCapture(
                device=self.device,
                force_module_load=True,
                apic=True,
            )
            self._capture = self._scope.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            scope_result = False
            if self._scope is not None:
                scope_result = self._scope.__exit__(exc_type, exc_value, traceback)
            try:
                if exc_type is None and self._segment is not None:
                    graph = self._capture.graph
                    self._segment.apic_graph = graph
                    self._segment.add_event({"kind": "scoped_capture"})
                    # Save before replay so formal inputs and closure buffers
                    # are snapshotted at the capture boundary, not after execution.
                    with self._warp_backend.paused():
                        wrp_archive = _save_warp_bundle(self.node_name, self._segment)
                        _insert_warp_marker(
                            self.node_ref,
                            self._segment,
                            wrp_archive=wrp_archive,
                        )
                        wp.capture_launch(graph)
                        wp.synchronize()
            finally:
                self._warp_backend.pop_segment(self._segment)

            return scope_result
