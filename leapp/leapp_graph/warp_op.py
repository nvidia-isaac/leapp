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
import os
import tempfile
from pathlib import Path

from leapp.utils.caller_identity import (
    caller_identity_has_same_anchor,
    format_caller_identity,
    get_caller_stack_identity,
)
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
        if segment.runner_name is None:
            raise RuntimeError(
                f"Warp segment for node '{segment.node_name}' has no runner name."
            )

        # this step inserts the FX proxy and updates all the outputs.
        warp_runner, output_proxies = node_ref.create_warp_proxy(
            encoded_metadata,
            input_proxies,
            wrp_archive,
            len(output_refs),
            segment.runner_name,
        )

        segment.marker_proxy = warp_runner
        segment.proxy_name = str(warp_runner.node.name)
        warp_runner.node.meta["leapp_warp_segment"] = segment

        for ref, proxy in zip(output_refs, output_proxies):
            ref.proxy = proxy
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
            self._exit_function = None
            self.device = device

        def __enter__(self):
            if not self.node_ref.is_warp_capture_active:
                self._segment = self._warp_backend.begin_discovery_segment(
                    node_name=self.node_name,
                    call_stack=get_caller_stack_identity(),
                )
                self._exit_function = self._exit_discovery
            else:
                stored_segment = self.node_ref.acquire_warp_segment()
                if stored_segment is None:
                    # TODO: switch to fatal instead of error
                    raise RuntimeError(
                        f"[{self.node_name}] Warp capture encountered more "
                        "regions than discovery."
                    )
                call_stack = get_caller_stack_identity()
                if not caller_identity_has_same_anchor(
                    stored_segment.call_stack,
                    call_stack,
                ):
                    # TODO: switch to fatal instead of error
                    message = (
                        f"[{self.node_name}] Warp segment was re-entered from "
                        "a different annotation origin.\n"
                        "Discovery origin:\n"
                        f"{format_caller_identity(stored_segment.call_stack)}\n"
                        f"Capture origin:\n{format_caller_identity(call_stack)}"
                    )
                    _get_logger().error(f"Fatal: {message}")
                    raise RuntimeError(message)

                self._segment = self._warp_backend.begin_capture_segment(
                    segment=stored_segment
                )
                self._exit_function = self._exit_capture
                with self._warp_backend.paused():
                    self._scope = wp.ScopedCapture(
                        device=self.device,
                        force_module_load=True,
                        apic=True,
                    )
                    self._capture = self._scope.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self._segment is None or self._exit_function is None:
                return False
            call_stack = get_caller_stack_identity()
            if not self._warp_backend.call_stack_matches_segment(
                self._segment,
                call_stack,
            ):
                return False
            return self._exit_function(
                exc_type,
                exc_value,
                traceback,
                call_stack,
            )

        def _exit_discovery(
            self,
            exc_type,
            exc_value,
            traceback,
            call_stack,
        ):
            try:
                if exc_type is None and self._segment is not None:
                    closed = self._warp_backend.end_discovery_segment(
                        self._segment,
                        call_stack,
                    )
                    if closed is None:
                        return False
                    self.node_ref.add_warp_segment(self._segment)
                    self._segment.status = "open"
                    _insert_warp_marker(
                        self.node_ref,
                        self._segment,
                        wrp_archive=b"\0",
                    )
            finally:
                if self._segment is not None:
                    active = self._warp_backend.active_segment
                    if active is self._segment:
                        self._warp_backend.deactivate_segment(self._segment)
            return False

        def _exit_capture(
            self,
            exc_type,
            exc_value,
            traceback,
            call_stack,
        ):
            scope_result = False
            segment_popped = False
            try:
                if self._scope is not None:
                    with self._warp_backend.paused():
                        scope_result = self._scope.__exit__(
                            exc_type, exc_value, traceback
                        )
                if exc_type is None and self._segment is not None:
                    graph = self._capture.graph
                    self._segment.apic_graph = graph
                    self._segment.add_event({"kind": "scoped_capture"})
                    if not self._warp_backend.end_capture_segment(
                        self._segment,
                        call_stack,
                    ):
                        return False
                    # The Warp capture is closed at this point. Deactivate the
                    # LEAPP segment before internal save/replay work so CUPTI
                    # warnings only cover user CUDA work inside the segment.
                    segment_popped = True
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
                    self.node_ref.complete_warp_segment(self._segment)
            finally:
                if self._segment is not None and not segment_popped:
                    self._warp_backend.deactivate_segment(self._segment)

            return scope_result
