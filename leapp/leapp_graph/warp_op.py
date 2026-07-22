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

import os
import tempfile
from pathlib import Path
from typing import Any

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
            _get_logger().fatal(
                f"[{node_name}] Warp segment '{segment.proxy_name or segment.node_name}' "
                "has no APIC graph to save.",
                error_type=RuntimeError,
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
        node_ref: Any, ref, proxy
    ) -> None:
        if isinstance(ref.array, TracedWpArray):
            ref.array.rebind_tracing_proxy(ref.name, node_ref, proxy)


    def _insert_warp_marker(
        node_ref: Any,
        segment: WarpSegment,
        *,
        wrp_archive: bytes,
    ) -> WarpSegment:
        """Build Warp runtime metadata and ask the node to emit FX proxies."""
        if segment.status == "closed":
            return segment
        if segment.status == "invalid":
            _get_logger().fatal(
                "Cannot close invalid Warp segment.",
                error_type=RuntimeError,
            )
        if segment.node_name != node_ref.name:
            _get_logger().fatal(
                f"Warp segment belongs to node '{segment.node_name}', not "
                f"'{node_ref.name}'.",
                error_type=ValueError,
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
                _get_logger().fatal(
                    f"Warp segment output '{ref.name}' has no observed shape; "
                    f"cannot emit {warp_operator.QUALIFIED_NAME}.",
                    error_type=RuntimeError,
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
            _get_logger().fatal(
                f"Warp segment for node '{segment.node_name}' has no runner name.",
                error_type=RuntimeError,
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
            node_ref: Any,
            *,
            session: Any,
            device: str = "cuda:0",
        ):
            self.node_ref = node_ref
            self.node_name = node_ref.name
            self.node_graph = node_ref.graph
            self._session = session

            # scoped capture variables
            self._scope = None
            self._capture = None
            self._segment = None
            self._exit_function = None
            self._mode = None
            self.device = device

        @property
        def segment(self):
            return self._segment

        def __enter__(self):
            call_stack = get_caller_stack_identity()
            self.begin(call_stack, owner_token=self)
            return self

        def begin(self, call_stack, *, owner_token=None):
            if not self.node_ref.is_warp_capture_active:
                self._begin_discovery(call_stack, owner_token=owner_token)
            else:
                self._begin_capture(call_stack, owner_token=owner_token)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self._segment is None or self._exit_function is None:
                return False
            call_stack = get_caller_stack_identity()
            if not caller_identity_has_same_anchor(
                self._segment.open_call_stack,
                call_stack,
            ):
                return False
            self._session.close_warp_segment(
                requester=self,
                close_call_stack=call_stack,
                exc_type=exc_type,
                exc_value=exc_value,
                traceback=traceback,
            )
            return False

        def _begin_discovery(self, call_stack, *, owner_token=None) -> None:
            self._mode = "discovery"
            self._segment = WarpSegment(
                node_name=self.node_name,
                open_call_stack=call_stack,
            )
            self._session.register_warp_op(
                self,
                self._segment,
                owner_token=owner_token,
                close_call_stack=call_stack,
            )
            self._exit_function = self._exit_discovery

        def _begin_capture(self, call_stack, *, owner_token=None) -> None:
            self._mode = "capture"
            if self._session.active_warp_op is not None:
                closed = self._session.close_warp_segment(
                    close_call_stack=call_stack,
                )
                if not closed:
                    _get_logger().fatal(
                        "Cannot begin WarpOp because the active WarpOp is "
                        "protected by an owner token. An operation is likely attempting to open a new"
                        "segment inside a warp_op.",
                        error_type=RuntimeError,
                    )
            stored_segment = self.node_ref.acquire_warp_segment()
            if stored_segment is None:
                _get_logger().fatal(
                    f"[{self.node_name}] Warp capture encountered more "
                    "regions than discovery.",
                    error_type=RuntimeError,
                )
            if not caller_identity_has_same_anchor(
                stored_segment.open_call_stack,
                call_stack,
            ):
                message = (
                    f"[{self.node_name}] Warp segment was re-entered from "
                    "a different annotation origin.\n"
                    "Discovery origin:\n"
                    f"{format_caller_identity(stored_segment.open_call_stack)}\n"
                    f"Capture origin:\n{format_caller_identity(call_stack)}"
                )
                _get_logger().fatal(message, error_type=RuntimeError)

            self._segment = self._prepare_capture_segment(stored_segment)
            self._session.register_warp_op(
                self,
                self._segment,
                owner_token=owner_token,
                close_call_stack=call_stack,
            )
            self._exit_function = self._exit_capture
            with self._session.pause():
                self._scope = wp.ScopedCapture(
                    device=self.device,
                    force_module_load=True,
                    apic=True,
                )
                self._capture = self._scope.__enter__()

        def terminate(
            self,
            exc_type=None,
            exc_value=None,
            traceback=None,
            *,
            close_call_stack=None,
        ):
            if self._segment is None or self._exit_function is None:
                return None
            if self._mode == "discovery":
                return self._terminate_discovery(close_call_stack=close_call_stack)
            if self._mode == "capture":
                return self._exit_capture(
                    exc_type,
                    exc_value,
                    traceback,
                    close_call_stack,
                )
            return None

        def _terminate_discovery(self, *, close_call_stack):
            closed = self._finalize_discovery(close_call_stack=close_call_stack)
            self.node_ref.add_warp_segment(closed)
            self._segment.status = "open"
            _insert_warp_marker(
                self.node_ref,
                self._segment,
                wrp_archive=b"\0",
            )
            return closed

        def _exit_discovery(
            self,
            exc_type,
            exc_value,
            traceback,
            call_stack,
        ):
            if exc_type is None and self._segment is not None:
                self._terminate_discovery(close_call_stack=call_stack)
            return False

        def _finalize_discovery(self, *, close_call_stack) -> WarpSegment:
            if self._segment is None:
                _get_logger().fatal(
                    "Discovery WarpOp has no active segment.",
                    error_type=RuntimeError,
                )

            warp_call_events = tuple(
                event for event in self._segment.events
                if event.get("kind") == "warp_call"
            )
            self._segment.call_qualnames = tuple(
                str(event["qualname"]) for event in warp_call_events
            )
            self._segment.close_call_stack = close_call_stack
            self._segment.status = "closed"
            return self._segment

        def _prepare_capture_segment(self, segment: WarpSegment) -> WarpSegment:
            segment.status = "open"
            segment.events.clear()
            segment.input_refs.clear()
            segment.output_refs.clear()
            segment.marker_proxy = None
            segment.proxy_name = None
            segment.apic_graph = None
            return segment

        def _finalize_capture(self) -> bool:
            if self._segment is None:
                _get_logger().fatal(
                    "Capture WarpOp has no active segment.",
                    error_type=RuntimeError,
                )
            expected_qualnames = self._segment.call_qualnames
            actual_qualnames = tuple(
                str(event["qualname"])
                for event in self._segment.events
                if event.get("kind") == "warp_call"
            )
            if actual_qualnames != expected_qualnames:
                self._segment.invalidate()
                _get_logger().fatal(
                    f"[{self.node_name}] Warp segment diverged between "
                    "discovery and capture. "
                    f"Expected calls {expected_qualnames}, got {actual_qualnames}.",
                    error_type=RuntimeError,
                )
            return True

        def _exit_capture(
            self,
            exc_type,
            exc_value,
            traceback,
            call_stack,
        ):
            scope_result = False
            if self._scope is not None:
                with self._session.pause():
                    scope_result = self._scope.__exit__(
                        exc_type, exc_value, traceback
                    )
            if exc_type is None and self._segment is not None:
                graph = self._capture.graph
                self._segment.apic_graph = graph
                if not self._finalize_capture():
                    return False
                # Save before replay so formal inputs and closure buffers are
                # snapshotted at the capture boundary, not after execution.
                with self._session.pause():
                    wrp_archive = _save_warp_bundle(self.node_name, self._segment)
                    _insert_warp_marker(
                        self.node_ref,
                        self._segment,
                        wrp_archive=wrp_archive,
                    )
                    wp.capture_launch(graph)
                    wp.synchronize()
                self.node_ref.complete_warp_segment(self._segment)

            return scope_result
