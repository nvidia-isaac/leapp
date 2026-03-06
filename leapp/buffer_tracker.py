#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Auto-detection of stateful buffers for nn.Module export.

Provides BufferTracker, which detects which registered buffers are mutated
during a forward pass and wires them as state tensor I/O in the LEAPP graph.
This allows exporting stateful models (GRU, LSTM, etc.) without any LEAPP
annotations inside the model code.

The model just needs to use standard PyTorch patterns:
    - ``register_buffer("h_state", torch.zeros(...))``
    - ``self.h_state = h_out``  (reassignment in forward)

Usage::

    obs_traced = annotate.input_tensors("policy", {"obs": obs})

    annotate.module("policy", model)
    action = model(obs_traced)

    annotate.output_tensors("policy", {"action": action},
                            export_with="onnx-torchscript")

Note:
    This detects *reassignment* (``self.h = h_out``), not in-place mutation
    (``self.h[:] = h_out`` or ``self.h.copy_(h_out)``). For in-place patterns,
    use the explicit ``state_tensors()`` / ``update_state()`` API instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import TYPE_CHECKING

from leapp.utils.logging import _get_logger

if TYPE_CHECKING:
    from leapp.export_manager import ExportManager
    from leapp.leapp_graph.datatypes import TracedTensor


@dataclass
class _BufferInfo:
    """Tracking info for a single injected buffer."""
    clean_name: str
    original_traced: TracedTensor
    original_buffer: torch.Tensor
    owning_module: nn.Module
    attr_name: str


class BufferTracker:
    """Auto-detects stateful registered buffers by TracedTensor injection.

    Called by ``annotate.module()``.  ``inject()`` replaces registered buffers
    with TracedTensor inputs (via the ``state_tensors()`` API).  ``collect()``
    is triggered automatically by ``compile_trace()`` before the graph is
    frozen: it detects which buffers were reassigned and registers them as
    state outputs (via ``update_state()``).  Non-mutated buffers are baked as
    constants in the exported model.  ``restore()`` re-registers the original
    buffers on the model.

    Args:
        model: The ``nn.Module`` whose buffers to track.
        node_name: Name of the LEAPP node (must already exist from ``input_tensors()``).
        export_manager: The LEAPP ``annotate`` singleton.
        buffer_names: Optional list of buffer names to track (dotted names like
            ``"h_state"`` or ``"encoder.running_mean"``). If ``None``, all
            registered buffers are tracked.
    """

    def __init__(
        self,
        model: nn.Module,
        node_name: str,
        export_manager: ExportManager,
        buffer_names: list[str] | None = None,
    ):
        self._model = model
        self._node_name = node_name
        self._export_manager = export_manager
        self._buffer_names = set(buffer_names) if buffer_names is not None else None
        self._injected: dict[str, _BufferInfo] = {}
        self._collected = False

    def inject(self) -> None:
        """Replace registered buffers with TracedTensors via ``state_tensors()``."""
        from leapp.export_manager import ExportManager

        if not ExportManager._interpret_graph:
            return

        traced_node = self._export_manager.nodes.get(self._node_name)
        if traced_node is None or not traced_node.is_tracing:
            return

        if getattr(traced_node, 'dry_run', False):
            return

        # Collect buffers to track
        buffers_to_track: dict[str, tuple[torch.Tensor, nn.Module, str]] = {}
        for name, buf in self._model.named_buffers():
            if self._buffer_names is not None and name not in self._buffer_names:
                continue
            parts = name.split(".")
            obj = self._model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            attr_name = parts[-1]
            buffers_to_track[name] = (buf, obj, attr_name)

        if not buffers_to_track:
            return

        # Build state dict with clean names (dots -> underscores)
        state_dict = {}
        for full_name, (buf, _, _) in buffers_to_track.items():
            clean_name = full_name.replace(".", "_")
            state_dict[clean_name] = buf

        # Create state tensors via public API
        state_names = list(state_dict.keys())
        traced_values = self._export_manager.state_tensors(self._node_name, state_dict)

        # Handle single-value unpacking: state_tensors() returns TracedTensor
        # for single-entry dicts, tuple for multiple
        if len(state_names) == 1:
            traced_dict = {state_names[0]: traced_values}
        else:
            traced_dict = dict(zip(state_names, traced_values))

        # Inject TracedTensors into model buffers
        # TracedTensor is a torch.Tensor subclass, so setattr accepts it as a buffer
        for full_name, (buf, owning_module, attr_name) in buffers_to_track.items():
            clean_name = full_name.replace(".", "_")
            traced_tensor = traced_dict[clean_name]
            setattr(owning_module, attr_name, traced_tensor)

            self._injected[full_name] = _BufferInfo(
                clean_name=clean_name,
                original_traced=traced_tensor,
                original_buffer=buf,
                owning_module=owning_module,
                attr_name=attr_name,
            )

        _get_logger().info(
            f"BufferTracker: injected {len(self._injected)} buffer(s) "
            f"into node '{self._node_name}'")

    def collect(self) -> None:
        """Detect mutated buffers and register as state outputs.

        Mutated buffers (reassigned during forward) are wired as state outputs
        via ``update_state()``. Non-mutated buffers are baked as constants in
        the FX graph, preserving their trained values (e.g. normalizer mean/var).
        """
        from leapp.export_manager import ExportManager
        from leapp.leapp_graph.traced_node import TracedTensorNode

        if not ExportManager._interpret_graph or not self._injected:
            self._collected = True
            return

        traced_node = self._export_manager.nodes.get(self._node_name)
        if traced_node is None or not traced_node.is_tracing:
            self._collected = True
            return

        mutated: dict[str, object] = {}
        non_mutated: list[str] = []

        for full_name, info in self._injected.items():
            current = getattr(info.owning_module, info.attr_name)
            if current is not info.original_traced:
                mutated[info.clean_name] = current
            else:
                non_mutated.append(info.clean_name)

        # Wire mutated buffers as state outputs
        if mutated:
            self._export_manager.update_state(self._node_name, mutated)
            _get_logger().info(
                f"BufferTracker: detected {len(mutated)} mutated buffer(s): "
                f"{list(mutated.keys())}")

        # Bake non-mutated buffers as constants in the FX graph.
        # They should keep their trained values (e.g. normalizer mean/var)
        # rather than becoming dynamic inputs.
        if non_mutated and isinstance(traced_node, TracedTensorNode):
            clean_to_info = {info.clean_name: info for info in self._injected.values()}

            for clean_name in non_mutated:
                # Remove from state tracking
                if clean_name in traced_node._state_tensors:
                    del traced_node._state_tensors[clean_name]

                # Store original buffer value as constant on tracer root module
                info = clean_to_info[clean_name]
                const_attr = f"_buffer_{clean_name}"
                setattr(traced_node.tracer.root, const_attr, info.original_buffer)

                # Replace placeholder node with get_attr (constant) in FX graph
                for node in list(traced_node.graph.nodes):
                    if node.op == "placeholder" and node.name == clean_name:
                        with traced_node.graph.inserting_before(node):
                            const_node = traced_node.graph.get_attr(const_attr)
                        node.replace_all_uses_with(const_node)
                        traced_node.graph.erase_node(node)
                        break

                # Remove from input descriptions (no longer a dynamic input)
                traced_node.inputs = [
                    d for d in traced_node.inputs if d.name != clean_name
                ]

            _get_logger().info(
                f"BufferTracker: baked {len(non_mutated)} non-mutated buffer(s) "
                f"as constants: {non_mutated}")

        self._collected = True

    def restore(self) -> None:
        """Re-register original buffers on the model.

        Extracts raw tensors from any TracedTensors and restores them as
        proper ``register_buffer`` entries.
        """
        for full_name, info in self._injected.items():
            current = getattr(info.owning_module, info.attr_name)
            # Extract raw tensor from TracedTensor
            raw = current.tensor if hasattr(current, "tensor") else current
            if not isinstance(raw, torch.Tensor):
                raw = info.original_buffer
            info.owning_module.register_buffer(info.attr_name, raw)

        self._injected.clear()

