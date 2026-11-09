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
    (``self.h[:] = h_out`` or ``self.h.copy_(h_out)``), which raises an error
    when LEAPP detects it (the graph guide gives the exact rule).
    For in-place patterns, leave the buffer out of ``buffer_names`` and use the
    explicit ``state_tensors()`` / ``update_state()`` API instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.multiprocessing.reductions import StorageWeakRef
from typing import TYPE_CHECKING

from leapp.utils.logging import _get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import fx

    from leapp.export_manager import ExportManager
    from leapp.leapp_graph.datatypes import TracedTensor


def _memory(value: torch.Tensor) -> StorageWeakRef:
    """The memory a tensor or a TracedTensor uses."""
    from leapp.leapp_graph.datatypes import is_traced_type

    plain = value.tensor if is_traced_type(value) else value
    return StorageWeakRef(plain.untyped_storage())


def _plain(value: torch.Tensor, copies: dict, ended: bool = True) -> torch.Tensor:
    """A plain tensor for a tensor, a TracedTensor, or a view of one, made once
    per tensor in ``copies``.

    Without autograd history it is an alias without the TracedTensor as its
    base, so it keeps the memory, views, version counter, and lazy negation or
    conjugation it shares with other tensors, as in eager. A value with
    autograd history gets a view
    of its plain base again, or a clone, which keeps that history, once the
    session ``ended``. Before that it gets
    the alias the TracedTensor gives, which keeps its history and its memory,
    so a later node that declares the value as an input still reads it there.
    Either is made the way the value was: inference_mode(False) also turns
    autograd back on.
    """
    from leapp.leapp_graph.datatypes import is_traced_type

    if id(value) not in copies:
        plain = value.tensor if is_traced_type(value) else value
        # The root of a view, read from the plain alias, since reading it from
        # a TracedTensor records into a graph.
        base = plain._base
        if plain.requires_grad and not ended:
            copies[id(value)] = plain
        else:
            with torch.inference_mode(plain.is_inference()):
                if (
                    plain.requires_grad
                    and base is not None
                    and not is_traced_type(base)
                ):
                    # A view of a plain tensor, as of a parameter, is an alias
                    # made again here, where autograd is on, so it shares its
                    # memory, version counter, lazy negation or conjugation,
                    # and exactly the gradient connection the value has.
                    copies[id(value)] = value.tensor if is_traced_type(value) else plain
                elif not plain.requires_grad:
                    copies[id(value)] = plain.detach()
                elif (moved := _moved(plain, copies)) is not None:
                    # A view of a copy of its whole memory, so the buffers on
                    # that memory keep sharing it.
                    copies[id(value)] = moved.as_strided(
                        plain.size(), plain.stride(), plain.storage_offset()
                    )
                else:
                    copies[id(value)] = plain.clone()
    return copies[id(value)]


def _moved(plain: torch.Tensor, copies: dict) -> torch.Tensor | None:
    """A copy of the whole memory of a tensor with autograd history, made once
    per tensor it views in ``copies`` through that tensor, so every view of it
    keeps its history. None when the view reads the memory another way."""
    from leapp.leapp_graph.datatypes import is_traced_type
    from leapp.leapp_graph.traced_node import lazy_flags

    base = plain._base
    root = plain if base is None else base
    root = root.tensor if is_traced_type(root) else root
    # A copy reads its bytes plainly, so neither may read them negated or
    # conjugated.
    if root.dtype != plain.dtype or any(lazy_flags(plain) + lazy_flags(root)):
        return None
    memory = _memory(plain)
    # One copy per tensor viewed, so views of one tensor share its history,
    # while another tensor with its own history on the same memory, as an
    # independent leaf, gets its own. The first one is what plain buffers on
    # those bytes view, found by address, since another storage can hold them.
    key = (memory, id(plain if base is None else base))
    if key not in copies:
        storage = plain.untyped_storage()
        size = storage.nbytes() // plain.element_size()
        copies[key] = root.as_strided((size,), (1,), 0).clone()
        copies.setdefault(
            ("bytes", storage.device, storage.data_ptr()),
            (storage.nbytes(), copies[key]),
        )
    return copies[key]


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors are one tensor, its bytes in the same layout read
    the same way, or empty tensors of one shape, which hold no values."""
    from leapp.leapp_graph.datatypes import is_traced_type
    from leapp.leapp_graph.datatypes.proxy import layout_key
    from leapp.leapp_graph.traced_node import lazy_flags

    if a is b:
        return True
    plain_a, plain_b = (v.tensor if is_traced_type(v) else v for v in (a, b))
    if plain_a.numel() == 0 and plain_b.numel() == 0:
        return plain_a.shape == plain_b.shape
    key = layout_key(a)
    return key is not None and key == layout_key(b) and lazy_flags(a) == lazy_flags(b)


def _source(node: fx.Node) -> object:
    """The first argument of an FX node, the value a view views, also when it
    is passed by keyword, as in torch.reshape(input=x, shape=s)."""
    return (
        node.args[0] if node.args else node.kwargs.get("input", node.kwargs.get("self"))
    )


def _leads_to(node: object, nodes: set) -> bool:
    """Whether an FX node, or one its chain of first arguments reaches, is in
    ``nodes``."""
    while isinstance(node, torch.fx.Node):
        if node in nodes:
            return True
        node = _source(node)
    return False


def _same_value(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors hold the same values up to rounding, a NaN matching
    a NaN."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # A meta tensor holds no values. Most writes leave the bytes equal, which
    # costs no copy, unlike isclose().
    if a.is_meta or torch.equal(a, b):
        return True
    if a.is_floating_point():
        # A few units in the last place of the dtype kernels accumulate in,
        # float64 for float64 and float32 otherwise, as when one writes back the
        # mean of many identical copies, or one of the buffer's own dtype.
        accumulate = torch.float64 if a.dtype == torch.float64 else torch.float32
        rtol = max(16 * torch.finfo(accumulate).eps, torch.finfo(a.dtype).eps)
        return bool(torch.isclose(a, b, rtol=rtol, atol=0, equal_nan=True).all())
    return torch.equal(a, b)


@dataclass
class _BufferInfo:
    """Tracking info for a single injected buffer."""

    clean_name: str
    original_traced: TracedTensor
    original_buffer: torch.Tensor
    owning_module: nn.Module
    attr_name: str
    # It was a plain tensor when tracked, so restore() turns it back.
    promoted: bool
    # The graph input the buffer entered as.
    placeholder: fx.Node
    # A copy when tracked, which a write the graph does not record is compared
    # with, since the input's description holds the first pass's value on a
    # later pass.
    snapshot: torch.Tensor
    # It held a value the node computed before module(), which a constant would freeze.
    computed: bool
    # Declared with state_tensors() before module(), so it stays the user's state.
    declared: bool
    # Took the input declared under its name before module(), so it stays that input.
    adopted: bool
    # It held another node's output of this session when tracked, or a plain
    # tensor on the memory of a value an earlier node set into a buffer.
    held_output: bool


class BufferTracker:
    """Auto-detects stateful registered buffers by TracedTensor injection.

    Called by ``annotate.module()``.  ``inject()`` replaces registered buffers
    with TracedTensor inputs (via the ``state_tensors()`` API).  ``collect()``
    is triggered automatically by ``compile_trace()`` before the graph is
    frozen: it detects which buffers were reassigned and registers them as
    state outputs (via ``update_state()``).  Non-mutated buffers the graph
    reads are baked as constants in the exported model.  ``restore()`` puts
    the buffers back on the model as plain tensors, except the ones a later
    node may still read (see ``restore()``).

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
        # Promoted buffers a compiled node published as an output or a state.
        self._published: dict[str, _BufferInfo] = {}
        self._collected = False
        # The graph of the pass that tracked the buffers, and the writes it
        # recorded before.
        self._graph = None
        self._earlier_writes: set = set()

    def inject(self) -> None:
        """Replace registered buffers with TracedTensors via ``state_tensors()``."""
        from leapp.export_manager import ExportManager

        if not ExportManager._interpret_graph:
            return

        traced_node = self._export_manager.nodes.get(self._node_name)
        if traced_node is None or not traced_node.is_tracing:
            return

        from leapp.leapp_graph.datatypes import is_traced_type
        from leapp.leapp_graph.datatypes.proxy import (
            WRITTEN_LAYOUT_META_KEY,
            layouts_overlap,
        )
        from leapp.utils.dtype import value_to_name_and_shape

        written = [
            node.meta[WRITTEN_LAYOUT_META_KEY]
            for node in traced_node.graph.nodes
            if node.meta.get(WRITTEN_LAYOUT_META_KEY) is not None
        ]
        # Collect buffers to track
        buffers_to_track: dict[str, tuple[torch.Tensor, nn.Module, str]] = {}
        declared_inputs = []
        computed = set()
        slots = set()
        for name, buf in self._model.named_buffers(remove_duplicate=False):
            if self._buffer_names is not None and name not in self._buffer_names:
                continue
            parts = name.split(".")
            obj = self._model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            attr_name = parts[-1]
            # A submodule reachable under several names has one slot per buffer.
            if (id(obj), attr_name) in slots:
                continue
            slots.add((id(obj), attr_name))
            node = getattr(getattr(buf, "proxy", None), "node", None)
            if (
                is_traced_type(buf)
                and buf.context_obj is traced_node
                and node is not None
                and node.graph is not traced_node.graph
            ):
                # An earlier capture pass left it traced, and its graph is gone.
                buf.clear_tracing_state()
            elif (
                is_traced_type(buf)
                and node is not None
                and node.graph is traced_node.graph
                and name.replace(".", "_") not in traced_node._state_tensors
            ):
                # state_tensors() would bind it to a new graph input, so the
                # user's input or state would lose its reader, or the link to
                # what the node computed it from. A state under its own name
                # is checked below.
                if node.op == "placeholder":
                    declared_inputs.append(name)
                elif node.op.startswith("call_") or any(
                    layouts_overlap(layout, buf) for layout in written
                ):
                    # Also a write in place, as model.h.mul_(1 - done), which
                    # leaves the buffer on a constant of the graph.
                    computed.add(name)
            buffers_to_track[name] = (buf, obj, attr_name)

        if declared_inputs:
            _get_logger().fatal(
                f"annotate.module() cannot track buffer(s) "
                f"{sorted(declared_inputs)} of node '{self._node_name}', since "
                "the node already declares them as inputs, or as states under "
                "other names. Leave them out of buffer_names if the forward "
                "only reads them, or declare them with annotate.state_tensors() "
                "under their state names "
                f"{sorted(n.replace('.', '_') for n in declared_inputs)} if it "
                "reassigns them.",
                error_type=ValueError,
            )
        if not buffers_to_track:
            return
        self._graph = traced_node.graph
        self._earlier_writes = {
            node for node in self._graph.nodes if node.meta.get("leapp_mutation", False)
        }

        # Build state dict with clean names (dots -> underscores)
        state_dict = {}
        groups: dict[str, list[str]] = {}
        # A tensor registered in several slots is declared once, and every slot
        # gets its traced tensor.
        declared_as: dict[int, str] = {}
        same_as: dict[str, str] = {}
        for full_name, (buf, _, _) in buffers_to_track.items():
            clean_name = full_name.replace(".", "_")
            groups.setdefault(clean_name, []).append(full_name)
            if id(buf) in declared_as:
                same_as[clean_name] = declared_as[id(buf)]
                continue
            declared_as[id(buf)] = clean_name
            state_dict[clean_name] = buf
        # Buffers under one name would all get the same traced tensor.
        clashing = sorted(
            n for group in groups.values() if len(group) > 1 for n in group
        )
        if clashing:
            _get_logger().fatal(
                f"annotate.module() cannot track buffers {clashing} of node "
                f"'{self._node_name}', since their names become the same state "
                "name. Leave all but one of them out of buffer_names.",
                error_type=ValueError,
            )
        plain = {
            n
            for n, (buf, _, _) in buffers_to_track.items()
            if type(buf) is torch.Tensor
        }

        # Checked before any buffer is promoted, since state_tensors() fails on
        # such a buffer only after registering the ones before it.
        states = traced_node._state_tensors
        unusable = []
        for full_name, (buf, _, _) in buffers_to_track.items():
            try:
                value_to_name_and_shape(buf)
            except (KeyError, ValueError):
                unusable.append(f"{full_name} ({buf.dtype})")
                continue
            if buf.layout is not torch.strided:
                unusable.append(f"{full_name} ({buf.layout})")
            elif (
                is_traced_type(buf)
                and buf.is_tracing
                and buf.context_obj is not traced_node
            ):
                unusable.append(f"{full_name} (traced by node '{buf.context}')")
        if unusable:
            _get_logger().fatal(
                f"annotate.module() cannot track buffer(s) {unusable} of node "
                f"'{self._node_name}', since a node input cannot take their dtype "
                "or layout, or a node whose capture has not ended traces them. "
                "Leave them out of buffer_names, or call output_tensors() on "
                "the other node first.",
                error_type=ValueError,
            )

        # Checked before any buffer is promoted, since state_tensors() fails on
        # an input name halfway through, and for a state name returns the
        # node's existing tensor, which the forward would read instead of the
        # buffer, unless it is the buffer itself or its bytes in the same layout.
        declared_states = {n for n in buffers_to_track if n.replace(".", "_") in states}
        # By placeholder, since a Warp capture pass keeps the inputs of the
        # first pass, which the buffers re-declare.
        input_names = {
            node.target for node in traced_node.graph.nodes if node.op == "placeholder"
        }
        taken = []
        adopted = set()
        for full_name, (buf, _, _) in buffers_to_track.items():
            name = full_name.replace(".", "_")
            state = states.get(name)
            if state is None:
                if name in input_names:
                    # Accepted when the input is the buffer's own bytes in the
                    # same layout, as for a state, since the buffer adopts it.
                    declared = traced_node.find_declared_alias(buf)
                    if getattr(declared, "name", None) != name:
                        taken.append(full_name)
                    else:
                        adopted.add(full_name)
            elif not _same_bytes(state["input"], buf):
                taken.append(full_name)
        if taken:
            _get_logger().fatal(
                f"annotate.module() cannot track buffer(s) {sorted(taken)} of node "
                f"'{self._node_name}', since the node already has other inputs "
                f"under their state names {sorted(n.replace('.', '_') for n in taken)}. "
                "Rename one of them, declare the buffer itself rather than a copy, "
                "or leave the buffers out of buffer_names.",
                error_type=ValueError,
            )

        # Create state tensors via public API
        state_names = list(state_dict.keys())
        existing = set(states)
        try:
            traced_values = self._export_manager.state_tensors(
                self._node_name, state_dict
            )

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
                traced_tensor = traced_dict[same_as.get(clean_name, clean_name)]
                setattr(owning_module, attr_name, traced_tensor)

                self._injected[full_name] = _BufferInfo(
                    clean_name=clean_name,
                    original_traced=traced_tensor,
                    original_buffer=buf,
                    owning_module=owning_module,
                    attr_name=attr_name,
                    promoted=full_name in plain,
                    placeholder=traced_tensor.proxy.node,
                    snapshot=traced_tensor.tensor.clone(),
                    computed=full_name in computed,
                    declared=full_name in declared_states,
                    adopted=full_name in adopted,
                    held_output=self._holds_output(buf, traced_node),
                )
        except Exception:
            # state_tensors() promotes the buffers one by one, and setattr()
            # can refuse one, also after storing it, while the node keeps no
            # tracker that would restore the others.
            for full_name, (buf, owning_module, attr_name) in buffers_to_track.items():
                if full_name in plain and is_traced_type(buf):
                    buf.clear_tracing_state()
                owning_module._buffers[attr_name] = buf
            # A tensor subclass gets its own traced tensor instead, which a
            # repeated call would take for a state the user declared.
            for name in states.keys() - existing:
                if is_traced_type(states[name]["input"]):
                    states[name]["input"] = states[name]["input"].tensor
            raise

        _get_logger().info(
            f"BufferTracker: injected {len(self._injected)} buffer(s) "
            f"into node '{self._node_name}'"
        )

    def collect(self, outputs: dict[str, object]) -> None:
        """Detect mutated buffers and register as state outputs.

        Mutated buffers (reassigned during forward) are wired as state outputs
        via ``update_state()``. Non-mutated buffers the graph reads are baked as
        constants in the FX graph, preserving their trained values (e.g.
        normalizer mean/var).

        Args:
            outputs: The node's outputs, which decide what the graph reads.
        """
        from leapp.export_manager import ExportManager
        from leapp.leapp_graph.datatypes import is_traced_type
        from leapp.leapp_graph.datatypes.proxy import (
            WRITTEN_LAYOUT_META_KEY,
            layouts_overlap,
        )
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
        kept_inputs: list[str] = []

        # Buffers on the same memory in the same layout share one graph input,
        # which cannot carry a new value for only one of them.
        sharing: dict[object, list[str]] = {}
        for full_name, info in self._injected.items():
            sharing.setdefault(info.placeholder, []).append(full_name)
        # A slot left out of buffer_names that holds a tracked tensor reads
        # its input too, and keeps that tensor.
        inputs = {id(i.original_traced): i.placeholder for i in self._injected.values()}
        left_out: dict[object, list[tuple[str, torch.Tensor]]] = {}
        for name, buf in self._model.named_buffers(remove_duplicate=False):
            if name not in self._injected and id(buf) in inputs:
                left_out.setdefault(inputs[id(buf)], []).append((name, buf))
        for placeholder, names in sharing.items():
            current = [
                getattr(self._injected[n].owning_module, self._injected[n].attr_name)
                for n in names
            ]
            reassigned = any(
                value is not self._injected[n].original_traced
                for n, value in zip(names, current)
            )
            names = names + [name for name, _ in left_out.get(placeholder, [])]
            current += [buf for _, buf in left_out.get(placeholder, [])]
            # Views of one memory in one layout still describe one value, and
            # a slot set to None no longer reads it.
            tensors = [value for value in current if value is not None]
            if reassigned and any(
                not _same_bytes(value, tensors[0]) for value in tensors
            ):
                _get_logger().fatal(
                    f"Buffers {sorted(names)} of node '{self._node_name}' share "
                    "memory in the same layout"
                    + (
                        ", or one left out of buffer_names holds the tensor of a "
                        "tracked one"
                        if placeholder in left_out
                        else ""
                    )
                    + ", so the graph reads them through one input, and the "
                    "forward sets them to different tensors. Give each buffer its "
                    "own tensor"
                    + (
                        ", or list every buffer the forward sets in buffer_names"
                        if placeholder in left_out
                        else ""
                    )
                    + ".",
                    error_type=RuntimeError,
                )
        tracked = {info.clean_name for info in self._injected.values()}
        # The graph keeps only what the outputs, states and buffers depend on,
        # so a read in a discarded branch does not count, while one a write
        # into a buffer makes does, also one the graph replays on an input.
        roots = [*outputs.values()]
        roots += [state["output"] for state in traced_node._state_tensors.values()]
        roots += [
            getattr(info.owning_module, info.attr_name)
            for info in self._injected.values()
        ]
        pending = [
            value.proxy.node
            for value in roots
            if is_traced_type(value) and value.proxy is not None
        ]
        writes = [
            node
            for node in traced_node.graph.nodes
            if node.meta.get("leapp_mutation", False)
        ]
        live = set()
        while pending:
            while pending:
                node = pending.pop()
                if node not in live:
                    live.add(node)
                    pending.extend(node.all_input_nodes)
            # A recorded write into a live value, as in y += self.goal, or into
            # a view of one, as in y[:2].div_(self.scale), keeps what it reads.
            # A view's first argument leads back to the value it views, while
            # a write into a temporary computed from a live value also counts.
            pending = [
                n for n in writes if n not in live and _leads_to(n.args[0], live)
            ]
        # A write recorded before module() is part of the value the buffer
        # holds, which the computed check covers.
        written = [
            node.meta[WRITTEN_LAYOUT_META_KEY]
            for node in writes
            if node not in self._earlier_writes
            and node.meta.get(WRITTEN_LAYOUT_META_KEY) is not None
        ]

        for full_name, info in self._injected.items():
            current = getattr(info.owning_module, info.attr_name)
            reassigned = current is not info.original_traced
            # A tracked buffer on the same memory may own the graph input, while
            # a buffer read through an input the user declared reads that input
            # instead, and the graph replays a write to it. Matched by target,
            # since FX renames a placeholder such as max to max_1.
            own_input = info.placeholder.target in tracked
            read = own_input and any(user in live for user in info.placeholder.users)
            # Unless the user declared its tensor as another input after
            # module(), which the graph replays a write to.
            node = getattr(info.original_traced.proxy, "node", None)
            traced_here = own_input and (
                node is info.placeholder or getattr(node, "op", None) != "placeholder"
            )
            # Declared with state_tensors() and updated with update_state(), so
            # it is the user's state, not a constant, and the update carries an
            # in-place change. A buffer the user declared otherwise stays an
            # input, so the graph replays the change.
            state = traced_node._state_tensors.get(info.clean_name)
            user_state = state is not None and state["output"] is not None
            # Views share one counter, so only a write on the buffer's bytes
            # counts.
            recorded = any(
                layouts_overlap(layout, info.original_traced) for layout in written
            )
            # A write the graph did not record counts only when it moved the
            # value beyond rounding, since some kernels write back the same
            # values, as InstanceNorm in evaluation mode does on torch 2.7, and
            # it may go through a tensor with its own counter or none. A recorded
            # one counts always, since it may do nothing only for the capture
            # sample.
            moved = not _same_value(info.original_traced.tensor, info.snapshot)
            changed = moved or recorded
            unrecorded = moved and not recorded
            # The constant the buffer is baked as would lose the write, or the
            # graph would replay it, so the export would not match, also for a
            # later node that reads the buffer when this one does not. For
            # another node's output, even one the forward then reassigns, the
            # write would change that node's output in eager only. The graph
            # replays a recorded write before a reassignment, but not one it did
            # not record. Declaring the buffer does not change that for another
            # node's output.
            if (
                (traced_here or info.held_output)
                and (
                    info.held_output
                    or not (user_state or info.declared or info.adopted)
                )
                and (info.held_output or not reassigned or unrecorded)
                and changed
            ):
                _get_logger().fatal(
                    f"Buffer '{full_name}' of node '{self._node_name}' was changed "
                    "in place during the capture, or another tensor on its bytes "
                    "was. annotate.module() cannot carry an in-place "
                    "change. Assign a new tensor instead, as in "
                    f"self.{info.attr_name} = self.{info.attr_name} + 1, or leave "
                    "the buffer out of buffer_names and declare it with "
                    "annotate.state_tensors() and annotate.update_state(). For a "
                    "layer's running statistics or num_batches_tracked, leave "
                    "them out of buffer_names if the layer runs in training "
                    "mode on purpose, or call model.eval() before the capture "
                    "if the model is meant for inference, which also makes the "
                    "layer use those statistics."
                    + (
                        " The buffer was created under torch.inference_mode(), "
                        "where a read that returns the tensor itself, as "
                        ".float(), also counts as a write. Create it outside "
                        "inference mode."
                        if info.original_traced.tensor.is_inference()
                        else ""
                    ),
                    error_type=RuntimeError,
                )
            # A value the graph made before module() changes every step, while
            # the export would bake it, or start the feedback from it, also
            # one the user's update_state() feeds back.
            if (
                read
                and not (info.declared or info.adopted)
                and (info.computed or info.held_output)
            ):
                _get_logger().fatal(
                    f"Buffer '{full_name}' of node '{self._node_name}' holds a "
                    "value the graph computed before annotate.module(), in this "
                    "node or as another node's output, which the export would "
                    "not compute again. Set a value of this node after "
                    "annotate.module(), where the graph records it. Output "
                    "another node's value from that node, pass it to "
                    "annotate.input_tensors() of this node, set the buffer to "
                    "what that returns, and leave the buffer out of "
                    "buffer_names.",
                    error_type=RuntimeError,
                )
            # The buffer reads an input the user declared for another node's
            # output, also one it adopted under its own name, which the
            # feedback of its new value would replace.
            if (
                reassigned
                and info.held_output
                and (info.adopted or info.placeholder.target not in tracked)
            ):
                _get_logger().fatal(
                    f"Buffer '{full_name}' of node '{self._node_name}' reads "
                    f"input '{info.placeholder.target}', which holds another "
                    "node's output, and the forward sets it to a new tensor, "
                    "which the export would feed back into that input in place "
                    "of that output. Leave the buffer out of buffer_names, so "
                    "the node reads that input on every step.",
                    error_type=RuntimeError,
                )
            # The user's update_state() wins over the forward's reassignment,
            # and a state the user declared and did not update stays a regular
            # input.
            if user_state or (info.declared and not reassigned):
                continue
            if sharing[info.placeholder][0] != full_name:
                # The first slot on a shared graph input decides, which
                # module() declared as a state, also when the input is one
                # the user declared under the name of another slot.
                continue
            if reassigned:
                mutated[info.clean_name] = current
                continue
            description = traced_node.get_io_description_by_name(
                info.clean_name, traced_node.inputs
            )
            if info.adopted or any(n not in tracked for n in description.aliases):
                # An input the user declared for the same memory reads its
                # placeholder, which baking would erase, so it stays an input.
                kept_inputs.append(info.clean_name)
                continue
            non_mutated.append(info.clean_name)

        # The node changes only after every check passed, so a failed call
        # can be repeated.
        if mutated:
            self._export_manager.update_state(self._node_name, mutated)
        for clean_name in kept_inputs:
            del traced_node._state_tensors[clean_name]

        if mutated:
            _get_logger().info(
                f"BufferTracker: detected {len(mutated)} mutated buffer(s): "
                f"{list(mutated.keys())}"
            )

        # Bake non-mutated buffers as constants in the FX graph.
        # They should keep their trained values (e.g. normalizer mean/var)
        # rather than becoming dynamic inputs.
        if non_mutated and isinstance(traced_node, TracedTensorNode):
            clean_to_info = {info.clean_name: info for info in self._injected.values()}
            baked = []

            for clean_name in non_mutated:
                # Remove from state tracking
                if clean_name in traced_node._state_tensors:
                    del traced_node._state_tensors[clean_name]

                # Replace placeholder node with get_attr (constant) in FX graph
                info = clean_to_info[clean_name]
                for node in list(traced_node.graph.nodes):
                    if node.op == "placeholder" and node.target == clean_name:
                        if node.users:
                            # A copy, so a later in-place change to the buffer
                            # does not reach the export, made outside inference
                            # mode, since validation cannot use an inference
                            # tensor as a constant. A read the graph drops
                            # gets the buffer itself, since pruning removes it.
                            const_attr = f"_buffer_{clean_name}"
                            value = info.original_traced.tensor.detach()
                            if any(user in live for user in node.users):
                                with torch.inference_mode(False):
                                    value = value.clone()
                            setattr(traced_node.tracer.root, const_attr, value)
                            with traced_node.graph.inserting_before(node):
                                const_node = traced_node.graph.get_attr(const_attr)
                            node.replace_all_uses_with(const_node)
                            baked.append(clean_name)
                        traced_node.graph.erase_node(node)
                        break

                # Remove from input descriptions (no longer a dynamic input)
                traced_node.inputs = [
                    d for d in traced_node.inputs if d.name != clean_name
                ]

            _get_logger().info(
                f"BufferTracker: baked {len(baked)} non-mutated buffer(s) "
                f"as constants: {baked}"
            )

        # A tensor constant the graph took from an untracked tensor on a
        # buffer's memory, as a view of it, still has the traced buffer as its
        # base, which torch.export rejects.
        root = traced_node.tracer.root
        for node in traced_node.graph.nodes:
            value = getattr(root, node.target, None) if node.op == "get_attr" else None
            if (
                isinstance(value, torch.Tensor)
                and not is_traced_type(value)
                and is_traced_type(value._base)
            ):
                setattr(root, node.target, _plain(value, {}))

        self._collected = True

    def _holds_output(self, buf: torch.Tensor, traced_node) -> bool:
        """Whether a buffer holds an output of a node of this session, or a
        plain tensor on the memory of a value another node's forward set into
        a buffer, its output or state, which that node's restore() made plain.
        An output left from an earlier session belongs to no node now."""
        from leapp.leapp_graph.datatypes.proxy import layouts_overlap

        if getattr(buf, "output_port", None) is not None:
            return self._in_session(buf)
        # By address, since another storage, as from torch.from_dlpack(), may
        # wrap the same bytes.
        return any(
            node is not traced_node and layouts_overlap(value, buf)
            for value, node in self._export_manager.node_outputs
        )

    def _in_session(self, value: TracedTensor) -> bool:
        """Whether a traced value belongs to a node of the current session."""
        return any(
            node is value.context_obj for node in self._export_manager.nodes.values()
        )

    def restore(
        self,
        published: Iterable[torch.Tensor] = (),
        compiled: bool = False,
        ended: bool = False,
        copies: dict | None = None,
    ) -> None:
        """Put the buffers back on the model as plain tensors.

        A buffer the forward kept is the original tensor again, and a
        reassigned buffer holds a plain tensor with its new value. Like every
        node output and declared input, some stay traced, so a later node that
        reads them connects to them: another node's output the buffer held,
        a buffer's tensor in ``published`` (the node's outputs) or among its
        state outputs, and one declared with ``input_tensors()`` after
        ``module()``. ``module()`` rejects a buffer
        already declared as an input of the node, and one declared with
        ``state_tensors()`` stays as the user made it. Once the session
        ``ended``, every buffer it tracked holds a plain tensor, and
        make_traced_buffers_plain() handles the rest of the model, with the
        same ``copies``, so slots that held one tensor still do.

        Args:
            published: The node's outputs.
            compiled: Whether the node compiled. After a failure, the outputs
                and state inputs stay traced for a repeated call.
            ended: Whether the session ended.
            copies: The plain tensors made so far, shared across trackers.
        """
        from leapp.leapp_graph.datatypes import is_traced_type

        this_node = self._export_manager.nodes.get(self._node_name)
        kept = {id(tensor) for tensor in published}
        outputs = set(kept)
        if not ended:
            # A later node can read a state output too, and a repeated call
            # after a failure reads the node's states.
            for state in this_node._state_tensors.values():
                kept.add(id(state["output"]))
                outputs.add(id(state["output"]))
                if not compiled:
                    kept.add(id(state["input"]))
        pending = {}
        infos = dict(self._injected)
        if ended:
            infos.update(self._published)
            self._published = {}
        # Cleared in place, so every module and attribute that holds the tensor
        # sees it plain, since torch.export cannot take a TracedTensor.
        for name, info in infos.items():
            buf = info.original_buffer
            if not is_traced_type(buf):
                continue
            node = getattr(buf.proxy, "node", None)
            redeclared = (
                node is not info.placeholder
                and getattr(node, "op", None) == "placeholder"
            )
            # Stays traced: a node output, a tensor the user made traced, as with
            # state_tensors(), one declared again as a graph input, or one
            # another node wrote into and now owns. After a failure the node
            # keeps it for a repeated call, and the end of the session makes its
            # buffer plain. A tensor of the user's that this node published
            # becomes plain again in place once the session ends.
            own = info.promoted and not redeclared and buf.context_obj is this_node
            if id(buf) in kept or not own:
                if not ended:
                    if not compiled:
                        pending[name] = info
                    elif own:
                        self._published[name] = info
                continue
            buf.clear_tracing_state()
        copies = {} if copies is None else copies
        for info in self._injected.values():
            # The capture deleted it or registered a parameter or a module in
            # its place.
            if info.attr_name not in info.owning_module._buffers:
                continue
            current = getattr(info.owning_module, info.attr_name)
            if current is info.original_traced:
                current = info.original_buffer
            # Another node that is still capturing traced it into the slot.
            if (
                is_traced_type(current)
                and current.context_obj is not this_node
                and current.is_tracing
            ):
                continue
            # The original tensor is still traced only when it is kept on
            # purpose, so only another traced value, or one left from an
            # earlier session, is replaced by a plain tensor, one per tensor, so
            # slots that held one tensor still do. The end of the session makes
            # the rest plain.
            if is_traced_type(current) and (
                current is not info.original_buffer or id(current) in outputs
            ):
                # This buffer still holds this node's output or state for a
                # later node that reads it, also before a repeated call after a
                # failure compiles it, and also when it is the original tensor.
                self._export_manager.node_outputs.append((current, this_node))
            if is_traced_type(current) and (
                current is not info.original_buffer or not self._in_session(current)
            ):
                current = _plain(current, copies, ended)
            # Into the slot, since register_buffer() would make a non-persistent
            # buffer persistent, and nn.Module.__setattr__ would make a
            # Parameter a parameter.
            info.owning_module._buffers[info.attr_name] = current
        self._injected = pending

    def make_traced_buffers_plain(self, copies: dict) -> None:
        """Give every buffer of the model that still holds a traced tensor, or
        a view of one, a plain tensor, once the session ended and no node can
        connect to it, as one a later pass, another node or the user left
        there."""
        from leapp.leapp_graph.datatypes import is_traced_type

        for module in self._model.modules():
            for attr, value in module._buffers.items():
                if is_traced_type(value) or is_traced_type(
                    getattr(value, "_base", None)
                ):
                    module._buffers[attr] = _plain(value, copies)
        # A plain buffer on bytes a copy replaced views the copy instead, so it
        # keeps sharing them with the buffers that moved. A tensor subclass, as
        # a Parameter, keeps its object, which a view would not be.
        moved = [
            (key[1], key[2], *entry)
            for key, entry in copies.items()
            if isinstance(key, tuple) and isinstance(key[0], str)
        ]
        for module in self._model.modules():
            for attr, value in module._buffers.items():
                if (
                    type(value) is not torch.Tensor
                    or value.layout != torch.strided
                    or value.requires_grad
                    or value.is_neg()
                    or value.is_conj()
                ):
                    continue
                storage = value.untyped_storage()
                size = value.element_size()
                for device, start, nbytes, copy in moved:
                    begin = storage.data_ptr() - start
                    if (
                        storage.device == device
                        and 0 <= begin <= nbytes - storage.nbytes()
                        and begin % size == 0
                        and nbytes % size == 0
                    ):
                        module._buffers[attr] = (
                            copy.detach()
                            .view(value.dtype)
                            .as_strided(
                                value.size(),
                                value.stride(),
                                value.storage_offset() + begin // size,
                            )
                        )
                        break
