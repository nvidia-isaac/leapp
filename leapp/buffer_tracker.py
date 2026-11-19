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
    (``self.h[:] = h_out`` or ``self.h.copy_(h_out)``). For in-place patterns,
    use the explicit ``state_tensors()`` / ``update_state()`` API instead.
"""

from __future__ import annotations

import collections
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
    was_buffer: bool


def _register_buffer(module, name: str, tensor: torch.Tensor) -> None:
    """Register ``tensor`` as buffer ``name``.

    torch's own ``register_buffer()`` also gets the buffer's persistent flag.
    Any other gets ``register_buffer(name, tensor)``, since it may take the
    flag in a way LEAPP cannot see.
    """
    if _is_torch_register_buffer(module):
        persistent = name not in module._non_persistent_buffers_set
        module.register_buffer(name, tensor, persistent)
    else:
        module.register_buffer(name, tensor)


def _requires_grad(value) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    # Reading it through a TracedTensor would record a node in the capture.
    with torch._C.DisableTorchFunctionSubclass():
        return value.requires_grad


# torch's own; another __setattr__ can store something other than what LEAPP
# assigns, or write another model's buffer.
_TORCH_SETATTRS = (
    nn.Module.__setattr__,
    nn.RNNBase.__setattr__,
    torch.jit.RecursiveScriptModule.__setattr__,
)


def _from_torch(obj) -> bool:
    # functools.wraps copies __module__ from what it wraps.
    module = getattr(obj, "__module__", None)
    return (
        isinstance(module, str)
        and module.startswith("torch.")
        and not hasattr(obj, "__wrapped__")
    )


def _is_torch_register_buffer(module) -> bool:
    """Whether ``module.register_buffer`` is torch's own, bound to ``module``."""
    if not isinstance(module, nn.Module):
        return False
    register = module.register_buffer
    function = getattr(register, "__func__", None)
    return (
        function is nn.Module.register_buffer
        and getattr(register, "__self__", None) is module
        # A replacement of torch's method, also one made before this module
        # was imported, is defined outside torch.
        and _from_torch(function)
    )


def _only_torch_code_runs(owner) -> bool:
    """Whether registering a buffer of ``owner`` runs only torch's code.

    That is ``owner``'s register_buffer(), __setattr__, the attribute
    reads that torch's register_buffer() makes, the parameter and buffer
    containers and the buffer registration hooks. A read that gives back another
    tensor is caught in ``BufferTracker.restore()``.
    """
    if not _is_torch_register_buffer(owner):
        return False
    setattr_ = type(owner).__setattr__
    # nn.Module.__getattr__ looks in _parameters before _buffers.
    containers = [
        type(owner.__dict__.get(name)) for name in ("_parameters", "_buffers")
    ]
    return (
        setattr_ in _TORCH_SETATTRS
        # Also a replacement made before this module was imported.
        and _from_torch(setattr_)
        and type(owner).__getattribute__ is object.__getattribute__
        and _from_torch(type(owner).__getattr__)
        # torch 2.4 and older used OrderedDicts, which old pickles keep.
        and all(
            kind in (dict, collections.OrderedDict) or _from_torch(kind)
            for kind in containers
        )
        and type(getattr(owner, "_non_persistent_buffers_set", None)) is set
        # torch's own hooks are allowed too, such as the one a failed
        # torch.export leaves behind on torch 2.7.
        and all(
            _from_torch(hook)
            for hook in nn.modules.module._global_buffer_registration_hooks.values()
        )
    )


# What _slot() returns for a name that holds nothing, not even None.
_EMPTY = object()


def _slot(owner, name: str):
    """What ``owner`` holds under ``name``: its registered buffer, if any."""
    buffers = owner.__dict__.get("_buffers") if isinstance(owner, nn.Module) else None
    if buffers is not None and name in buffers:
        return buffers[name]
    return getattr(owner, name, _EMPTY)


class BufferTracker:
    """Auto-detects stateful registered buffers by TracedTensor injection.

    Called by ``annotate.module()``.  ``inject()`` replaces registered buffers
    with TracedTensor inputs (via the ``state_tensors()`` API).  ``collect()``
    is triggered automatically by ``compile_trace()`` before the graph is
    frozen: it detects which buffers were reassigned and registers them as
    state outputs (via ``update_state()``).  Non-mutated buffers are baked as
    constants in the exported model.  ``restore()`` puts each injected buffer's
    tensor back, which can still be tied to the capture, and ``release()``,
    run when the session ends unless buffers must stay tied, makes those
    buffers plain tensors, with the exceptions it lists.

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
        self._restored: list[tuple[nn.Module, str]] = []
        self._collected = False

    def inject(self) -> None:
        """Replace registered buffers with TracedTensors via ``state_tensors()``."""
        from leapp.export_manager import ExportManager

        if not ExportManager._interpret_graph:
            return

        traced_node = self._export_manager.nodes.get(self._node_name)
        if traced_node is None or not traced_node.is_tracing:
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

        # state_tensors() refuses a buffer still active in another node only
        # when it reaches it, after promoting the buffers before it. Refuse
        # first, with the same error and without recording an input, so the
        # call leaves the model and the node unchanged. A name the node
        # already has is reused, not refused.
        for full_name, (buf, _, _) in buffers_to_track.items():
            clean_name = full_name.replace(".", "_")
            owner_node = getattr(buf, "context_obj", None)
            if (
                getattr(buf, "is_tracing", False)
                and owner_node is not traced_node
                and clean_name not in traced_node._state_tensors
            ):
                traced_node._create_io_helper(buf, clean_name, to="traced")

        grad_buffers = self._export_manager.grad_buffers
        # Read from the buffers: the TracedTensor that a later node injects
        # for a buffer can lack its flag.
        grad_names = {
            full_name
            for full_name, (buf, _, _) in buffers_to_track.items()
            if _requires_grad(buf)
        }

        # Build state dict with clean names (dots -> underscores)
        state_dict = {}
        for full_name, (buf, owning_module, attr_name) in buffers_to_track.items():
            clean_name = full_name.replace(".", "_")
            # Two buffers of this call with one state name share the last
            # one's TracedTensor (state_tensors() handles a name the node
            # already has). Code other than torch's that runs when the buffer
            # is assigned or put back can change it, or another model's
            # buffer, unseen. If named_buffers() lists another tensor under
            # the name of something the owner holds, even a buffer registered
            # as None, LEAPP would write that tensor over it.
            if (
                clean_name in state_dict
                or not _only_torch_code_runs(owning_module)
                or (slot := _slot(owning_module, attr_name)) is not _EMPTY
                and slot is not buf
            ):
                self._export_manager.keep_buffers_tied = True
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
            was_buffer = (
                isinstance(owning_module, nn.Module)
                and attr_name in owning_module._buffers
            )
            if was_buffer and _only_torch_code_runs(owning_module):
                # setattr() would take the flag of an nn.Buffer, which the
                # promoted tensor still carries, over the buffer's own.
                _register_buffer(owning_module, attr_name, traced_tensor)
            else:
                setattr(owning_module, attr_name, traced_tensor)
            if full_name in grad_names:
                grad_buffers.append(traced_tensor)

            self._injected[full_name] = _BufferInfo(
                clean_name=clean_name,
                original_traced=traced_tensor,
                original_buffer=buf,
                owning_module=owning_module,
                attr_name=attr_name,
                was_buffer=was_buffer,
            )

        _get_logger().info(
            f"BufferTracker: injected {len(self._injected)} buffer(s) "
            f"into node '{self._node_name}'"
        )

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
                f"{list(mutated.keys())}"
            )

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
                f"as constants: {non_mutated}"
            )

        self._collected = True

    def restore(self, at_session_end: bool = False, latest: bool = True) -> None:
        """Put each injected buffer's tensor back into the model.

        Each injected buffer gets the tensor of its current TracedTensor, or the
        tensor a forward pass assigned. That tensor can still be tied to the
        capture until ``release()`` runs. It goes back through the model's own
        ``register_buffer()``, so an override or a buffer registration hook
        sees it. A non-persistent buffer stays non-persistent when that
        ``register_buffer()`` is torch's own.

        Args:
            at_session_end: Set by ``leapp.stop()`` and ``leapp.start()``. A
                registered buffer that a forward then set to None, deleted, or
                replaced with something else keeps that change, as in eager
                PyTorch, instead of getting its old tensor back. A buffer that
                still holds the TracedTensor ``module()`` put in for a tensor
                that requires grad gets the tensor of that TracedTensor.
            latest: False at the session end for a tracker that is not its
                node's latest one, and for every tracker at ``leapp.start()``.
        """
        # register_buffer() or a hook may have changed since inject().
        if not self.only_torch_code_runs():
            self._export_manager.keep_buffers_tied = True
        for full_name, info in self._injected.items():
            module = info.owning_module
            # At the session end only a node's latest tracker puts back a
            # buffer that requires grad, or a slot that is not a registered
            # buffer.
            # The record covers the TracedTensor module() put in, not a tensor
            # the forward replaced it with.
            slot = _slot(module, info.attr_name)
            requires_grad = any(slot is t for t in self._export_manager.grad_buffers)
            if at_session_end and not latest and (requires_grad or not info.was_buffer):
                continue
            if at_session_end and info.was_buffer and not requires_grad:
                buffers = module._buffers
                if info.attr_name not in buffers or buffers[info.attr_name] is None:
                    continue
                current = buffers[info.attr_name]
            else:
                current = getattr(module, info.attr_name)
                buffers = getattr(module, "__dict__", {}).get("_buffers")
                # A __getattr__ or a property can give back another tensor,
                # which then goes back in the buffer's place.
                if (
                    info.was_buffer
                    and isinstance(buffers, dict)
                    and info.attr_name in buffers
                    and buffers[info.attr_name] is not current
                ):
                    self._export_manager.keep_buffers_tied = True
            # Extract raw tensor from TracedTensor
            raw = current.tensor if hasattr(current, "tensor") else current
            if not isinstance(raw, torch.Tensor):
                raw = info.original_buffer
            _register_buffer(module, info.attr_name, raw)
            if info.was_buffer and not requires_grad:
                self._restored.append((module, info.attr_name))

        self._injected.clear()

    def only_torch_code_runs(self) -> bool:
        """Whether putting back or releasing its buffers runs only torch's code."""
        owners = [info.owning_module for info in self._injected.values()]
        owners += [module for module, _ in self._restored]
        return all(_only_torch_code_runs(owner) for owner in owners)

    def release(self) -> None:
        """Give each buffer that ``restore()`` put back a plain tensor.

        ``leapp.stop()`` and ``leapp.start()`` call it after ``restore()``,
        unless buffers must stay tied for the rest of the process. If
        such a buffer is a TracedTensor or a view of one, it gets a detached
        tensor on the same memory. A buffer that requires grad is left as it is,
        tied to the capture. Other buffers, buffers of an object that is not
        an ``nn.Module``, and other references to these tensors are not
        touched.
        """
        from leapp.leapp_graph.datatypes import is_traced_type

        restored, self._restored = self._restored, []
        for module, name in restored:
            # A later node may have deleted the buffer and used its name for
            # something else.
            if name not in module._buffers:
                continue
            value = module._buffers[name]
            if is_traced_type(value):
                value = value.tensor
            elif not isinstance(value, torch.Tensor) or not is_traced_type(value._base):
                continue
            if not value.requires_grad:
                # Outside inference mode, detach() would drop the inference
                # tensor's refusal of in-place updates.
                with torch.inference_mode(value.is_inference()):
                    plain = value.detach()
                _register_buffer(module, name, plain)
