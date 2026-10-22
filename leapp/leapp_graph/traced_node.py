#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections import deque
from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from typing import TYPE_CHECKING
import operator

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.datatypes import (
    TracedData,
    TracedTensor,
    as_traced,
    bind_shared_view,
    is_tracable_tensor_type,
    promote_in_place,
    is_traced_type,
    layout_key,
    may_adopt_view,
    to_export_torch_tensor,
)
from leapp.leapp_graph.datatypes.patching import get_warp_backend
from leapp.leapp_graph.datatypes.proxy_view import STORAGE_LAYOUT_META_KEY
from leapp.leapp_graph.aten_rewrites import rewrite_aten_ops
from leapp.leapp_graph.functionalization import functionalize_graph
from leapp.utils.tensor_description import (
    resolve_tensor_descriptions_to_names,
    flatten_io_structure,
)

# Importing registers the ``leapp::warp_runner`` custom op (import side effect)
# and exposes ``get_op``/``QUALIFIED_NAME`` used when emitting segment nodes.
from leapp.leapp_graph.custom_operator_registry import warp_operator
import collections
from dataclasses import replace

if TYPE_CHECKING:
    from leapp.leapp_graph.datatypes.warp import WarpSegment


# fx graph overloads to create custom behavior when interacting with the graph.
# keep local so that only the tracedTensorNode can create it.
class _LeappFXGraph(fx.Graph):
    """FX graph that notifies the Warp backend before graph mutation."""

    def __init__(self):
        super().__init__()
        self.recording = True
        self.requires_functionalization = False

    def create_node(self, *args, **kwargs):
        warp_backend = get_warp_backend()
        if warp_backend is not None:
            warp_backend.close_warp_segment()
        return super().create_node(*args, **kwargs)


class TracedTensorNode(LeappNode):
    def __init__(self, name, *args, **kwargs):
        if args or kwargs:
            _get_logger().warning(
                f"{name} received unexpected arguments on initialization. these arguments will be ignored."
            )
        super().__init__(name)
        self.graph = _LeappFXGraph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}
        self._discovered_warp_segments: list["WarpSegment"] = []
        self._pending_warp_segments: deque["WarpSegment"] = deque()
        self._is_warp_capture_active = False

        # Carriers declared on this pass, keyed on the bytes they cover, so a
        # value aliasing one of them can find it. A ProxyView holds a proxy
        # belonging to one graph, so this is rebuilt per pass.
        self._layout_index: dict[tuple, TracedData] = {}

        self.m = None

        # State tensor tracking: name -> {"input": TracedTensor, "output": TracedTensor | None}
        self._state_tensors: dict[str, dict] = {}

        # Buffer tracker for auto-detecting stateful buffers (set by annotate.module())
        self._buffer_tracker = None
        self._next_buffer_idx = 0

    def _get_required_io_description(self, name: str, io_list: list, io_kind: str):
        desc = self.get_io_description_by_name(name, io_list)
        if desc is None:
            _get_logger().fatal(
                f"State tensor '{name}' in node '{self.name}' is missing its {io_kind} description.",
                error_type=RuntimeError,
            )
        return desc

    @property
    def is_tracing(self) -> bool:
        return not self._model_captured

    @property
    def has_pending_warp_segments(self) -> bool:
        return bool(self._pending_warp_segments)

    @property
    def is_warp_capture_active(self) -> bool:
        return self._is_warp_capture_active

    @property
    def warp_segments(self) -> tuple["WarpSegment", ...]:
        return tuple(self._discovered_warp_segments)

    def prepare_warp_capture(self) -> None:
        if not self.has_pending_warp_segments or self.is_tracing:
            return
        self._is_warp_capture_active = True
        self.reset_trace_state()

    def acquire_warp_segment(self) -> "WarpSegment | None":
        return self._pending_warp_segments[0] if self._pending_warp_segments else None

    def add_warp_segment(self, segment: "WarpSegment") -> None:
        segment.runner_name = f"warp_segment_{len(self._discovered_warp_segments)}"
        self._discovered_warp_segments.append(segment)
        self._pending_warp_segments.append(segment)

    def complete_warp_segment(self, segment: "WarpSegment") -> None:
        if (
            not self._pending_warp_segments
            or self._pending_warp_segments[0] is not segment
        ):
            _get_logger().fatal(
                f"[{self.name}] Captured Warp regions out of discovery order.",
                error_type=RuntimeError,
            )
        if self.exports_model and segment.apic_graph is None:
            _get_logger().fatal(
                f"[{self.name}] Captured Warp segment has no APIC graph.",
                error_type=RuntimeError,
            )
        self._pending_warp_segments.popleft()

    def reset_trace_state(self) -> None:
        self.graph = _LeappFXGraph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}
        self.trimmed_inputs = set()
        self._next_buffer_idx = 0
        self.m = None
        self._model_captured = False
        self._layout_index.clear()
        # State placeholders belong to the graph being replaced, so a second
        # pass has to re-declare them rather than reuse the old carriers.
        self._state_tensors.clear()

    def find_declared_alias(self, value):
        """Carrier declared on this node that ``value`` may share a root with.

        This is how an alias created before any tracing session began attaches:
        the pair is invisible to the interception Phase 1 relies on, so the
        buffer is matched by layout instead. ``may_adopt_view`` re-checks the
        match and the sharing policy, so a stale index entry cannot promote an
        unrelated value into a shared root.
        """
        key = layout_key(value)
        if key is None:
            return None
        candidate = self._layout_index.get(key)
        if candidate is None or candidate is value:
            return None
        return candidate if may_adopt_view(candidate, value) else None

    def _validate_declared_input_aliases(self, outputs: dict) -> None:
        """Reject an output whose bytes a surviving declared input represents differently.

        This guards against a tensor in the graph sharing memory with another
        tensor without leapp knowing about it.

        Two roots over one allocation only diverge if the exported graph still
        reads both, so this runs after pruning. A declared input nothing
        consumed is trimmed, which leaves the output's root as the only
        representation of those bytes, and pruning an unused input is ordinary
        rather than an error.
        """
        for name, value in outputs.items():
            if not isinstance(value, TracedData):
                continue
            declared = self._layout_index.get(layout_key(value))
            if (
                declared is None
                or declared.proxy_view is value.proxy_view
                or (declared.proxy_view, value.proxy_view)
                in self._captured_view_aliases
            ):
                continue
            if self.get_io_description_by_name(declared.name, self.inputs) is None:
                continue
            _get_logger().fatal(
                f"Error: output '{name}' of node '{self.name}' shares memory "
                f"with declared input '{declared.name}', but they carry different "
                "tracing roots.\n"
                "Eager mutations through either alias would not be reflected in "
                "the other alias's exported graph. Use the carrier returned by "
                "input_tensors(), or avoid exposing both aliases at this node "
                "boundary.",
                error_type=Exception,
            )

    def compile_trace(
        self,
        tensors: dict[str, "TracedTensor"],
        backend=None,
        backend_params={},
        static_tensors: dict[str, torch.Tensor] | None = None,
        semantics_map: dict[str, object] | None = None,
        static_semantics_map: dict[str, object] | None = None,
    ):
        # Auto-collect buffer mutations before merging state outputs.
        # collect() may remove non-mutated buffers from _state_tensors and
        # bake them as constants, so it must run before get_state_outputs().
        if self._buffer_tracker is not None and not self._buffer_tracker._collected:
            self._buffer_tracker.collect()
            self._buffer_tracker.restore()

        # Merge state outputs into the tensor dict before processing
        state_outputs = self.get_state_outputs()
        if state_outputs:
            _get_logger().info(
                f"Adding {len(state_outputs)} state outputs: {list(state_outputs.keys())}"
            )
            tensors = {**tensors, **state_outputs}

        # Held for the post-pruning alias check below. Static outputs join
        # `tensors` further down but are frozen into buffers at trace time, so
        # the graph never reads the bytes a declared input could alias.
        traced_outputs = dict(tensors)

        unwrapped_tensors = []
        for name, tensor in tensors.items():
            unwrapped = self.create_output(
                tensor, name, semantics=(semantics_map or {}).get(name)
            )
            unwrapped_tensors.append(unwrapped)
            if name in self._state_tensors:
                self._state_tensors[name]["output_desc"] = (
                    self._get_required_io_description(name, self.outputs, "output")
                )

        # Static tensors go through the same create_output path with static=True
        if static_tensors:
            for name, tensor in static_tensors.items():
                wrapped = self.create_output(
                    tensor,
                    name,
                    static=True,
                    semantics=(static_semantics_map or {}).get(name),
                )
                tensors[name] = wrapped

        self.build_graph_module(list(tensors.values()))
        self._validate_declared_input_aliases(traced_outputs)

        self.m = fx.GraphModule(self.tracer.root, self.graph)

        # Sanitize the FX graph for compatibility with export backends.
        rewrite_aten_ops(self.m.graph)
        self._rewrite_method_descriptors(self.m.graph)
        self._make_tensor_attrs_contiguous(self.m)
        self.m.recompile()

        _get_logger().debug(f"Compiled graph module for {self.name}:\n{self.m.graph}")
        _get_logger().debug(
            f"Graph module inputs: {[resolve_tensor_descriptions_to_names(input) for input in self.input_formats]}"
        )
        _get_logger().debug(
            f"Graph module outputs: {[resolve_tensor_descriptions_to_names(output) for output in self.output_formats]}"
        )
        self.setup_backend(backend, backend_params)

        return (
            unwrapped_tensors[0]
            if len(unwrapped_tensors) == 1
            else tuple(unwrapped_tensors)
        )

    def state_feedback_pairs(self):
        """State tensors are an eager form of feedback.

        The state output from one execution is fed back into the matching state
        input on the next, which the user declared by pairing the two names, so
        a pair only becomes an edge once both halves exist.
        """
        return tuple(
            (state_info["input_desc"], state_info["output_desc"])
            for state_info in self._state_tensors.values()
            if state_info.get("input_desc") is not None
            and state_info.get("output_desc") is not None
        )

    @staticmethod
    def _rewrite_method_descriptors(graph: fx.Graph):
        """Convert tensor-method call_function nodes to call_method.

        Method descriptors (e.g. torch.Tensor.view, torch.Tensor.float) don't
        support weak references, which causes torch.jit.script to fail with:
            TypeError: cannot create weak reference to 'method_descriptor' object

        Recent PyTorch versions also surface some tensor methods (for example
        ``torch.Tensor.norm``) as plain functions from ``torch._tensor``. If
        those targets remain as call_function nodes, FX may serialize them as
        ``torch._tensor.*`` calls that TorchScript cannot resolve at runtime.

        Converting both representations to call_method nodes uses a string
        target instead, sidestepping the issue entirely. The args layout is
        identical: args[0] is already ``self`` for unbound tensor methods,
        which is exactly what call_method expects.
        """
        rewritten = []
        for node in graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            target_type_name = type(target).__name__
            target_module = getattr(target, "__module__", "")
            is_tensor_method_target = (
                target_type_name == "method_descriptor"
                or target_module.startswith("torch._tensor")
            )
            if not is_tensor_method_target:
                continue

            method_name = getattr(target, "__name__", None)
            if method_name is not None:
                node.op = "call_method"
                node.target = method_name
                rewritten.append(method_name)
        if rewritten:
            _get_logger().debug(
                f"Rewrote {len(rewritten)} tensor-method call_function node(s) "
                f"to call_method: {rewritten}"
            )
        graph.lint()

    @staticmethod
    def _make_tensor_attrs_contiguous(graph_module: fx.GraphModule):
        """Make all tensor constant attributes contiguous.

        make_fx decomposition of nn.Linear produces addmm(bias, x, weight.T)
        where weight.T is a non-contiguous transposed view.  Serialization
        backends (ONNX, TorchScript) may not respect non-standard strides,
        resulting in corrupted weight data in the exported file.

        FX GraphModule moves tensor constants into registered buffers during
        construction, so we must check named_buffers() (not just vars()).
        """
        count = 0
        for name in list(vars(graph_module)):
            attr = getattr(graph_module, name)
            if isinstance(attr, torch.Tensor) and not attr.is_contiguous():
                setattr(graph_module, name, attr.contiguous())
                count += 1
        for name, buf in list(graph_module.named_buffers()):
            if not buf.is_contiguous():
                graph_module.register_buffer(name, buf.contiguous())
                count += 1
        for name, param in list(graph_module.named_parameters()):
            if not param.is_contiguous():
                graph_module.register_parameter(
                    name,
                    torch.nn.Parameter(
                        param.contiguous(), requires_grad=param.requires_grad
                    ),
                )
                count += 1
        if count:
            _get_logger().debug(
                f"Made {count} non-contiguous tensor constant(s) contiguous"
            )

    def _create_io_helper(self, data, name: str, to: str):
        if isinstance(data, collections.abc.Mapping):
            new_data = {}
            for key, value in data.items():
                child_name = "_".join([name, key]) if name else key
                new_data[key] = self._create_io_helper(value, child_name, to)
            return new_data
        elif isinstance(data, (list, tuple)):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)]) if name else str(idx)
                new_data.append(self._create_io_helper(value, child_name, to))
            return type(data)(new_data)

        elif is_tracable_tensor_type(data):
            is_traced = isinstance(data, TracedData)
            is_active_traced = is_traced and data.is_tracing
            if is_active_traced:
                # The value is actively tracing in a node context.
                # Check if the traced tensor is from the same context (node)
                if data.context_obj is not self:
                    # Different context: error - cannot use traced tensor from another node
                    _get_logger().fatal(
                        f"Error: when creating inputs for '{self.name}', "
                        f"detected data '{name}' is an active {data.__class__.__name__} from a different node '{data.context}'. \n"
                        f"Mixing active contexts is not allowed. "
                        f"Call output_tensors() on the source node first.",
                        error_type=Exception,
                    )

            if to == "traced":
                # A root that is a placeholder means the value entered its node
                # as a declaration rather than being computed there. Declaring
                # promotes the caller's tensor in place, so one external tensor
                # reaching two nodes arrives here still carrying the first
                # node's placeholder, which is an ordinary shared input and not
                # a missing edge between the two.
                entered_by_declaration = (
                    is_traced
                    and getattr(getattr(data.proxy, "node", None), "op", None)
                    == "placeholder"
                )

                if is_active_traced:
                    # Same context: allow override with warning
                    _get_logger().warning(
                        f"Input '{name}' for node '{self.name}' is an active TracedTensor "
                        f"from the same node. Creating fresh input placeholder "
                        f"(previous trace will be discarded for this branch)."
                    )
                elif (
                    self.is_tracing
                    and is_traced
                    and data.context_obj is not self
                    and data.output_port is None
                    and not entered_by_declaration
                ):
                    # Came out of a previous node but was never registered as one
                    # of that node's outputs, so there is no edge to connect to.
                    # Only the first pass can say this, because a re-entry pass
                    # rebuilds its values without ports and reuses the sources
                    # its descriptions already hold.
                    # This can be deliberate, so keep going and treat it as a
                    # dangling graph input.
                    _get_logger().error(
                        f"Error: input '{name}' for node '{self.name}' was derived from "
                        f"node '{data.context}' but is not one of its registered outputs.\n"
                        f"Add it to that node's output_tensors() to connect the two nodes, "
                        f"or ignore this if '{name}' is meant to enter the graph from outside.\n"
                        f"Treating '{name}' as a dangling input."
                    )

                proxy = None
                if self.is_tracing:
                    node = self.graph.create_node("placeholder", name, (), {})
                    proxy = Proxy(node, self.tracer)
                if is_active_traced or type(data) is torch.Tensor:
                    # Declaring binds the object the caller holds rather than
                    # handing back a second one beside it: a plain tensor is
                    # promoted in place, and a live carrier of this node is
                    # rebound onto the new placeholder so it cannot keep
                    # pointing at a graph this declaration just replaced.
                    traced = promote_in_place(data, name, self, proxy)
                else:
                    traced = as_traced(data, name, self, proxy)
                if self.is_tracing:
                    # Declaring a buffer is what lets a persistent alias of it
                    # find this carrier later, so registration is the API and no
                    # separate one is needed.
                    key = layout_key(traced)
                    if key is not None:
                        self._layout_index.setdefault(key, traced)
                return traced

            elif to == "tensor":
                if not is_traced:
                    _get_logger().warning(
                        f"Warning: when creating outputs for {self.name}, "
                        f"detected data {name} is not a TracedTensor"
                    )
                return data

            elif to == "static":
                if is_traced:
                    _get_logger().fatal(
                        f"Cannot create static output from traced "
                        f"{type(data).__name__} '{name}'. "
                        "Static outputs must be raw tensors.",
                        error_type=Exception,
                    )

                # Create unique attribute name and store on root module
                attr_name = f"_static_{name}".replace(".", "_")
                self.tracer.root.register_buffer(
                    attr_name, to_export_torch_tensor(data).clone()
                )
                # Create get_attr node that retrieves the stored constant
                node = self.graph.create_node("get_attr", attr_name, (), {})
                proxy = Proxy(node, self.tracer)
                return as_traced(data, name, self, proxy)
        else:
            if to == "traced":
                _get_logger().warning(
                    f"Non-tracable input '{name}' (type={type(data).__name__}) in node '{self.name}' "
                    f"will be passed through as a constant."
                )
            return data

    def create_input(self, data, name: str, semantics=None) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            data: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        # An active carrier of *another* node falls through to the identity fatal
        # in _create_io_helper, and a published value keeps a port of its own so
        # the edge carrying it into this node survives. Everything else may share
        # a root with a buffer this node already declared.
        foreign_active = (
            getattr(data, "is_tracing", False)
            and getattr(data, "context_obj", None) is not self
        )
        adoptable = getattr(data, "output_port", None) is None and not foreign_active
        declared = self.find_declared_alias(data) if adoptable else None
        if declared is not None:
            _get_logger().info(
                f"Input '{name}' for node '{self.name}' shares memory with declared "
                f"input '{declared.name}'. Both names describe one graph value, so "
                f"'{declared.name}' will be the only port in the exported node interface."
            )
            # Adoption skips add_input, which is where semantics are normally
            # attached, so remember the alias and preserve any semantics the
            # surviving port does not already have.
            surviving = self.get_io_description_by_name(declared.name, self.inputs)
            if name not in surviving.aliases:
                surviving.aliases.append(name)
            if semantics is not None and surviving.semantics is None:
                surviving.init_semantics(replace(semantics, name=None))
            if isinstance(data, TracedData):
                # Rebind the object the caller holds rather than returning a new
                # carrier. A buffer promoted in place stays the same object on
                # every pass, so handing back a fresh carrier would leave the
                # caller's own value on the previous pass's discarded graph.
                bind_shared_view(data, declared.name, self, declared.proxy_view)
                return data
            return as_traced(data, declared.name, self, view=declared.proxy_view)

        existing = self.get_io_description_by_name(name, self.inputs)
        has_placeholder = any(
            node.op == "placeholder" and node.target == name
            for node in self.graph.nodes
        )
        if existing is not None and not has_placeholder:
            self.validate_input_and_update_sources(name, name, data)
            return self._create_io_helper(data, name, to="traced")
        self.add_input(name, name, data, semantics=semantics)
        traced_data = self._create_io_helper(data, name, to="traced")
        return traced_data

    def create_output(self, data, name: str, static: bool = False, semantics=None):
        existing = self.get_io_description_by_name(name, self.outputs)
        if static:
            self._validate_static_tensor(data, name)
            wrapped = self._create_io_helper(data, name, to="static")
            if existing is not None:
                self.publish_output_port(wrapped, existing.port)
                self.validate_output_and_update_sources(name, name, data)
                return wrapped
            descriptions = self.add_output(name, name, data, semantics=semantics)
            self.publish_output_ports(wrapped, name, descriptions)
            return wrapped
        else:
            unwrapped_data = self._create_io_helper(data, name, to="tensor")
            if existing is not None:
                self.publish_output_port(unwrapped_data, existing.port)
                self.validate_output_and_update_sources(name, name, unwrapped_data)
                return unwrapped_data
            descriptions = self.add_output(
                name, name, unwrapped_data, semantics=semantics
            )
            self.publish_output_ports(unwrapped_data, name, descriptions)
            return unwrapped_data

    def _validate_static_tensor(self, tensor, name: str):
        if not is_tracable_tensor_type(tensor):
            _get_logger().fatal(
                f"Error: static output '{name}' has type {type(tensor).__name__} "
                "but expected a raw tracable data type.\n"
                "**Static outputs must be raw tensors, not derived from input tensors.**\n"
                "If this value depends on inputs, use it as a regular output tensor instead.",
                error_type=Exception,
            )
        if is_traced_type(tensor):
            _get_logger().fatal(
                f"Error: static output '{name}' is a traced "
                f"{type(tensor).__name__}. "
                "Static outputs should be constant tensors, not traced computations.",
                error_type=Exception,
            )

    def create_static_tensors(self, static_outputs):
        """Wrap raw tensors as static graph nodes (for register_buffer).

        Unlike create_output(static=True), this does NOT publish an output port
        or register outputs — it only validates and wraps.
        Returns data in the same nested structure as the input payload.
        """
        flattened_static_outputs = flatten_io_structure(static_outputs, "")
        for tensor_name, tensor in flattened_static_outputs.items():
            self._validate_static_tensor(tensor, tensor_name)

        return self._create_io_helper(static_outputs, "", to="static")

    def build_graph_module(self, outputs: list["TracedTensor"]) -> fx.GraphModule:
        """Convert the traced computation to a torch.fx.GraphModule.

        Args:
            outputs: List of TracedTensor outputs to include in the graph module

        Returns:
            fx.GraphModule: A graph module that takes all inputs and returns all outputs
        """
        # Find all nodes that are actually used by traversing backwards from outputs
        output_nodes = [output.proxy.node for output in outputs]
        self._captured_view_aliases = set()
        for output in outputs:
            declared = self._layout_index.get(layout_key(output))
            if declared is None:
                continue
            pending, seen = [output.proxy.node], set()
            while pending:
                node = pending.pop()
                if node is declared.proxy.node:
                    self._captured_view_aliases.add(
                        (declared.proxy_view, output.proxy_view)
                    )
                    break
                if node not in seen:
                    seen.add(node)
                    pending.extend(node.all_input_nodes)
        for node in self.graph.nodes:
            node.meta.pop(STORAGE_LAYOUT_META_KEY, None)
        self.graph.recording = False
        if self.graph.requires_functionalization:
            # A write through a view can affect an output without an explicit
            # data edge yet. Keep writes and their inputs until normalized,
            # while pruning unused work and reporting required capture errors
            # through the same LEAPP checks as the final graph.
            self._prune_graph(output_nodes, preserve_mutations=True)
            self.graph.output(
                output_nodes[0] if len(output_nodes) == 1 else tuple(output_nodes)
            )
            self.tracer.root = functionalize_graph(
                self.graph,
                self.tracer.root,
                [to_export_torch_tensor(desc.value) for desc in self.inputs],
            )
            output_node = next(n for n in self.graph.nodes if n.op == "output")
            output_nodes = list(output_node.all_input_nodes)

        self._prune_graph(
            output_nodes,
            preserve_mutations=getattr(self.graph, "preserve_mutations", False),
        )

        # After pruning, record which Warp-segment outputs survived (are actually
        # used) as a binary mask on the segment marker node's metadata.
        self._stamp_warp_used_output_masks()

        # Check if graph already has an output node
        has_output = any(node.op == "output" for node in self.graph.nodes)

        if not has_output:
            # If single output, return it directly; if multiple, return as tuple
            if len(output_nodes) == 1:
                self.graph.output(output_nodes[0])
            else:
                self.graph.output(tuple(output_nodes))

    def _prune_graph(
        self, output_nodes: list[fx.Node], *, preserve_mutations: bool = False
    ) -> None:
        """Prune unused computation, validate capture, and update the node inputs."""
        used_nodes = set()

        def mark_used(node):
            """Recursively mark all nodes used in computing this node."""
            if node in used_nodes:
                return
            used_nodes.add(node)
            # Traverse all input nodes
            for arg in node.all_input_nodes:
                mark_used(arg)

        # Mark all nodes used by any output
        for output_node in output_nodes:
            mark_used(output_node)

        if preserve_mutations:
            for node in self.graph.nodes:
                if node.meta.get("leapp_mutation", False) or (
                    node.op.startswith("call_") and node.is_impure()
                ):
                    mark_used(node)

        # Capture can fail in work unrelated to the declared outputs. Only a
        # failure that survives the output dependency traversal blocks export.
        for node in self.graph.nodes:
            if node in used_nodes and (error := node.meta.get("leapp_capture_error")):
                _get_logger().fatal(
                    f"Cannot export node '{self.name}': a declared output depends on "
                    f"a failed capture. {error}",
                    error_type=NotImplementedError,
                )

        # Remove unused nodes (both placeholders and call_function nodes)
        nodes_to_remove = []
        for node in self.graph.nodes:
            # Skip output nodes
            if node.op == "output":
                continue
            # Remove any node not used in computing outputs
            if node not in used_nodes:
                nodes_to_remove.append(node)

        # Erase nodes in reverse order to avoid issues with dependencies
        trimmed_inputs = set()
        for node in reversed(nodes_to_remove):
            if node.op == "placeholder":
                input_description = LeappNode.get_io_description_by_name(
                    node.target, self.inputs
                )
                if input_description is None:
                    _get_logger().error(
                        f"Error: when building the graph module for {self.name}, "
                        f"could not find input description for placeholder '{node.target}'"
                    )
                    continue
                trimmed_inputs.add(input_description.name_str)

            if len(node.users) == 0:
                self.graph.erase_node(node)

        if trimmed_inputs:
            self.trimmed_inputs.update(trimmed_inputs)
            _get_logger().warning(
                f"Warning: when building the graph module for {self.name}, "
                "detected the following inputs are not used in the computation or directly returned as output: "
                f"{trimmed_inputs} \n"
                "For clarity and efficiency, consider removing these inputs"
            )

            # Remove trimmed inputs from self.inputs
            self.inputs = [
                inp for inp in self.inputs if inp.name_str not in trimmed_inputs
            ]

    def _stamp_warp_used_output_masks(self) -> None:
        """Compute runner output masks from the live graph after normalization.

        Export can replace nodes and discard LEAPP's accessor metadata. The
        runner's serialized output specs and positional getitems remain its
        contract, so derive masks from those rather than stale segment proxies.
        """
        op = warp_operator.get_op()
        for node in self.graph.nodes:
            if node.op != "call_function" or node.target not in (op, op.default):
                continue
            metadata = warp_operator.decode_runtime_metadata(node.args[1])
            outputs = metadata["outputs"]
            used = set()
            for user in node.users:
                if user.target is operator.getitem and isinstance(user.args[1], int):
                    used.add(user.args[1])
                else:
                    used.update(range(len(outputs)))
            for index, spec in enumerate(outputs):
                spec["mask"] = index in used
                if index not in used:
                    spec["shape"] = [0]
                    spec["num_bytes"] = 0
            node.update_arg(1, warp_operator.encode_runtime_metadata(metadata))

    def create_state_tensors(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, TracedTensor]:
        """Create state tensors that are both inputs and outputs."""
        result = {}
        for name, tensor in tensors.items():
            if name in self._state_tensors:
                _get_logger().warning(
                    f"State tensor '{name}' already registered for node '{self.name}'. "
                    "Returning existing TracedTensor."
                )
                result[name] = self._state_tensors[name]["input"]
                continue

            # Create input placeholder for this state
            traced_input = self.create_input(tensor, name)

            # Track as state tensor
            self._state_tensors[name] = {
                "input": traced_input,
                "input_desc": self._get_required_io_description(
                    name, self.inputs, "input"
                ),
                "output": None,  # Set via update_state_tensors()
                "output_desc": None,
            }

            result[name] = traced_input

        return result

    def update_state_tensors(self, tensors: dict[str, TracedTensor]) -> None:
        """Set output values for state tensors."""
        for name, value in tensors.items():
            if name not in self._state_tensors:
                _get_logger().fatal(
                    f"Error: update_state called for '{name}' but it was not registered "
                    f"as a state tensor. Call state_tensors() first.",
                    error_type=Exception,
                )

            # Validate shape and dtype match the input state
            input_tensor = self._state_tensors[name]["input"]
            input_underlying = (
                input_tensor.tensor
                if isinstance(input_tensor, TracedData)
                else input_tensor
            )
            value_underlying = value.tensor if isinstance(value, TracedData) else value
            if input_underlying.shape != value_underlying.shape:
                _get_logger().fatal(
                    f"Error: update_state for '{name}' has shape {value_underlying.shape} "
                    f"but expected {input_underlying.shape} (must match input state shape).",
                    error_type=Exception,
                )
            if input_underlying.dtype != value_underlying.dtype:
                _get_logger().fatal(
                    f"Error: update_state for '{name}' has dtype {value_underlying.dtype} "
                    f"but expected {input_underlying.dtype} (must match input state dtype).",
                    error_type=Exception,
                )

            self._state_tensors[name]["output"] = value

    def get_state_outputs(self) -> dict[str, TracedTensor]:
        """Get state outputs using the original state names.

        Only states explicitly updated via update_state() become feedback outputs.
        States declared without an update are treated as regular inputs.
        """
        inactive_states = [
            name for name, info in self._state_tensors.items() if info["output"] is None
        ]
        if inactive_states:
            _get_logger().warning(
                f"State tensors {inactive_states} in node '{self.name}' were not updated via "
                "update_state(). They will be treated as regular inputs and will not create feedback outputs."
            )
            for name in inactive_states:
                self._state_tensors.pop(name, None)

        result = {}
        for name, state_info in self._state_tensors.items():
            result[name] = state_info["output"]
        return result

    def _create_warp_bundle_proxy(self, wrp_archive: bytes, runner_name: str) -> Proxy:
        """Wire a pre-packed WRPB archive as a ``get_attr`` FX node."""
        buffer_name = f"_{runner_name}_bundle"
        bundle_tensor = torch.frombuffer(
            bytearray(wrp_archive), dtype=torch.uint8
        ).clone()
        self.tracer.root.register_buffer(buffer_name, bundle_tensor, persistent=True)
        bundle_node = self.graph.create_node(
            "get_attr",
            buffer_name,
            (),
            {},
            name=f"{runner_name}_bundle",
        )
        return Proxy(bundle_node, self.tracer)

    def create_warp_proxy(
        self,
        encoded_metadata: str,
        input_proxies: list[Proxy],
        wrp_archive: bytes,
        output_count: int,
        runner_name: str,
    ) -> tuple[Proxy, list[Proxy]]:
        """Create the FX proxy nodes for a Warp runner op.

        The caller owns Warp segment semantics and node metadata. This method
        only mutates the FX graph: bundle ``get_attr``, one ``leapp::warp_runner``
        call, plus one positional ``operator.getitem`` per candidate output.
        """
        bundle_proxy = self._create_warp_bundle_proxy(wrp_archive, runner_name)

        # The op consumes only the segment's traced inputs (as a Tensor[]) and
        # *produces* its outputs via per-output ``operator.getitem``. Segment
        # outputs must never be fed back in as op inputs, otherwise the FX graph
        # shows the results as get_attr constants flowing into the call instead
        # of being derived from it.
        #
        # Emit the ``.default`` OpOverload: ``torch.export`` (dynamo) consumes the
        # overload directly and the dynamo ONNX path lowers it to WrpRunner.
        warp_runner = self.tracer.create_proxy(
            "call_function",
            warp_operator.get_op().default,
            ([*input_proxies], encoded_metadata, bundle_proxy),
            {},
            name=runner_name,
        )

        # The op returns a Tensor[]; extract each output positionally so index i
        # always refers to the segment's i-th output, regardless of which
        # survive pruning. Only consumed outputs keep their getitem.
        output_proxies = []
        for idx in range(output_count):
            output_proxies.append(
                self.tracer.create_proxy(
                    "call_function",
                    operator.getitem,
                    (warp_runner, idx),
                    {},
                    name=f"{runner_name}_output_{idx}",
                )
            )

        return warp_runner, output_proxies

    @property
    def state_names(self) -> list[str]:
        """Get list of registered state tensor names."""
        return list(self._state_tensors.keys())
