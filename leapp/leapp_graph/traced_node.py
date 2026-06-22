#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from dataclasses import dataclass, field
from typing import Any, Literal
import operator

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.datatypes import (
    TracedData,
    TracedTensor,
    as_traced,
    is_tracable_tensor_type,
)
from leapp.utils.tensor_description import (
    resolve_tensor_descriptions_to_names,
    flatten_io_structure,
)
import collections


def warp_segment_marker(*inputs):
    """FX marker for one finalized Warp/APIC segment."""
    raise RuntimeError(
        "leapp_warp_segment is an FX marker and should be lowered before execution"
    )


@dataclass
class WarpTensorRef:
    # Segment-local canonical name used for APIC params / FX output labels.
    name: str
    # Live trace-time object, usually a wp.array, kept for capture/replay work.
    array: Any
    # FX proxy that represents this value in the owning TracedTensorNode graph.
    proxy: Proxy | None = None
    # Owning LEAPP trace context, normally the TracedTensorNode instance.
    context: Any | None = None
    # Detector path showing where the value was found, e.g. args[0] or kwargs['out'].
    path: str | None = None
    # Runtime array shape observed during tracing, used for validation/export metadata.
    shape: tuple | None = None
    # Runtime dtype observed during tracing, stored as text for lightweight metadata.
    dtype: str | None = None
    # Runtime device observed during tracing, stored as text for lightweight metadata.
    device: str | None = None
    # Device/host pointer when available; helps dedupe view-like wp.array objects.
    ptr: int | None = None
    # True when this ref is a runtime input to the Warp segment.
    is_input: bool = False
    # True when this ref is a runtime output of the segment.
    is_output: bool = False
    # Index of the detector event that first produced/wrote this ref, when known.
    produced_by_event_index: int | None = None
    # Extra detector/export annotations that do not deserve first-class fields yet.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        path: str | None = None,
        is_input: bool = False,
        is_output: bool = False,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WarpTensorRef":
        ptr = getattr(value, "ptr", None)
        try:
            ptr = int(ptr) if ptr else None
        except Exception:
            ptr = None

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                shape = tuple(shape)
            except TypeError:
                pass

        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)

        return cls(
            name=name,
            array=value,
            proxy=getattr(value, "proxy", None),
            context=getattr(value, "context_obj", None),
            path=path,
            shape=shape,
            dtype=str(dtype) if dtype is not None else None,
            device=str(device) if device is not None else None,
            ptr=ptr,
            is_input=is_input,
            is_output=is_output,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata or {},
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "ptr": self.ptr,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "produced_by_event_index": self.produced_by_event_index,
            "metadata": self.metadata,
        }


@dataclass
class WarpSegment:
    # Owning LEAPP node name. A segment should not span multiple node graphs.
    node_name: str
    # Lifecycle state; invalid segments fail closed instead of silently exporting.
    status: Literal["open", "closed", "invalid"] = "open"
    # Detector-recorded top-level Warp events/calls that make up the segment.
    events: list[Any] = field(default_factory=list)
    # Runtime APIC/FX inputs, keyed by segment-local canonical name.
    input_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # Runtime APIC/FX outputs that will get marker-derived proxies.
    output_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # Conservative possible outputs seen by the detector, not yet confirmed.
    output_candidates: list[WarpTensorRef] = field(default_factory=list)
    # FX proxy for the single segment marker node.
    marker_proxy: Proxy | None = None
    # Per-output FX proxies derived from marker_proxy.
    output_proxies: dict[str, Proxy] = field(default_factory=dict)
    # Per-output get_attr proxies that anchor concrete output objects in FX.
    output_attr_proxies: dict[str, Proxy] = field(default_factory=dict)
    # Extra segment annotations such as capture strategy or detector details.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Warp device used for capture/replay when known.
    device: str | None = None
    # Saved .wrp path once export writes the APIC bundle.
    wrp_path: str | None = None
    # Live APIC graph object during trace/export; intentionally not serialized.
    apic_graph: Any | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def is_empty(self) -> bool:
        return not (
            self.events
            or self.input_refs
            or self.output_refs
            or self.output_candidates
        )

    def add_event(self, event: Any) -> None:
        self.events.append(event)

    def add_input_ref(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("input", len(self.input_refs)),
            path=path,
            is_input=True,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.input_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref, is_input=True)

        self.input_refs[ref.name] = ref
        return ref

    def add_output_ref(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("output", len(self.output_refs)),
            path=path,
            is_output=True,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.output_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref, is_output=True)

        self.output_refs[ref.name] = ref
        return ref

    def add_output_candidate(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name(
                "output_candidate", len(self.output_candidates)
            ),
            path=path,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.output_candidates)
        if existing is not None:
            return self._merge_ref(existing, ref)

        self.output_candidates.append(ref)
        return ref

    def _coerce_ref(
        self,
        value: Any,
        *,
        name: str,
        path: str | None = None,
        is_input: bool = False,
        is_output: bool = False,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        if isinstance(value, WarpTensorRef):
            ref = value
            ref.name = name
            if path is not None:
                ref.path = path
            if metadata:
                ref.metadata.update(metadata)
            ref.is_input = ref.is_input or is_input
            ref.is_output = ref.is_output or is_output
            if produced_by_event_index is not None:
                ref.produced_by_event_index = produced_by_event_index
            return ref

        return WarpTensorRef.from_value(
            name,
            value,
            path=path,
            is_input=is_input,
            is_output=is_output,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )

    @staticmethod
    def _default_ref_name(prefix: str, index: int) -> str:
        return f"{prefix}_{index}"

    @staticmethod
    def _ref_key(ref: WarpTensorRef) -> tuple[int, int | None]:
        return (id(ref.array), ref.ptr)

    def _find_ref(
        self, ref: WarpTensorRef, existing_refs: collections.abc.Iterable
    ) -> WarpTensorRef | None:
        ref_key = self._ref_key(ref)
        for existing in existing_refs:
            if self._ref_key(existing) == ref_key:
                return existing
        return None

    @staticmethod
    def _merge_ref(
        existing: WarpTensorRef,
        incoming: WarpTensorRef,
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> WarpTensorRef:
        existing.is_input = existing.is_input or is_input or incoming.is_input
        existing.is_output = existing.is_output or is_output or incoming.is_output
        if existing.path is None:
            existing.path = incoming.path
        if existing.proxy is None:
            existing.proxy = incoming.proxy
        if existing.context is None:
            existing.context = incoming.context
        if existing.produced_by_event_index is None:
            existing.produced_by_event_index = incoming.produced_by_event_index
        existing.metadata.update(incoming.metadata)
        return existing

    def invalidate(self) -> None:
        self.status = "invalid"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "status": self.status,
            "device": self.device,
            "wrp_path": self.wrp_path,
            "has_apic_graph": self.apic_graph is not None,
            "inputs": [ref.to_metadata() for ref in self.input_refs.values()],
            "outputs": [ref.to_metadata() for ref in self.output_refs.values()],
            "output_candidates": [
                ref.to_metadata() for ref in self.output_candidates
            ],
            "event_count": len(self.events),
            "metadata": self.metadata,
        }


class TracedTensorNode(LeappNode):
    def __init__(self, name, *args, dry_run=False, **kwargs):
        if args or kwargs:
            _get_logger().warning(f"{name} received unexpected arguments on initialization. these arguments will be ignored.")
        super().__init__(name, dry_run=dry_run)
        self.graph = fx.Graph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}
        self.warp_segments: list[WarpSegment] = []

        self.m = None

        # State tensor tracking: name -> {"input": TracedTensor, "output": TracedTensor | None}
        self._state_tensors: dict[str, dict] = {}

        # Buffer tracker for auto-detecting stateful buffers (set by annotate.module())
        self._buffer_tracker = None

        self._next_buffer_idx = 0

    def _get_required_io_description(self, name: str, io_list: list, io_kind: str):
        desc = self.get_io_description_by_name(name, io_list)
        if desc is None:
            raise RuntimeError(
                f"State tensor '{name}' in node '{self.name}' is missing its {io_kind} description."
            )
        return desc

    @property
    def is_tracing(self) -> bool:
        return not self._model_captured

    def compile_trace(self, tensors: dict[str, "TracedTensor"], backend=None, backend_params={},
                       static_tensors: dict[str, torch.Tensor] | None = None,
                       semantics_map: dict[str, object] | None = None,
                       static_semantics_map: dict[str, object] | None = None):
        # Auto-collect buffer mutations before merging state outputs.
        # collect() may remove non-mutated buffers from _state_tensors and
        # bake them as constants, so it must run before get_state_outputs().
        if self._buffer_tracker is not None and not self._buffer_tracker._collected:
            self._buffer_tracker.collect()
            self._buffer_tracker.restore()

        # Merge state outputs into the tensor dict before processing
        state_outputs = self.get_state_outputs()
        if state_outputs:
            _get_logger().info(f"Adding {len(state_outputs)} state outputs: {list(state_outputs.keys())}")
            tensors = {**tensors, **state_outputs}

        if self.dry_run:
            for name, tensor in tensors.items():
                self.tag_data(tensor, name)
                self.add_output(name, name, tensor, semantics=(semantics_map or {}).get(name))
                if name in self._state_tensors:
                    self._state_tensors[name]["output_desc"] = self._get_required_io_description(
                        name, self.outputs, "output"
                    )
            if static_tensors:
                for name, tensor in static_tensors.items():
                    self.tag_data(tensor, name)
                    self.add_output(name, name, tensor, semantics=(static_semantics_map or {}).get(name))
            self._apply_state_tags()
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        unwrapped_tensors = []
        for name, tensor in tensors.items():
            unwrapped = self.create_output(tensor, name, semantics=(semantics_map or {}).get(name))
            unwrapped_tensors.append(unwrapped)
            if name in self._state_tensors:
                self._state_tensors[name]["output_desc"] = self._get_required_io_description(
                    name, self.outputs, "output"
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

        # Apply state tags after buffer collection (which may remove placeholder
        # nodes for non-mutated buffers) but before build_graph_module.
        self._apply_state_tags()

        self.build_graph_module(list(tensors.values()))

        self.m = fx.GraphModule(
            self.tracer.root, self.graph)

        # Sanitize the FX graph for compatibility with export backends.
        self._rewrite_aten_ops(self.m.graph)
        self._rewrite_method_descriptors(self.m.graph)
        self._make_tensor_attrs_contiguous(self.m)
        self.m.recompile()

        _get_logger().debug(
            f"Compiled graph module for {self.name}:\n{self.m.graph}")
        _get_logger().debug(
            f"Graph module inputs: {[resolve_tensor_descriptions_to_names(input) for input in self.input_formats]}")
        _get_logger().debug(
            f"Graph module outputs: {[resolve_tensor_descriptions_to_names(output) for output in self.output_formats]}")
        self.setup_backend(backend, backend_params)

        return unwrapped_tensors[0] if len(unwrapped_tensors) == 1 else tuple(unwrapped_tensors)

    def _apply_state_tags(self):
        """Mirror each registered state output tag onto its paired state input.

        State tensors are an eager form of feedback: the state output from one
        execution is guaranteed to be fed back into the matching state input on
        the next execution. By copying the already-registered output tag onto
        the input description, both ends of that feedback edge share one
        internal identity without inferring a second "canonical" tag name.
        """
        for state_name in self._state_tensors:
            input_desc = self._state_tensors[state_name].get("input_desc")
            output_desc = self._state_tensors[state_name].get("output_desc")
            if input_desc is None or output_desc is None or output_desc.tag is None:
                continue

            input_desc.tag = output_desc.tag

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
                target_type_name == "method_descriptor" or
                target_module.startswith("torch._tensor")
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
                f"to call_method: {rewritten}")
        graph.lint()

    @staticmethod
    def _rewrite_aten_ops(graph: fx.Graph):
        """Replace low-level ``torch.ops.aten.*`` OpOverload targets with their
        high-level ``torch.*`` or ``torch.nn.functional.*`` equivalents.

        When a TorchScript model is decomposed via ``TS2EPConverter``, the
        resulting graph contains aten-level ops (e.g.
        ``torch.ops.aten.addmm.default``).  These work fine for tracing and
        ONNX export, but ``torch.jit.script`` cannot resolve them.  This
        pass rewrites them to the public API equivalents that TorchScript
        understands.
        """
        import torch.nn.functional as F

        # -----------------------------------------------------------------
        # Ops that should become call_method nodes (tensor methods).
        # key = overloadpacket name, value = method name on the tensor.
        # -----------------------------------------------------------------
        _TO_METHOD = {
            'view': 'view',
            'reshape': 'reshape',
            'contiguous': 'contiguous',
            'to': 'to',
        }

        # -----------------------------------------------------------------
        # Ops with custom call_function replacements (different name or
        # needs arg rewriting).  Each value is (replacement_fn, arg_rewriter)
        # where arg_rewriter is None or a callable(args, kwargs)->(args, kwargs).
        # -----------------------------------------------------------------
        def _upsample_bilinear2d_args(args, kwargs):
            # aten sig: (input, output_size, align_corners)
            # F.interpolate sig: (input, size=, mode=, align_corners=)
            inp, size, align_corners = args[0], args[1], args[2]
            return (inp,), {
                'size': size, 'mode': 'bilinear',
                'align_corners': align_corners,
            }

        def _upsample_bicubic2d_args(args, kwargs):
            inp, size, align_corners = args[0], args[1], args[2]
            return (inp,), {
                'size': size, 'mode': 'bicubic',
                'align_corners': align_corners,
            }

        def _upsample_nearest2d_args(args, kwargs):
            inp, size = args[0], args[1]
            return (inp,), {'size': size, 'mode': 'nearest'}

        _CUSTOM_FN = {
            'upsample_bilinear2d': (F.interpolate, _upsample_bilinear2d_args),
            'upsample_bicubic2d':  (F.interpolate, _upsample_bicubic2d_args),
            'upsample_nearest2d':  (F.interpolate, _upsample_nearest2d_args),
        }

        # -----------------------------------------------------------------
        # Ops that can be handled by torch.narrow (slice with dim/start/end).
        # aten::slice.Tensor(self, dim, start, end, step=1)
        # When step==1 this is equivalent to torch.narrow.
        # Identity slices (start=0, end>=2^62) are replaced by the input.
        # -----------------------------------------------------------------
        _SLICE_OPS = {'slice', 'slice_copy'}

        rewritten = []
        nodes_to_erase = []

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if not isinstance(target, torch._ops.OpOverload):
                continue
            op_name = target.overloadpacket.__name__

            # --- Method conversion ---
            if op_name in _TO_METHOD:
                method_name = _TO_METHOD[op_name]
                node.op = "call_method"
                node.target = method_name
                # First arg is self; remaining stay as-is.
                rewritten.append(op_name)
                continue

            # --- Custom function replacement ---
            if op_name in _CUSTOM_FN:
                fn, arg_rewriter = _CUSTOM_FN[op_name]
                if arg_rewriter is not None:
                    node.args, node.kwargs = arg_rewriter(
                        node.args, node.kwargs)
                node.target = fn
                rewritten.append(op_name)
                continue

            # --- Slice handling ---
            # aten::slice.Tensor(self, dim, start, end, step=1)
            # Rewrite to torch.narrow for all backends:
            #   - jit-script can't resolve aten OpOverloads
            #   - onnx-dynamo produces bad ONNX Slice nodes for MAX_INT end
            # Identity slices (start=0, end=MAX) are removed entirely.
            # "To end" slices compute length dynamically via size()-start.
            if op_name in _SLICE_OPS:
                tensor_arg = node.args[0]
                dim = node.args[1] if len(node.args) > 1 else 0
                start = node.args[2] if len(node.args) > 2 else 0
                end = node.args[3] if len(node.args) > 3 else (1 << 62)
                step = node.args[4] if len(node.args) > 4 else 1
                is_end_max = isinstance(end, int) and end >= (1 << 62)

                if step == 1 and start == 0 and is_end_max:
                    # Identity slice — remove (no-op)
                    node.replace_all_uses_with(tensor_arg)
                    nodes_to_erase.append(node)
                    rewritten.append(op_name)
                    continue

                if step == 1:
                    if is_end_max:
                        # "To end" slice: length = tensor.size(dim) - start
                        with graph.inserting_before(node):
                            size_node = graph.call_function(
                                torch.Tensor.size,
                                args=(tensor_arg, dim),
                            )
                            length_node = graph.call_function(
                                torch.sub,
                                args=(size_node, start),
                            )
                        node.target = torch.narrow
                        node.args = (tensor_arg, dim, start, length_node)
                        node.kwargs = {}
                    else:
                        # Bounded slice: length = end - start
                        length = end - start
                        node.target = torch.narrow
                        node.args = (tensor_arg, dim, start, length)
                        node.kwargs = {}
                else:
                    # Stepped slice — fall through to generic lookup
                    pass
                rewritten.append(op_name)
                continue

            # --- Generic name-based lookup ---
            replacement = getattr(torch, op_name, None)
            if replacement is None or not callable(replacement):
                replacement = getattr(F, op_name, None)
            if replacement is not None and callable(replacement):
                node.target = replacement
                rewritten.append(op_name)
            else:
                _get_logger().warning(
                    f"Could not find a high-level equivalent for aten op "
                    f"'{target._name}'. torch.jit.script may fail.")

        # Erase identity-slice nodes (must be done after iteration)
        for node in nodes_to_erase:
            graph.erase_node(node)

        if rewritten:
            _get_logger().debug(
                f"Rewrote {len(rewritten)} aten OpOverload node(s) to "
                f"high-level equivalents: {rewritten}")
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
                    name, torch.nn.Parameter(param.contiguous(),
                                             requires_grad=param.requires_grad))
                count += 1
        if count:
            _get_logger().debug(
                f"Made {count} non-contiguous tensor constant(s) contiguous")

    def _create_io_helper(self, data, name: str, to: str):
        if isinstance(data, collections.abc.Mapping):
            new_data = {}
            for key, value in data.items():
                child_name = "_".join([name, key]) if name else key
                new_data[key] = self._create_io_helper(
                    value, child_name, to)
            return new_data
        elif isinstance(data, (list, tuple)):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)]) if name else str(idx)
                new_data.append(self._create_io_helper(
                    value, child_name, to))
            return type(data)(new_data)
        
        elif is_tracable_tensor_type(data):
            is_traced = False
            if isinstance(data, TracedData) and data.is_tracing:
                is_traced = True # is already a traced tensor that is currently tracing AND the context is the same as the current node
                # Check if the traced tensor is from the same context (node)
                if data.context_obj is not self:
                    # Different context: error - cannot use traced tensor from another node
                    _get_logger().fatal(
                        f"Error: when creating inputs for '{self.name}', "
                        f"detected data '{name}' is an active {data.__class__.__name__} from a different node '{data.context}'. \n"
                        f"Mixing active contexts is not allowed. "
                        f"Call output_tensors() on the source node first.",
                        error_type=Exception)

            if to=="traced":
                if self.dry_run:
                    return data

                if is_traced:
                    # Same context: allow override with warning
                    _get_logger().warning(
                        f"Input '{name}' for node '{self.name}' is an active TracedTensor "
                        f"from the same node. Creating fresh input placeholder "
                        f"(previous trace will be discarded for this branch)."
                    )

                node = self.graph.create_node("placeholder", name, (), {})
                proxy = Proxy(node, self.tracer)
                return as_traced(data, name, self, proxy)

            elif to=="tensor":
                if not is_traced:
                    _get_logger().warning(
                        f"Warning: when creating outputs for {self.name}, "
                        f"detected data {name} is not a TracedTensor")
                return data
            
            elif to=="static":
                if is_traced:
                    _get_logger().fatal(
                        f"Cannot create static output from TracedTensor '{name}'. "
                        "Static outputs must be raw tensors.",
                        error_type=Exception)

                # Create unique attribute name and store on root module
                attr_name = f"_static_{name}".replace(".", "_")
                self.tracer.root.register_buffer(attr_name, data.clone())
                # Create get_attr node that retrieves the stored constant
                node = self.graph.create_node("get_attr", attr_name, (), {})
                proxy = Proxy(node, self.tracer)
                return as_traced(data, name, self, proxy)
        else:
            if to == "traced":
                _get_logger().warning(
                    f"Non-tracable input '{name}' (type={type(data).__name__}) in node '{self.name}' "
                    f"will be passed through as a constant.")
            return data

    def create_input(self, data, name: str, semantics=None) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            data: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        self.add_input(name, name, data, semantics=semantics)
        traced_data = self._create_io_helper(data, name, to="traced")
        return traced_data

    def create_output(self, data, name: str, static: bool = False, semantics=None):
        if static:
            self._validate_static_tensor(data, name)
            wrapped = self._create_io_helper(data, name, to="static")
            self.tag_data(data, name)
            self.add_output(name, name, data, semantics=semantics)
            return wrapped
        else:
            unwrapped_data = self._create_io_helper(data, name, to="tensor")
            self.tag_data(unwrapped_data, name)
            self.add_output(name, name, unwrapped_data, semantics=semantics)
            return unwrapped_data

    def _validate_static_tensor(self, tensor, name: str):
        if not isinstance(tensor, torch.Tensor):
            _get_logger().fatal(
                f"Error: static output '{name}' has type {type(tensor).__name__} "
                "but expected torch.Tensor.\n"
                "**Static outputs must be raw tensors, not derived from input tensors.**\n"
                "If this value depends on inputs, use it as a regular output tensor instead.",
                error_type=Exception)
        if isinstance(tensor, TracedTensor):
            _get_logger().fatal(
                f"Error: static output '{name}' is a TracedTensor. "
                "Static outputs should be constant tensors, not traced computations.",
                error_type=Exception)

    def create_static_tensors(self, static_outputs):
        """Wrap raw tensors as static graph nodes (for register_buffer).

        Unlike create_output(static=True), this does NOT tag or register
        outputs — it only validates and wraps.
        Returns data in the same nested structure as the input payload.
        """
        flattened_static_outputs = flatten_io_structure(static_outputs, '')
        for tensor_name, tensor in flattened_static_outputs.items():
            self._validate_static_tensor(tensor, tensor_name)

        return self._create_io_helper(static_outputs, '', to="static")

    def build_graph_module(self, outputs: list["TracedTensor"]) -> fx.GraphModule:
        """Convert the traced computation to a torch.fx.GraphModule.

        Args:
            outputs: List of TracedTensor outputs to include in the graph module

        Returns:
            fx.GraphModule: A graph module that takes all inputs and returns all outputs
        """
        # Find all nodes that are actually used by traversing backwards from outputs
        output_nodes = [output.proxy.node for output in outputs]
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
        for node in reversed(nodes_to_remove):
            if node.op == "placeholder":
                input_description = LeappNode.get_io_description_by_name(
                    node.target, self.inputs)
                if input_description is None:
                    _get_logger().error(
                        f"Error: when building the graph module for {self.name}, "
                        f"could not find input description for placeholder '{node.target}'")
                    continue
                self.trimmed_inputs.add(input_description.name_str)

            if len(node.users) == 0:
                self.graph.erase_node(node)

        if len(self.trimmed_inputs) > 0:
            _get_logger().warning(f"Warning: when building the graph module for {self.name}, "
                                  "detected the following inputs are not used in the computation or directly returned as output: "
                                  f"{self.trimmed_inputs} \n"
                                  "For clarity and efficiency, consider removing these inputs")

            # Remove trimmed inputs from self.inputs
            self.inputs = [
                inp for inp in self.inputs if inp.name_str not in self.trimmed_inputs]

        # Check if graph already has an output node
        has_output = any(node.op == "output" for node in self.graph.nodes)

        if not has_output:
            # If single output, return it directly; if multiple, return as tuple
            if len(output_nodes) == 1:
                self.graph.output(output_nodes[0])
            else:
                self.graph.output(tuple(output_nodes))

    def create_state_tensors(self, tensors: dict[str, torch.Tensor]) -> dict[str, TracedTensor]:
        """Create state tensors that are both inputs and outputs."""
        result = {}
        for name, tensor in tensors.items():
            if name in self._state_tensors:
                _get_logger().warning(
                    f"State tensor '{name}' already registered for node '{self.name}'. "
                    "Returning existing TracedTensor.")
                result[name] = self._state_tensors[name]["input"]
                continue

            # Create input placeholder for this state
            traced_input = self.create_input(tensor, name)

            # Track as state tensor
            self._state_tensors[name] = {
                "input": traced_input,
                "input_desc": self._get_required_io_description(name, self.inputs, "input"),
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
                    error_type=Exception)

            # Validate shape and dtype match the input state
            input_tensor = self._state_tensors[name]["input"]
            input_underlying = input_tensor.tensor if isinstance(input_tensor, TracedTensor) else input_tensor
            value_underlying = value.tensor if isinstance(value, TracedTensor) else value
            if input_underlying.shape != value_underlying.shape:
                _get_logger().fatal(
                    f"Error: update_state for '{name}' has shape {value_underlying.shape} "
                    f"but expected {input_underlying.shape} (must match input state shape).",
                    error_type=Exception)
            if input_underlying.dtype != value_underlying.dtype:
                _get_logger().fatal(
                    f"Error: update_state for '{name}' has dtype {value_underlying.dtype} "
                    f"but expected {input_underlying.dtype} (must match input state dtype).",
                    error_type=Exception)

            self._state_tensors[name]["output"] = value

    def get_state_outputs(self) -> dict[str, TracedTensor]:
        """Get state outputs using the original state names.

        Only states explicitly updated via update_state() become feedback outputs.
        States declared without an update are treated as regular inputs.
        """
        inactive_states = [
            name for name, info in self._state_tensors.items()
            if info["output"] is None
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

    def add_warp_segment(
        self,
        *,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpSegment:
        """Create a new open Warp segment for this node.

        If the previous segment is still open, finalize it into an FX marker
        before starting the next one. This keeps the FX graph at one marker per
        Warp segment while leaving full model compilation to compile_trace().
        """
        if self.warp_segments and self.warp_segments[-1].is_open:
            self.close_warp_segment(self.warp_segments[-1])

        segment = WarpSegment(
            node_name=self.name,
            device=device,
            metadata=metadata or {},
        )
        self.warp_segments.append(segment)
        return segment

    def get_warp_segment(
        self,
        *,
        device: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpSegment:
        """Return the latest open Warp segment, creating one if needed."""
        if not self.warp_segments or not self.warp_segments[-1].is_open:
            return self.add_warp_segment(device=device, metadata=metadata)

        segment = self.warp_segments[-1]
        if device is not None:
            segment.device = device
        if metadata:
            segment.metadata.update(metadata)
        return segment

    def close_warp_segment(self, segment: WarpSegment | None = None) -> WarpSegment:
        """Finalize a Warp segment into a single FX marker.

        The live ``WarpSegment`` remains the source of truth during tracing. The
        FX nodes keep pointers back to the segment/ref objects, and a single
        metadata snapshot is written after finalization for debugging/export.
        """
        if segment is None:
            if not self.warp_segments:
                raise RuntimeError(
                    f"Node '{self.name}' has no Warp segment to close."
                )
            segment = self.warp_segments[-1]

        if segment.status == "closed":
            return segment
        if segment.status == "invalid":
            raise RuntimeError("Cannot close invalid Warp segment.")
        if segment.node_name != self.name:
            raise ValueError(
                f"Warp segment belongs to node '{segment.node_name}', not "
                f"'{self.name}'."
            )

        for segment_index, known_segment in enumerate(self.warp_segments):
            if known_segment is segment:
                break
        else:
            raise ValueError("Warp segment is not registered on this node.")

        if segment.is_empty:
            segment.status = "closed"
            return segment

        marker_name = f"warp_segment_{segment_index}"
        input_proxies = tuple(
            ref.proxy for ref in segment.input_refs.values() if ref.proxy is not None
        )
        output_refs = list(segment.output_refs.values())
        output_attr_proxies = tuple(
            self._create_warp_output_attr_proxy(segment, marker_name, ref)
            for ref in output_refs
        )

        marker = self.tracer.create_proxy(
            "call_function",
            warp_segment_marker,
            (*input_proxies, *output_attr_proxies),
            {},
            name=marker_name,
        )
        segment.marker_proxy = marker
        marker.node.meta["leapp_warp_segment"] = segment

        if len(output_refs) == 1:
            output_proxies = {output_refs[0].name: marker}
        else:
            output_proxies = {}
            for idx, ref in enumerate(output_refs):
                output_proxies[ref.name] = self.tracer.create_proxy(
                    "call_function",
                    operator.getitem,
                    (marker, idx),
                    {},
                    name=f"{marker_name}_{ref.name}",
                )

        for ref in output_refs:
            proxy = output_proxies[ref.name]
            ref.proxy = proxy
            ref.context = self
            segment.output_proxies[ref.name] = proxy
            proxy.node.meta["leapp_warp_segment"] = segment
            proxy.node.meta["leapp_warp_output_ref"] = ref
            self._update_output_ref_proxy(ref, proxy)

        segment.status = "closed"
        marker.node.meta["leapp_warp_segment_metadata"] = segment.to_metadata()
        return segment

    def _create_warp_output_attr_proxy(
        self, segment: WarpSegment, marker_name: str, ref: WarpTensorRef
    ) -> Proxy:
        attr_name = self._unique_warp_output_attr_name(marker_name, ref.name)
        value = ref.array
        if isinstance(value, torch.Tensor):
            self.tracer.root.register_buffer(attr_name, value)
        else:
            setattr(self.tracer.root, attr_name, value)

        proxy = self.tracer.create_proxy("get_attr", attr_name, (), {})
        segment.output_attr_proxies[ref.name] = proxy
        proxy.node.meta["leapp_warp_segment"] = segment
        proxy.node.meta["leapp_warp_output_ref"] = ref
        return proxy

    def _unique_warp_output_attr_name(self, marker_name: str, output_name: str) -> str:
        base = f"_{marker_name}_{output_name}_value"
        base = "".join(ch if ch.isalnum() else "_" for ch in base)
        candidate = base
        suffix = 0
        while hasattr(self.tracer.root, candidate):
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    def _update_output_ref_proxy(self, ref: WarpTensorRef, proxy: Proxy) -> None:
        if ref.array is None:
            return

        init_tracing_state = getattr(ref.array, "_init_tracing_state", None)
        if init_tracing_state is None:
            return

        init_tracing_state(ref.name, self, proxy)


    @property
    def state_names(self) -> list[str]:
        """Get list of registered state tensor names."""
        return list(self._state_tensors.keys())
