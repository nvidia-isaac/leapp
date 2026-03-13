from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
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
                       static_tensors: dict[str, torch.Tensor] | None = None):
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
                self.add_output(name, name, tensor)
                if name in self._state_tensors:
                    self._state_tensors[name]["output_desc"] = self._get_required_io_description(
                        name, self.outputs, "output"
                    )
            if static_tensors:
                for name, tensor in static_tensors.items():
                    self.tag_data(tensor, name)
                    self.add_output(name, name, tensor)
            self._apply_state_tags()
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        unwrapped_tensors = []
        for name, tensor in tensors.items():
            unwrapped = self.create_output(tensor, name)
            unwrapped_tensors.append(unwrapped)
            if name in self._state_tensors:
                self._state_tensors[name]["output_desc"] = self._get_required_io_description(
                    name, self.outputs, "output"
                )

        # Static tensors go through the same create_output path with static=True
        if static_tensors:
            for name, tensor in static_tensors.items():
                wrapped = self.create_output(tensor, name, static=True)
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
        """Convert call_function nodes with method_descriptor targets to call_method.

        Method descriptors (e.g. torch.Tensor.view, torch.Tensor.float) don't
        support weak references, which causes torch.jit.script to fail with:
            TypeError: cannot create weak reference to 'method_descriptor' object

        Converting them to call_method nodes uses a string target instead,
        sidestepping the issue entirely. The args layout is identical — args[0]
        is already ``self`` for unbound method descriptors, which is exactly
        what call_method expects.
        """
        rewritten = []
        for node in graph.nodes:
            if node.op == "call_function" and type(node.target).__name__ == "method_descriptor":
                method_name = getattr(node.target, "__name__", None)
                if method_name is not None:
                    node.op = "call_method"
                    node.target = method_name
                    rewritten.append(method_name)
        if rewritten:
            _get_logger().debug(
                f"Rewrote {len(rewritten)} method_descriptor call_function node(s) "
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
                    _get_logger().error(
                        f"Error: when creating inputs for '{self.name}', "
                        f"detected data '{name}' is an active {data.__class__.__name__} from a different node '{data.context}'. \n"
                        f"Mixing active contexts is not allowed. "
                        f"Call output_tensors() on the source node first."
                    )
                    raise Exception(
                        f"Error: when creating inputs for '{self.name}', "
                        f"detected data '{name}' is an active {data.__class__.__name__} from a different node '{data.context}'. \n"
                        f"Mixing active contexts is not allowed. "
                        f"Call output_tensors() on the source node first."
                    )

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
                    _get_logger().error(f"Cannot create static output from TracedTensor '{name}'. "
                    "Static outputs must be raw tensors.")
                    raise Exception("Error in TracedTensorNode")

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

    def create_input(self, data, name: str) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            data: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        self.add_input(name, name, data)
        traced_data = self._create_io_helper(data, name, to="traced")
        return traced_data

    def create_output(self, data, name: str, static: bool = False):
        if static:
            self._validate_static_tensor(data, name)
            wrapped = self._create_io_helper(data, name, to="static")
            self.tag_data(data, name)
            self.add_output(name, name, data)
            return wrapped
        else:
            unwrapped_data = self._create_io_helper(data, name, to="tensor")
            self.tag_data(unwrapped_data, name)
            self.add_output(name, name, unwrapped_data)
            return unwrapped_data

    def _validate_static_tensor(self, tensor, name: str):
        if not isinstance(tensor, torch.Tensor):
            _get_logger().error(
                f"Error: static output '{name}' has type {type(tensor).__name__} "
                "but expected torch.Tensor.\n"
                "**Static outputs must be raw tensors, not derived from input tensors.**\n"
                "If this value depends on inputs, use it as a regular output tensor instead.")
            raise Exception("Error: exception detected in output_tensors declaration")
        if isinstance(tensor, TracedTensor):
            _get_logger().error(
                f"Error: static output '{name}' is a TracedTensor. "
                "Static outputs should be constant tensors, not traced computations.")
            raise Exception("Error: exception detected in output_tensors declaration")

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
                _get_logger().error(
                    f"Error: update_state called for '{name}' but it was not registered "
                    f"as a state tensor. Call state_tensors() first.")
                raise Exception("Error: exception detected in update_state_tensors")

            # Validate shape and dtype match the input state
            input_tensor = self._state_tensors[name]["input"]
            input_underlying = input_tensor.tensor if isinstance(input_tensor, TracedTensor) else input_tensor
            value_underlying = value.tensor if isinstance(value, TracedTensor) else value
            if input_underlying.shape != value_underlying.shape:
                _get_logger().error(
                    f"Error: update_state for '{name}' has shape {value_underlying.shape} "
                    f"but expected {input_underlying.shape} (must match input state shape).")
                raise Exception("Error: exception detected in update_state_tensors")
            if input_underlying.dtype != value_underlying.dtype:
                _get_logger().error(
                    f"Error: update_state for '{name}' has dtype {value_underlying.dtype} "
                    f"but expected {input_underlying.dtype} (must match input state dtype).")
                raise Exception("Error: exception detected in update_state_tensors")

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

    @property
    def state_names(self) -> list[str]:
        """Get list of registered state tensor names."""
        return list(self._state_tensors.keys())
