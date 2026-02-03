from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.leapp_graph.datatypes import TracedTensor
from leapp.utils import (resolve_tensor_descriptions_to_names,
                         is_tracable_tensor_type)
from leapp.leapp_graph.datatypes.traced_data import TracedData


class TracedTensorNode(LeappNode):
    def __init__(self, name, node_index, *args, **kwargs):
        super().__init__(name, node_index)
        """Initialize a shared tracing context."""
        self.graph = fx.Graph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}

        self.compiled_graph_module = None

    @property
    def is_tracing(self) -> bool:
        return not self._model_captured

    def compile_trace(self, tensors: dict[str, "TracedTensor"], backend=None, backend_params={}):
        # No longer checking this because upstream will validate
        # if any(not isinstance(tensor, TracedTensor) for tensor in tensors.values()):
        #     _get_logger().error(
        #         f"Error: Call to compile_trace for {self.name} using non-TracedTensors")
        #     raise ValueError("Error in TracedTensorNode")

        unwrapped_tensors = []
        for name, tensor in tensors.items():
            unwrapped_tensors.append(self.create_output(tensor, name))
        self.build_graph_module(list(tensors.values()))

        self.compiled_graph_module = fx.GraphModule(
            self.tracer.root, self.graph)
        _get_logger().debug(
            f"Compiled graph module for {self.name}: {self.compiled_graph_module.graph}")
        _get_logger().debug(
            f"Graph module inputs: {[resolve_tensor_descriptions_to_names(input) for input in self.input_formats]}")
        _get_logger().debug(
            f"Graph module outputs: {[resolve_tensor_descriptions_to_names(output) for output in self.output_formats]}")
        self.setup_backend(backend, backend_params)

        return unwrapped_tensors[0] if len(unwrapped_tensors) == 1 else tuple(unwrapped_tensors)

    def setup_backend(self, backend=None, backend_params={}):
        super().setup_backend(backend, backend_params)
        if self.compiled_graph_module is None:
            _get_logger().error(
                f"Error: TracedTensorNode {self.name} has no compiled graph module, please compile the trace first")
            raise ValueError("Error in TracedTensorNode")

        self.export_backend.override_module_builder(
            lambda: self.compiled_graph_module)

    def _create_io_helper(self, data, name: str, to: str):
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                child_name = "_".join([name, key]) if name else key
                new_data[key] = self._create_io_helper(
                    value, child_name, to)
            return new_data
        elif isinstance(data, list):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)]) if name else str(idx)
                new_data.append(self._create_io_helper(
                    value, child_name, to))
            return new_data
        elif isinstance(data, tuple):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)]) if name else str(idx)
                new_data.append(self._create_io_helper(
                    value, child_name, to))
            return tuple(new_data)

        elif is_tracable_tensor_type(data):
            is_traced = False
            if isinstance(data, TracedData) and data.is_tracing:
                is_traced = True # is already a traced tensor that is currently tracing AND the context is the same as the current node
                # Check if the traced tensor is from the same context (node)
                if data.context_obj is not self:
                    # Different context: error - cannot use traced tensor from another node
                    _get_logger().error(
                        f"Error: when creating inputs for '{self.name}', "
                        f"detected data '{name}' is an active TracedTensor from a different node '{data.context}'. "
                        f"Cannot use TracedTensor from one node as input to another. "
                        f"Call output_tensors() on the source node first."
                    )
                    raise Exception("Error in TracedTensorNode")

            if to=="traced":
                if is_traced:
                    # Same context: allow override with warning
                    _get_logger().warning(
                        f"Input '{name}' for node '{self.name}' is an active TracedTensor "
                        f"from the same node. Creating fresh input placeholder "
                        f"(previous trace will be discarded for this branch)."
                    )

                node = self.graph.create_node("placeholder", name, (), {})
                proxy = Proxy(node, self.tracer)
                return TracedTensor(data, name, self, proxy)

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
                return TracedTensor(data, name, self, proxy)



        else:
            _get_logger().error(f"Error: when creating inputs for {self.name}, detected data {name} is {type(data).__name__}"
                                " which is not a dict, list, tuple, or accepted tracable tensor type")
            raise Exception("Error in TracedTensorNode")

    def create_input(self, data, name: str) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            tensor: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        self.add_input(name, name, data)
        traced_data = self._create_io_helper(data, name, to="traced")
        return traced_data

    def create_output(self, data, name: str):
        unwrapped_data = self._create_io_helper(data, name, to="tensor")
        self.tag_data(unwrapped_data, name)
        self.add_output(name, name, unwrapped_data)
        return unwrapped_data
    
    def create_static_tensors(self, flattened_static_outputs):
        ''' assumes the input is already flattened in the form of Dict[str, torch.Tensor] '''
        # Validate all static outputs are raw tensors (not TracedTensors)
        for tensor_name, tensor in flattened_static_outputs.items():
            if not isinstance(tensor, torch.Tensor):
                _get_logger().error(
                    f"Error: static output '{tensor_name}' has type {type(tensor).__name__} "
                    "but expected torch.Tensor.\n"
                    "**Static outputs must be raw tensors, not derived from input tensors.**\n"
                    "If this value depends on inputs, use it as a regular output tensor instead.")
                raise Exception("Error: exception detected in output_tensors declaration")
            if isinstance(tensor, TracedTensor):
                _get_logger().error(
                    f"Error: static output '{tensor_name}' is a TracedTensor. "
                    "Static outputs should be constant tensors, not traced computations.")
                raise Exception("Error: exception detected in output_tensors declaration")
        
        # Wrap static tensors as TracedTensors so they appear in the graph output
        wrapped_static_outputs = self._create_io_helper(
            flattened_static_outputs, '', to="static")
        
        return wrapped_static_outputs

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
                    node.name, self.inputs)
                if input_description is None:
                    _get_logger().error(
                        f"Error: when building the graph module for {self.name}")
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
