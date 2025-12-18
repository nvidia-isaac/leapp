from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.leapp_graph.traced_tensor import TracedTensor
from leapp.utils import resolve_tensor_descriptions_to_names


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

    def _create_io_helper(self, data, name: str, to_traced: bool):
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                child_name = "_".join([name, key])
                new_data[key] = self._create_io_helper(
                    value, child_name, to_traced)
            return new_data
        elif isinstance(data, list):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)])
                new_data.append(self._create_io_helper(
                    value, child_name, to_traced))
            return new_data
        elif isinstance(data, tuple):
            new_data = []
            for idx, value in enumerate(data):
                child_name = "_".join([name, str(idx)])
                new_data.append(self._create_io_helper(
                    value, child_name, to_traced))
            return tuple(new_data)
        elif isinstance(data, TracedTensor):
            tensor_val = data.tensor
            if to_traced:
                # cannot retrace a currently active traced tensor
                if data.is_tracing:
                    _get_logger().error(f"Error: when creating inputs for {self.name}, \
                                            detected data {name} is already a TracedTensor and is tracing")
                    raise ValueError("Error in TracedTensorNode")
                return self._create_io_helper(tensor_val, name, to_traced)
            else:
                # return the underlying tensor value
                if not data.is_tracing:
                    _get_logger().warning(f"Warning: when creating outputs for {self.name}, \
                                            detected data {name} is a TracedTensor but is not tracing")

                return tensor_val

        elif isinstance(data, torch.Tensor):
            if to_traced:  # convert the tensor to a TracedTensor
                """ Future warp support
                    if isinstance(tensor, wp.array):
                        tensor = wp.to_torch(tensor)
                """
                node = self.graph.create_node("placeholder", name, (), {})
                proxy = Proxy(node, self.tracer)
                return TracedTensor(data, name, self, proxy)
            else:
                _get_logger().warning(f"Warning: when creating outputs for {self.name}, \
                                       detected data {name} is not a TracedTensor")
                return data

        else:
            _get_logger().error(f"Error: when creating inputs for {self.name}, detected data {name} is {type(data).__name__}"
                                " which is not a dict, list, tuple, or torch.Tensor")
            raise ValueError("Error in TracedTensorNode")

    def create_input(self, data, name: str) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            tensor: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        self.add_input(name, name, data)
        traced_data = self._create_io_helper(data, name, to_traced=True)
        return traced_data

    def create_output(self, data, name: str):
        unwrapped_data = self._create_io_helper(data, name, to_traced=False)
        self.tag_data(unwrapped_data, name)
        self.add_output(name, name, unwrapped_data)
        return unwrapped_data

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
            if len(node.users) == 0:
                self.graph.erase_node(node)

        # Check if graph already has an output node
        has_output = any(node.op == "output" for node in self.graph.nodes)

        if not has_output:
            # If single output, return it directly; if multiple, return as tuple
            if len(output_nodes) == 1:
                self.graph.output(output_nodes[0])
            else:
                self.graph.output(tuple(output_nodes))
