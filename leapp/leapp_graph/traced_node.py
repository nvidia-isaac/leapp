from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.leapp_graph.traced_tensor import TracedTensor


class TracedTensorNode(LeappNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialize a shared tracing context."""
        self.graph = fx.Graph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}

        self.is_tracing = True
        self.compiled_graph_module = None

    def compile_trace(self, tensors: dict[str, "TracedTensor"], backend=None, backend_params={}):
        if any(not isinstance(tensor, TracedTensor) for tensor in tensors.values()):
            _get_logger().error(
                f"Error: Call to compile_trace for {self.name} using non-TracedTensors")
            raise ValueError("Error in TracedTensorNode")
        for name, tensor in tensors.items():
            self.create_output(tensor, name)
        self.build_graph_module(list(tensors.values()))

        self.compiled_graph_module = fx.GraphModule(
            self.tracer.root, self.graph)
        _get_logger().debug(
            f"Compiled graph module for {self.name}: {self.compiled_graph_module.graph}")
        self.setup_backend(backend, backend_params)

        # this will also set the flag that stops all tensors in this node from being traced
        self.is_tracing = False

    def setup_backend(self, backend=None, backend_params={}):
        super().setup_backend(backend, backend_params)
        if self.compiled_graph_module is None:
            _get_logger().error(
                f"Error: TracedTensorNode {self.name} has no compiled graph module, please compile the trace first")
            raise ValueError("Error in TracedTensorNode")

        self.export_backend.override_module_builder(
            lambda: self.compiled_graph_module)

    def create_input(self, tensor: torch.Tensor, name: str) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            tensor: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """

        """ Future warp support
        if isinstance(tensor, wp.array):
            tensor = wp.to_torch(tensor)
        """
        if type(tensor) is not torch.Tensor:
            _get_logger().error(f"Error: tensor {name} is not a torch.Tensor")
            raise ValueError("Error in TracedTensorNode")

        self.add_input(name, name, tensor)
        node = self.graph.create_node("placeholder", name, (), {})
        proxy = Proxy(node, self.tracer)
        return TracedTensor(tensor, name, self, proxy)

    def create_output(self, tensor: "TracedTensor", name: str):
        self.tag_data(tensor.tensor, name)
        self.add_output(name, name, tensor.tensor)

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
