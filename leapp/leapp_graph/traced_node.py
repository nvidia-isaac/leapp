import operator
import warnings

from leapp.leapp_graph.leapp_node import LeappNode
import torch
import torch.fx as fx
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
class TracedTensorNode(LeappNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Initialize a shared tracing context."""
        self.graph = fx.Graph()
        self.tracer = fx.Tracer()
        self.tracer.graph = self.graph
        self.tracer.root = torch.nn.Module()
        self.tracer.tensor_attrs = {}
        self.input_count = 0

        self.is_tracing = False
        self.compiled_graph_module = None

    def compile_trace(self, tensors: dict[str, "TracedTensor"], backend = None, backend_params = {}):
        for name, tensor in tensors.items():
            self.create_output(tensor, name)
        self.to_graph_module(list(tensors.values()))

        self.compiled_graph_module = fx.GraphModule(self.tracer.root, self.graph)
        self.setup_backend(backend, backend_params)
    
    def setup_backend(self, backend = None, backend_params = {}):
        super().setup_backend(backend, backend_params)
        if self.compiled_graph_module is None:
            raise Exception(f"Error: TracedTensorNode {self.name} has no compiled graph module, please compile the trace first")

        self.export_backend.override_module_builder(lambda: self.compiled_graph_module)
    
    def create_input(self, tensor: torch.Tensor, name: str) -> "TracedTensor":
        """Create a TrackedTensor as an input to this context.

        Args:
            tensor: The tensor to track
            name: Name for this input (e.g., "joint_pos")

        Returns:
            TracedTensor: A traced tensor in this context
        """
        self.input_count += 1
        
        """ Future warp support
        if isinstance(tensor, wp.array):
            tensor = wp.to_torch(tensor)
        """

        self.is_tracing = True
        self.add_input(name, name, tensor)
        node = self.graph.create_node("placeholder", name, (), {})
        proxy = Proxy(node, self.tracer)
        return TracedTensor(tensor, name, self, proxy)
    
    def create_output(self, tensor: "TracedTensor", name: str):
        self.add_output(name, name, tensor.tensor)

    def to_graph_module(self, outputs: list["TracedTensor"]) -> fx.GraphModule:
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

        # Remove unused placeholder nodes
        placeholders_to_remove = []
        for node in self.graph.nodes:
            if node.op == "placeholder" and node not in used_nodes:
                placeholders_to_remove.append(node)

        for node in placeholders_to_remove:
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

        # self.graph.lint()  # Linting fails with spurious errors about constant




class TracedTensor:
    """A tensor wrapper that records operations using torch.fx.

    This class wraps a torch.Tensor and records all operations performed on it
    by maintaining a computation graph via torch.fx.Proxy. The recorded graph
    can be exported to TorchScript or ONNX.

    TracedTensors must be created via TraceContext.create_input().
    """

    def __init__(self, tensor: torch.Tensor, name: str, context: TracedTensorNode, proxy: Proxy):
        """Initialize a TracedTensor.

        TracedTensors can only be created via TraceContext.create_input().

        Args:
            tensor: The actual torch.Tensor to wrap
            name: Name for the tensor (used in ONNX export and graph)
            context: The TraceContext that owns this tensor
            proxy: The fx.Proxy for graph recording
        """
        self._tensor = tensor
        self._name = name
        self._context = context
        self._proxy = proxy

    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying torch.Tensor."""
        return self._tensor

    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy

    @property
    def name(self) -> str:
        """Get the name of the tensor."""
        return self._name

    @property
    def context(self) -> TracedTensorNode:
        """Get the TracedTensorNode that owns this tensor."""
        return self._context

    def _new(self, tensor: torch.Tensor, proxy: Proxy) -> "TracedTensor":
        """Create a new TracedTensor in the same context.

        Intermediate tensors get auto-generated names based on the operation.
        """
        # Generate a name based on the proxy node's name
        intermediate_name = str(proxy.node.name)
        return TracedTensor(tensor, intermediate_name, self._context, proxy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch operations to record them in the graph.

        This method is called by PyTorch whenever a torch function is applied
        to this object. We execute the operation on the real tensor and record
        it in the FX graph.
        """
        if kwargs is None:
            kwargs = {}

        # Find the first TracedTensor in args (including nested in lists/tuples)
        def find_traced_tensor(obj):
            if isinstance(obj, TracedTensor):
                return obj
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    result = find_traced_tensor(item)
                    if result is not None:
                        return result
            return None

        traced_tensor = find_traced_tensor(args)

        if traced_tensor is None:
            # Fallback to default behavior if no TracedTensor found
            return NotImplemented

        # Helper to recursively unwrap TracedTensors
        def unwrap_traced_tensor(obj):
            if isinstance(obj, TracedTensor):
                return obj.tensor
            elif isinstance(obj, (list, tuple)):
                return type(obj)(unwrap_traced_tensor(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: unwrap_traced_tensor(v) for k, v in obj.items()}
            return obj

        # Helper to recursively extract proxies
        def extract_proxy(obj):
            if isinstance(obj, TracedTensor):
                return obj.proxy
            elif isinstance(obj, (list, tuple)):
                return type(obj)(extract_proxy(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: extract_proxy(v) for k, v in obj.items()}
            return obj

        # Extract real tensors for actual computation
        real_args = tuple(unwrap_traced_tensor(arg) for arg in args)
        real_kwargs = {k: unwrap_traced_tensor(v) for k, v in kwargs.items()}

        # Execute the actual operation
        tensor_out = func(*real_args, **real_kwargs)

        # Extract proxies for graph recording
        proxy_args = tuple(extract_proxy(arg) for arg in args)
        proxy_kwargs = {k: extract_proxy(v) for k, v in kwargs.items()}

        # Record the operation in the graph
        proxy_out = traced_tensor.context.tracer.create_proxy(
            "call_function", func, proxy_args, proxy_kwargs
        )

        # Handle multiple outputs (e.g., torch.split returns a tuple)
        if isinstance(tensor_out, (tuple, list)):
            # For operations that return tuples, we need to index the proxy for each output
            result = []
            for i, t in enumerate(tensor_out):
                if isinstance(t, torch.Tensor):
                    # Create a proxy for this specific output by indexing
                    item_proxy = traced_tensor.context.tracer.create_proxy(
                        "call_function", operator.getitem, (proxy_out, i), {}
                    )
                    result.append(traced_tensor._new(t, item_proxy))
                else:
                    result.append(t)
            return type(tensor_out)(result)
        elif isinstance(tensor_out, torch.Tensor):
            return traced_tensor._new(tensor_out, proxy_out)
        else:
            # Non-tensor return (e.g., .item(), .tolist())
            return tensor_out

    # Provide convenient access to tensor attributes
    @property
    def shape(self) -> torch.Size:
        """Get the shape of the underlying tensor."""
        return self._tensor.shape

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the underlying tensor."""
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying tensor."""
        return self._tensor.device

    def __str__(self) -> str:
        """String representation of TracedTensor."""
        return f"TracedTensor({self._tensor})"

    def __repr__(self) -> str:
        """String representation of TracedTensor."""
        return f"TracedTensor({self._tensor})"

    def __format__(self, format_spec: str) -> str:
        """Format the TracedTensor by delegating to the underlying tensor."""
        return self._tensor.__format__(format_spec)

    def __getattr__(self, name: str):
        """Forward unknown attributes to torch functions or tensor methods.

        This allows tensor methods like x.sum() to automatically work by
        forwarding to torch.sum(x), which then gets intercepted by
        __torch_function__ for graph recording.

        For attributes that don't have corresponding torch functions,
        we forward to the underlying tensor's attribute.
        """
        # Try to find a torch function with this name
        torch_func = getattr(torch, name, None)
        if torch_func is not None and callable(torch_func):
            # Return a wrapper that calls the torch function with self as first arg
            def method(*args, **kwargs):
                return torch_func(self, *args, **kwargs)

            return method

        # Fall back to the underlying tensor's attribute
        attr = getattr(self._tensor, name, None)
        if attr is not None:
            return attr

        # Attribute not found
        raise AttributeError(f"'TracedTensor' object has no attribute '{name}'")

    # Special methods: reshape and permute need custom handling due to signature differences
    def reshape(self, *shape):
        """Reshape the tensor.

        Note: torch.reshape expects (tensor, shape) but tensor.reshape
        allows both reshape(shape) and reshape(*shape), so we handle both.
        """
        return torch.reshape(self, shape)

    def permute(self, *dims):
        """Permute dimensions.

        Note: torch.permute expects (tensor, dims) but tensor.permute
        allows both permute(dims) and permute(*dims), so we handle both.
        """
        # Handle both permute(2, 0, 1) and permute((2, 0, 1))
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        return torch.permute(self, dims)

    # Arithmetic operators (must be explicit, __getattr__ doesn't work for these)
    def __add__(self, other):
        """Addition operator."""
        return torch.add(self, other)

    def __radd__(self, other):
        """Reverse addition operator."""
        return torch.add(other, self)

    def __sub__(self, other):
        """Subtraction operator."""
        return torch.sub(self, other)

    def __rsub__(self, other):
        """Reverse subtraction operator."""
        return torch.sub(other, self)

    def __mul__(self, other):
        """Multiplication operator."""
        return torch.mul(self, other)

    def __rmul__(self, other):
        """Reverse multiplication operator."""
        return torch.mul(other, self)

    def __truediv__(self, other):
        """Division operator."""
        return torch.div(self, other)

    def __rtruediv__(self, other):
        """Reverse division operator."""
        return torch.div(other, self)

    def __pow__(self, other):
        """Power operator."""
        return torch.pow(self, other)

    def __rpow__(self, other):
        """Reverse power operator."""
        return torch.pow(other, self)

    def __neg__(self):
        """Negation operator."""
        return torch.neg(self)

    def __matmul__(self, other):
        """Matrix multiplication operator (@)."""
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        """Reverse matrix multiplication operator (@)."""
        return torch.matmul(other, self)

    # Comparison operators (must be explicit, __getattr__ doesn't work for these)
    def __gt__(self, other):
        """Greater than operator."""
        return torch.gt(self, other)

    def __lt__(self, other):
        """Less than operator."""
        return torch.lt(self, other)

    def __ge__(self, other):
        """Greater than or equal operator."""
        return torch.ge(self, other)

    def __le__(self, other):
        """Less than or equal operator."""
        return torch.le(self, other)

    def __eq__(self, other):
        """Equal operator."""
        return torch.eq(self, other)

    def __ne__(self, other):
        """Not equal operator."""
        return torch.ne(self, other)

    def __getitem__(self, key):
        """Indexing operator for TracedTensor.

        Supports all Python indexing operations: slicing, integer indexing,
        tensor indexing, etc. The operation is recorded in the computation graph.

        Note: This requires special handling because __getitem__ is a Python
        special method, not a torch.* function, so __torch_function__ doesn't
        intercept it. We need to manually create the proxy and wrap the result.

        Args:
            key: The indexing key (int, slice, tensor, list, tuple, etc.)

        Returns:
            TracedTensor or scalar with the indexing operation recorded
        """
        result_tensor = self._tensor[key]
        proxy_out = self._context.tracer.create_proxy(
            "call_function", operator.getitem, (self._proxy, key), {}
        )
        if isinstance(result_tensor, torch.Tensor):
            return self._new(result_tensor, proxy_out)
        return result_tensor

    def __len__(self) -> int:
        """Length operator for TracedTensor.

        Returns:
            int: The length of the first dimension of the tensor
        """
        return len(self._tensor)

    def __bool__(self) -> bool:
        """Boolean conversion for TracedTensor.

        Delegates to the underlying tensor. This allows using TracedTensors
        in control flow (if/else) based on their trace-time values.
        """
        return bool(self._tensor)

    @property
    def T(self):
        """Transpose property (for 2D tensors)."""
        return torch.transpose(self, 0, 1)

    def to(self, *args, **kwargs):
        """Convert tensor to different dtype or device.

        Note: This requires special handling because .to() is a tensor method,
        not a torch.* function, so __torch_function__ doesn't intercept it.
        We need to manually create the proxy and wrap the result.
        """
        result_tensor = self._tensor.to(*args, **kwargs)
        # For type conversions, we track it as an operation
        proxy_out = self._context.tracer.create_proxy(
            "call_method", "to", (self._proxy,) + args, kwargs
        )
        return self._new(result_tensor, proxy_out)