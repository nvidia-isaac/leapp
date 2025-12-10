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

        self.is_tracing = True
        self.compiled_graph_module = None

    def compile_trace(self, tensors: dict[str, "TracedTensor"], backend=None, backend_params={}):
        if any(not isinstance(tensor, TracedTensor) for tensor in tensors.values()):
            _get_logger().error(f"Error: Call to compile_trace with some tensors are not TracedTensors")
            raise ValueError(
                f"Error: Call to compile_trace with some tensors are not TracedTensors")
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
            raise Exception(
                f"Error: TracedTensorNode {self.name} has no compiled graph module, please compile the trace first")

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
        self.input_count += 1

        """ Future warp support
        if isinstance(tensor, wp.array):
            tensor = wp.to_torch(tensor)
        """
        if type(tensor) is not torch.Tensor:
            _get_logger().error(f"Error: tensor {name} is not a torch.Tensor")
            raise ValueError(f"Error: tensor {name} is not a torch.Tensor")

        self.add_input(name, name, tensor)
        node = self.graph.create_node("placeholder", name, (), {})
        proxy = Proxy(node, self.tracer)
        return TracedTensor(tensor, name, self, proxy)

    def create_output(self, tensor: "TracedTensor", name: str):
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
    def context(self) -> str:
        """Get the name of the TracedTensorNode that owns this tensor."""
        return self._context.name

    @property
    def context_obj(self) -> TracedTensorNode:
        """Get the TracedTensorNode that owns this tensor."""
        return self._context

    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the TracedTensorNode that owns this tensor."""
        return self._context.is_tracing

    def _new(self, tensor: torch.Tensor, proxy: Proxy = None) -> "TracedTensor":
        """Create a new TracedTensor in the same context.

        Intermediate tensors get auto-generated names based on the operation.
        When not tracing, proxy can be None.
        """
        if proxy is not None:
            # Generate a name based on the proxy node's name
            intermediate_name = str(proxy.node.name)
        else:
            # When not tracing, use a placeholder name
            intermediate_name = "untraced"
        return TracedTensor(tensor, intermediate_name, self._context, proxy)

    @staticmethod
    def _convert_method_descriptor(func, context=None):
        """Convert torch.Tensor method descriptors to torch.* function equivalents.

        Method descriptors (like torch.Tensor.div) can't be used with TorchScript
        because they don't support weak references. This converts them to their
        equivalent torch.* functions (like torch.div).
        A common situation where this is needed is when the source coded uses
        normalTensor + tracedTensor(normal tensor is first)
        """
        # Check if it's a method descriptor on torch.Tensor
        func_type = type(func).__name__
        if func_type in ('method_descriptor', 'builtin_function_or_method'):
            # Try to get the method name
            func_name = getattr(func, '__name__', None)
            if func_name:
                # Look for equivalent in torch module
                torch_func = getattr(torch, func_name, None)
                if torch_func is not None and callable(torch_func):
                    return torch_func
                # Log warning if we couldn't find the equivalent torch function
                if context is not None:
                    _get_logger().warning(
                        f"while tracing for {context} detected method_descriptor or builtin_function_or_method {func_name} "
                        f"but failed to find equivalent torch function. this may cause issues during graph creation"
                    )
        return func

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

        # Convert method descriptors to function equivalents for TorchScript compatibility
        # Pass context if we have a traced_tensor for better error messages
        context_name = traced_tensor.context if traced_tensor is not None else None
        func = cls._convert_method_descriptor(func, context_name)

        if traced_tensor is None:
            # Fallback to default behavior if no TracedTensor found
            return NotImplemented

        # Check if we're trying to call a TorchScript module/method
        # This will fail when trying to script the graph later
        if traced_tensor.is_tracing:
            func_type_name = type(func).__name__
            func_module = type(func).__module__ if hasattr(
                type(func), '__module__') else ''

            # Detect TorchScript ScriptMethod, ScriptModule, or RecursiveScriptModule
            if 'ScriptMethod' in func_type_name or 'ScriptModule' in func_type_name:
                _get_logger().error(
                    f"TorchScript modules cannot be used with TracedTensor during tracing.\n"
                    f"Detected call to: {func_type_name}\n"
                    f"Issue: The FX graph will contain references to TorchScript objects that cannot be scripted later.\n"
                    f"Solutions:\n"
                    f"  1. Use the regular (non-scripted) nn.Module instead during tracing\n"
                    f"  2. Extract and call the underlying operations directly\n"
                    f"  3. Compile the traced graph first (it will work), but note you cannot script it afterward"
                )
                raise ValueError(
                    f"TorchScript modules cannot be used with TracedTensor during tracing. Detected call to: {func_type_name}")

            # Also check if the function comes from torch.jit module
            if func_module.startswith('torch.jit') or func_module.startswith('torch._C'):
                # Check if it's actually a ScriptMethod by trying to access __self__
                if hasattr(func, '__self__') and hasattr(func.__self__, '__class__'):
                    self_class_name = func.__self__.__class__.__name__
                    if 'Script' in self_class_name:
                        _get_logger().error(
                            f"TorchScript modules cannot be used with TracedTensor during tracing.\n"
                            f"Detected: {func} from {self_class_name}\n"
                            f"The compiled FX graph will contain TorchScript references that prevent scripting.\n"
                            f"Use the original nn.Module instead of the scripted version during tracing."
                        )
                        raise ValueError(
                            f"TorchScript modules cannot be used with TracedTensor during tracing. Detected call to: {func}")

        # Helper to recursively unwrap TracedTensors
        def unwrap_traced_tensor(obj):
            if isinstance(obj, TracedTensor):
                return obj.tensor
            elif isinstance(obj, (list, tuple)):
                return type(obj)(unwrap_traced_tensor(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: unwrap_traced_tensor(v) for k, v in obj.items()}
            return obj

        # Extract real tensors for actual computation
        real_args = tuple(unwrap_traced_tensor(arg) for arg in args)
        real_kwargs = {k: unwrap_traced_tensor(v) for k, v in kwargs.items()}

        # Execute the actual operation
        tensor_out = func(*real_args, **real_kwargs)

        # Skip tracing if context is not tracing - return raw tensors
        if not traced_tensor.is_tracing:
            return tensor_out

        # Helper to recursively extract proxies
        def extract_proxy(obj):
            if isinstance(obj, TracedTensor):
                return obj.proxy
            elif isinstance(obj, (list, tuple)):
                return type(obj)(extract_proxy(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: extract_proxy(v) for k, v in obj.items()}
            return obj

        # Extract proxies for graph recording
        proxy_args = tuple(extract_proxy(arg) for arg in args)
        proxy_kwargs = {k: extract_proxy(v) for k, v in kwargs.items()}

        # Record the operation in the graph
        proxy_out = traced_tensor._context.tracer.create_proxy(
            "call_function", func, proxy_args, proxy_kwargs
        )

        # Handle multiple outputs (e.g., torch.split returns a tuple)
        if isinstance(tensor_out, (tuple, list)):
            # For operations that return tuples, we need to index the proxy for each output
            result = []
            for i, t in enumerate(tensor_out):
                if isinstance(t, torch.Tensor):
                    # Create a proxy for this specific output by indexing
                    item_proxy = traced_tensor._context.tracer.create_proxy(
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
        # Handle in-place operations (methods ending with _)
        if name.endswith('_') and not name.startswith('_') and len(name) > 1:
            base_name = name[:-1]  # Remove trailing underscore
            torch_func = getattr(torch, base_name, None)

            if torch_func is not None and callable(torch_func):
                def inplace_method(*args, **kwargs):
                    if not self.is_tracing:
                        # Just call the in-place method on the tensor
                        getattr(self._tensor, name)(*args, **kwargs)
                        return self

                    # Record as functional operation
                    result = torch_func(self, *args, **kwargs)

                    # Update internal state to maintain in-place semantics
                    if isinstance(result, TracedTensor):
                        self._tensor = result.tensor
                        self._proxy = result.proxy

                    return self

                return inplace_method

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
            # If it's a callable method, wrap it to ensure it goes through __torch_function__
            if callable(attr):
                def wrapped_method(*args, **kwargs):
                    # Call the method on the underlying tensor
                    result = attr(*args, **kwargs)
                    # If result is a tensor, wrap it in TracedTensor
                    if isinstance(result, torch.Tensor):
                        # Skip tracing if context is not tracing - return raw tensor
                        if not self.is_tracing:
                            return result
                        # Create a proxy node for this operation
                        proxy = self._proxy._tracer.create_proxy(
                            'call_method', name, (self._proxy,) + args, kwargs)
                        return self._new(result, proxy)
                    return result
                return wrapped_method
            return attr

        # Attribute not found
        raise AttributeError(
            f"'TracedTensor' object has no attribute '{name}'")

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

    # In-place arithmetic operators
    def __iadd__(self, other):
        """In-place addition (+=)."""
        if not self.is_tracing:
            self._tensor += other
            return self

        # Record as functional operation in graph
        result = torch.add(self, other)

        # Update internal state to maintain in-place semantics
        self._tensor = result.tensor
        self._proxy = result.proxy

        return self

    def __isub__(self, other):
        """In-place subtraction (-=)."""
        if not self.is_tracing:
            self._tensor -= other
            return self

        result = torch.sub(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __imul__(self, other):
        """In-place multiplication (*=)."""
        if not self.is_tracing:
            self._tensor *= other
            return self

        result = torch.mul(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __itruediv__(self, other):
        """In-place division (/=)."""
        if not self.is_tracing:
            self._tensor /= other
            return self

        result = torch.div(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __ipow__(self, other):
        """In-place power (**=)."""
        if not self.is_tracing:
            self._tensor **= other
            return self

        result = torch.pow(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __imatmul__(self, other):
        """In-place matrix multiplication (@=)."""
        if not self.is_tracing:
            self._tensor @= other
            return self

        result = torch.matmul(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

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

        Note: Boolean/mask indexing with TracedTensor masks is not supported
        because FX tracer cannot serialize TracedTensor objects as arguments.

        Note: This requires special handling because __getitem__ is a Python
        special method, not a torch.* function, so __torch_function__ doesn't
        intercept it. We need to manually create the proxy and wrap the result.

        Args:
            key: The indexing key (int, slice, tensor, list, tuple, etc.)

        Returns:
            TracedTensor or scalar with the indexing operation recorded

        Raises:
            NotImplementedError: If key is a TracedTensor (boolean mask or advanced indexing)
        """
        # Check for boolean mask indexing with TracedTensor
        if isinstance(key, TracedTensor):
            # Check if it's a boolean tensor (mask)
            if key.dtype == torch.bool:
                raise NotImplementedError(
                    "Boolean/mask indexing with TracedTensor is not supported. "
                    "The FX tracer cannot serialize TracedTensor objects as indexing arguments.\n"
                    "Alternatives:\n"
                    "  1. Use torch.masked_select(tensor, mask) instead\n"
                    "  2. Convert mask to regular tensor: tensor[mask.tensor]\n"
                    "  3. Use torch.where() for conditional selection"
                )
            else:
                # Non-boolean tensor indexing with TracedTensor
                raise NotImplementedError(
                    "Advanced indexing with TracedTensor indices is not supported. "
                    "The FX tracer cannot serialize TracedTensor objects as indexing arguments.\n"
                    "Convert to regular tensor first: tensor[indices.tensor]"
                )

        # Check for tuple containing TracedTensor
        if isinstance(key, tuple):
            for item in key:
                if isinstance(item, TracedTensor):
                    if item.dtype == torch.bool:
                        raise NotImplementedError(
                            "Boolean/mask indexing with TracedTensor is not supported. "
                            "The FX tracer cannot serialize TracedTensor objects as indexing arguments.\n"
                            "Use torch.masked_select() or convert mask to regular tensor."
                        )
                    else:
                        raise NotImplementedError(
                            "Advanced indexing with TracedTensor indices is not supported. "
                            "The FX tracer cannot serialize TracedTensor objects as indexing arguments.\n"
                            "Convert to regular tensor first."
                        )

        result_tensor = self._tensor[key]

        # Skip tracing if context is not tracing - return raw tensor
        if not self.is_tracing:
            return result_tensor

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

        # Skip tracing if context is not tracing - return raw tensor
        if not self.is_tracing:
            return result_tensor

        # For type conversions, we track it as an operation
        proxy_out = self._context.tracer.create_proxy(
            "call_method", "to", (self._proxy,) + args, kwargs
        )
        return self._new(result_tensor, proxy_out)
