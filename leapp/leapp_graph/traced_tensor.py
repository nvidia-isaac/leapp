import operator

import numpy as np
import torch
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.tracing_lock import TracingLock
from leapp.leapp_graph.datatypes.numpy_compatibility import (
    AXIS_TO_DIM_FUNCTIONS,
    convert_numpy_arg_to_torch,
    get_torch_equivalent_ufunc,
    get_torch_equivalent_func,
)


# =============================================================================
# Patch torch.from_numpy to handle TracedTensor
# =============================================================================
# torch.from_numpy does an early type check in C++ before __torch_function__
# can intercept it. We patch it here to pass through TracedTensors unchanged.

_original_torch_from_numpy = torch.from_numpy


def _patched_from_numpy(arr):
    """Patched torch.from_numpy that handles TracedTensor.
    
    If the input is a TracedTensor, return it directly (it's already a traced
    torch tensor). Otherwise, call the original torch.from_numpy.
    """
    # Import here to avoid circular import at module load time
    if isinstance(arr, TracedTensor):
        return arr
    return _original_torch_from_numpy(arr)


# Apply the patch
torch.from_numpy = _patched_from_numpy


class TracedTensor:
    """A tensor wrapper that records operations using torch.fx.

    This class wraps a torch.Tensor and records all operations performed on it
    by maintaining a computation graph via torch.fx.Proxy. The recorded graph
    can be exported to TorchScript or ONNX.

    TracedTensors must be created via TraceContext.create_input().
    """

    def __init__(self, tensor: torch.Tensor, name: str, context, proxy: Proxy):
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
        self._global_tracing_lock = TracingLock()

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
        """Get the name of the  that owns this tensor."""
        return self._context.name

    @property
    def context_obj(self):
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
                
                # Handle in-place operations (e.g., add_ -> torch.add)
                # In-place methods end with '_' but torch.* functions don't
                if func_name.endswith('_') and len(func_name) > 1:
                    base_name = func_name[:-1]  # Remove trailing underscore
                    torch_func = getattr(torch, base_name, None)
                    if torch_func is not None and callable(torch_func):
                        return torch_func
                
                # Log warning if we couldn't find the equivalent torch function
                if context is not None:
                    _get_logger().warning(
                        f"while tracing for {context} detected method_descriptor or builtin_function_or_method {func_name} "
                        f"but failed to find equivalent torch function. this may cause issues during graph creation"
                    )
        return func

    @staticmethod
    # Find the first TracedTensor in args (including nested in lists/tuples)
    def find_traced_tensor(obj):
        if isinstance(obj, TracedTensor):
            return obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = TracedTensor.find_traced_tensor(item)
                if result is not None:
                    return result
        return None
    
    # Helper to recursively unwrap TracedTensors
    @staticmethod
    def unwrap_traced_tensor(obj):
        if isinstance(obj, TracedTensor):
            return obj.tensor
        elif isinstance(obj, (list, tuple)):
            return type(obj)(TracedTensor.unwrap_traced_tensor(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: TracedTensor.unwrap_traced_tensor(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def find_all_contexts(obj, contexts=None):
        """Recursively find all unique context names."""
        if contexts is None:
            contexts = set()
        if isinstance(obj, TracedTensor) and obj.is_tracing:
            contexts.add(obj.context)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                TracedTensor.find_all_contexts(item, contexts)
        elif isinstance(obj, dict):
            for v in obj.values():
                TracedTensor.find_all_contexts(v, contexts)
        return contexts
    
    def validate_status(self, args = None, kwargs = None):
        if not self.is_tracing:
            return False
        if self.is_tracing and self._global_tracing_lock.is_active:
            _get_logger().error(
                f"Error: detected active TracedTensor {self._name} from node {self.context} inside of a traced function.\n"
                f"\n"
                f"This happens when you have an active TracedTensor and it is being used for computation inside of a traced function/block."
                f"\n"
                f"You must call output_tensors() to finalize the TracedTensor node first"
            )
            raise Exception(
                "Cannot use TracedTensor inside of a traced function/block. "
                "Call output_tensors() first to finalize the TracedTensor node"
            )
        
        contexts = set()
        if args is not None:
            for arg in args:
                contexts = TracedTensor.find_all_contexts(arg, contexts)
        if kwargs is not None:
            for kwarg in kwargs.values():
                contexts = TracedTensor.find_all_contexts(kwarg, contexts)

        if len(contexts) > 1:
            _get_logger().error(
                f"Error: detected multiple TracedTensor contexts: {contexts} inside of a traced function.\n"
                "\n"
                "This happens when you mix multiple active TracedTensors from different contexts inside of a traced function/block."
                "\n"
                "You can call output_tensors() to finalize one of the TracedTensor nodes first "
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
            raise Exception(
                "Cannot mix multiple active TracedTensors from different contexts inside of a traced function/block. "
                "Call output_tensors() to finalize one of the TracedTensor nodes first"
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
        return True


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch operations to record them in the graph.

        This method is called by PyTorch whenever a torch function is applied
        to this object. We execute the operation on the real tensor and record
        it in the FX graph.
        """
        if kwargs is None:
            kwargs = {}

        # Handle torch.from_numpy - if input is TracedTensor, just return it
        # This happens when code does: torch.from_numpy(traced_tensor.numpy())
        # Since .numpy() returns self (TracedTensor), we just pass it through
        if func is torch.from_numpy:
            if len(args) == 1 and isinstance(args[0], TracedTensor):
                return args[0]  # Already a TracedTensor, no conversion needed

        traced_tensor = TracedTensor.find_traced_tensor(args)

        # Convert method descriptors to function equivalents for TorchScript compatibility
        # Pass context if we have a traced_tensor for better error messages
        context_name = traced_tensor.context if traced_tensor is not None else None
        func = cls._convert_method_descriptor(func, context_name)

        if traced_tensor is None:
            # Fallback to default behavior if no TracedTensor found
            return NotImplemented


        # Extract real tensors for actual computation
        real_args = tuple(TracedTensor.unwrap_traced_tensor(arg) for arg in args)
        real_kwargs = {k: TracedTensor.unwrap_traced_tensor(v) for k, v in kwargs.items()}

        # ================== EXECUTE THE ACTUAL OPERATION ==================
        tensor_out = func(*real_args, **real_kwargs)
        # ================== END OF EXECUTE THE ACTUAL OPERATION ===========


        # Skip tracing if context is not tracing - return raw tensors
        if not traced_tensor.validate_status(args, kwargs):
            return tensor_out

        # Check if we're trying to call a TorchScript module/method
        # This will fail when trying to script the graph later
        # We check this here (after confirming we're tracing) to avoid unnecessary
        # type introspection when not recording to the graph
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
                f"  3. break the chain by calling output_tensors first then annotate the scripted module usage with other LEAPP api"
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

        # Helper to recursively extract proxies
        def extract_proxy(obj):
            if isinstance(obj, TracedTensor):
                return obj.proxy
            elif isinstance(obj, torch.nn.Parameter):
                # Convert Parameter to regular tensor so fx inlines it as constant
                # This is safe when exporting to ONNX/JIT freeze where weights are baked in
                return obj.data
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
                        # Just call the in-place method on the tensor and return raw tensor
                        getattr(self._tensor, name)(*args, **kwargs)
                        return self._tensor

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
                        if not self.validate_status():
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
            return self._tensor

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
            return self._tensor

        result = torch.sub(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __imul__(self, other):
        """In-place multiplication (*=)."""
        if not self.is_tracing:
            self._tensor *= other
            return self._tensor

        result = torch.mul(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __itruediv__(self, other):
        """In-place division (/=)."""
        if not self.is_tracing:
            self._tensor /= other
            return self._tensor

        result = torch.div(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __ipow__(self, other):
        """In-place power (**=)."""
        if not self.is_tracing:
            self._tensor **= other
            return self._tensor

        result = torch.pow(self, other)
        self._tensor = result.tensor
        self._proxy = result.proxy
        return self

    def __imatmul__(self, other):
        """In-place matrix multiplication (@=)."""
        if not self.is_tracing:
            self._tensor @= other
            return self._tensor

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
        if not self.validate_status():
            return result_tensor

        proxy_out = self._context.tracer.create_proxy(
            "call_function", operator.getitem, (self._proxy, key), {}
        )
        if isinstance(result_tensor, torch.Tensor):
            return self._new(result_tensor, proxy_out)
        return result_tensor

    def __setitem__(self, key, value):
        """Indexed assignment operator for TracedTensor.

        This allows assignments like `traced_tensor[:] = value` to be recorded
        in the computation graph. The TracedTensor is updated in-place to reflect
        the new values and proxy.

        If the value is a TracedTensor, this tensor "inherits" the tracing from
        that value, meaning subsequent operations on this tensor will continue
        to be traced.

        Args:
            key: The indexing key (int, slice, tensor, list, tuple, etc.)
            value: The value to assign (can be a TracedTensor or regular tensor/scalar)
        """
        # Unwrap value if it's a TracedTensor
        real_value = TracedTensor.unwrap_traced_tensor(value)
        
        # Perform the actual assignment
        self._tensor[key] = real_value
        
        # Skip tracing if context is not tracing
        if not self.validate_status():
            return
        
        # Extract proxy from value if it's a TracedTensor
        if isinstance(value, TracedTensor):
            value_proxy = value.proxy
        else:
            value_proxy = value
        
        # For full slice assignment [:], we can treat it as the value itself
        if key == slice(None) or (isinstance(key, tuple) and all(k == slice(None) for k in key)):
            # Full replacement - the proxy becomes the value's proxy
            if isinstance(value, TracedTensor):
                self._proxy = value.proxy
        else:
            # Partial assignment - record as a call_method
            proxy_out = self._context.tracer.create_proxy(
                "call_method", "__setitem__", (self._proxy, key, value_proxy), {}
            )
            self._proxy = proxy_out

    @staticmethod
    def copy_into(target: torch.Tensor, source: "TracedTensor") -> "TracedTensor":
        """Copy values from a TracedTensor into a regular tensor, returning a TracedTensor.
        
        This is useful when you have a pre-allocated buffer (regular tensor) and want
        to copy traced values into it while maintaining the trace. The returned
        TracedTensor wraps the target tensor but inherits the tracing graph from source.
        
        Usage:
            # Instead of: self._action[:] = action.to(self.device)
            # Use:        self._action = TracedTensor.copy_into(self._action, action.to(self.device))
        
        Args:
            target: The destination tensor (regular torch.Tensor) to copy into
            source: The source TracedTensor whose values and trace to use
            
        Returns:
            A TracedTensor wrapping the target tensor with source's proxy
        """
        if not isinstance(source, TracedTensor):
            raise TypeError(f"source must be a TracedTensor, got {type(source)}")
        
        # Copy the actual data
        target.copy_(source.tensor)
        
        # Return a new TracedTensor that wraps target but uses source's proxy
        # This effectively makes target "become" traced
        return TracedTensor(
            target, 
            source.name, 
            source._context, 
            source.proxy
        )

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
        if not self.validate_status():
            return result_tensor

        # For type conversions, we track it as an operation
        proxy_out = self._context.tracer.create_proxy(
            "call_method", "to", (self._proxy,) + args, kwargs
        )
        return self._new(result_tensor, proxy_out)

    def numpy(self):
        """Return self to maintain tracing through numpy operations.
        
        Instead of actually converting to numpy, we return the TracedTensor itself.
        This allows numpy operations (np.sin, np.sum, etc.) to be intercepted by
        __array_ufunc__ and __array_function__, which convert them to torch equivalents.
        
        This enables transparent tracing through code that uses numpy:
            x = traced_tensor.numpy()  # Returns TracedTensor
            y = np.sin(x)              # Intercepted → torch.sin(x) → TracedTensor
        
        Returns:
            TracedTensor: self, to maintain tracing continuity
        """
        return self

    def __array__(self, dtype=None):
        """NumPy array protocol - return self to maintain tracing.
        
        When np.array(traced_tensor) or np.asarray(traced_tensor) is called,
        we return self to keep the tracing chain intact. Subsequent numpy
        operations will be intercepted by __array_ufunc__/__array_function__.
        
        Args:
            dtype: Ignored (TracedTensor maintains its torch dtype)
            
        Returns:
            TracedTensor: self, to maintain tracing continuity
        """
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """Handle numpy ufuncs by converting to torch equivalents.
        
        This allows operations like np.sin(traced_tensor) to be traced by
        converting them to torch.sin(traced_tensor).
        
        Args:
            ufunc: The numpy ufunc being called
            method: The ufunc method ('__call__', 'reduce', etc.)
            *inputs: Input arguments to the ufunc
            out: Output array (not supported for tracing)
            **kwargs: Additional keyword arguments
            
        Returns:
            TracedTensor or result of the torch operation
        """
        # Only support __call__ method (not reduce, accumulate, etc.)
        if method != '__call__':
            _get_logger().warning(
                f"NumPy ufunc method '{method}' not supported for TracedTensor. "
                f"Only '__call__' is supported. Falling back to numpy."
            )
            return NotImplemented
        
        # Don't support out parameter as it breaks tracing
        if out is not None:
            _get_logger().warning(
                "NumPy ufunc 'out' parameter not supported for TracedTensor tracing."
            )
            return NotImplemented
        
        # Look up the torch equivalent
        torch_func = get_torch_equivalent_ufunc(ufunc)
        if torch_func is None:
            _get_logger().warning(
                f"NumPy ufunc '{ufunc.__name__}' has no torch equivalent. "
                f"Consider using torch.{ufunc.__name__} directly if available."
            )
            return NotImplemented
        
        # Convert inputs: numpy arrays -> tensors, keep TracedTensors
        converted_inputs = [convert_numpy_arg_to_torch(inp, self.device) for inp in inputs]
        
        # Call the torch function - will be traced via __torch_function__
        return torch_func(*converted_inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Handle numpy functions by converting to torch equivalents.
        
        This allows operations like np.sum(traced_tensor) to be traced by
        converting them to torch.sum(traced_tensor).
        
        Args:
            func: The numpy function being called
            types: Types of the arguments
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            TracedTensor or result of the torch operation
        """
        # Look up the torch equivalent
        torch_func = get_torch_equivalent_func(func)
        if torch_func is None:
            _get_logger().warning(
                f"NumPy function '{func.__name__}' has no torch equivalent. "
                f"Consider using torch.{func.__name__} directly if available."
            )
            return NotImplemented
        
        # Convert args: numpy arrays -> tensors, keep TracedTensors
        converted_args = [convert_numpy_arg_to_torch(arg, self.device) for arg in args]
        converted_kwargs = {k: convert_numpy_arg_to_torch(v, self.device) for k, v in kwargs.items()}
        
        # Handle axis -> dim conversion for functions that need it
        if 'axis' in converted_kwargs and torch_func in AXIS_TO_DIM_FUNCTIONS:
            converted_kwargs['dim'] = converted_kwargs.pop('axis')
        
        # Handle expand_dims axis -> dim (special case: axis is positional arg)
        if func == np.expand_dims and len(converted_args) >= 2:
            # np.expand_dims(a, axis) -> torch.unsqueeze(a, dim)
            converted_kwargs['dim'] = converted_args[1]
            converted_args = [converted_args[0]]
        
        # Call the torch function - will be traced via __torch_function__
        return torch_func(*converted_args, **converted_kwargs)
