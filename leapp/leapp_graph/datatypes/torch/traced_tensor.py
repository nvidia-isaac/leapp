#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""TracedTensor - A tensor subclass that records operations using torch.fx.

This class inherits from both TracedData (for tracing infrastructure) and
torch.Tensor (for tensor operations), enabling seamless operation interception
and graph recording.
"""

import operator
from abc import ABCMeta

import torch
from torch.fx.proxy import Proxy
import io as _io
from torch._export.converter import TS2EPConverter

from leapp.utils.logging import _get_logger
from leapp.utils.dtype import DtypeCodec, register_dtype_codec
from ..traced_data import TracedData


# torch dtype object -> common name string. Lives with the torch node library
# so the backend's dtype knowledge is unified with its implementation; the
# registry lets leapp core resolve dtypes without importing torch directly.
_TORCH_DTYPE_TO_NAME = {
    torch.float64: "float64",
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

register_dtype_codec(DtypeCodec(
    backend="torch",
    matches=lambda v: isinstance(v, torch.Tensor),
    value_dtype=lambda v: v.dtype,
    dtype_to_name=_TORCH_DTYPE_TO_NAME,
))



# Combined metaclass to resolve conflict between ABCMeta (from TracedData)
# and torch.Tensor's metaclass. In the future if this breaks, we need to convert
# the TracedTensor class to not inherit from tracedData and reimplement all the common functions
class _TracedTensorMeta(ABCMeta, type(torch.Tensor)):
    """Metaclass combining ABCMeta and torch.Tensor's metaclass."""
    pass


class TracedTensor(TracedData, torch.Tensor, metaclass=_TracedTensorMeta):
    """A tensor subclass that records operations using torch.fx.

    This class inherits from both TracedData (for tracing infrastructure)
    and torch.Tensor (for native tensor behavior). It records all operations
    performed on it by maintaining a computation graph via torch.fx.Proxy.

    TracedTensors must be created via TraceContext.create_input().
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, name: str, context, proxy: Proxy):
        """Create a new TracedTensor instance.
        
        Args:
            tensor: The actual torch.Tensor data to wrap
            name: Name for the tensor (used in ONNX export and graph)
            context: The TraceContext that owns this tensor
            proxy: The fx.Proxy for graph recording
            
        Returns:
            A new TracedTensor instance sharing storage with the input tensor
        """
        # Create a tensor subclass that shares storage with the input tensor
        instance = torch.Tensor._make_subclass(cls, tensor)
        return instance

    def __init__(self, tensor: torch.Tensor, name: str, context, proxy: Proxy):
        """Initialize a TracedTensor.

        Note: We don't call TracedData.__init__ because:
        1. We inherit from torch.Tensor, so 'self' IS the tensor data
        2. TracedData expects _value, but we use the tensor data directly

        Args:
            tensor: The actual torch.Tensor (data is shared via __new__)
            name: Name for the tensor (used in ONNX export and graph)
            context: The TraceContext that owns this tensor
            proxy: The fx.Proxy for graph recording
        """
        self._init_tracing_state(name, context, proxy)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying torch.Tensor (for compatibility with original API)."""
        return self.as_subclass(torch.Tensor)
    
    @property
    def data(self) -> torch.Tensor:
        """Get the underlying data."""
        return self.tensor

    @property
    def T(self):
        """Transpose property (for 2D tensors)."""
        return torch.transpose(self, 0, 1)

    # =========================================================================
    # Abstract method implementations from TracedData
    # =========================================================================

    def _new(self, tensor: torch.Tensor, proxy: Proxy = None) -> "TracedTensor":
        """Create a new TracedTensor in the same context.

        Intermediate tensors get auto-generated names based on the operation.
        When not tracing, proxy can be None.
        """
        intermediate_name = self._name_from_proxy(proxy)
        return TracedTensor(tensor, intermediate_name, self._context, proxy)

    # =========================================================================
    # Static Methods
    # =========================================================================

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

                if context is not None:
                    _get_logger().debug(
                        f"while tracing for {context} detected method_descriptor or builtin_function_or_method {func_name} "
                        f"but failed to find equivalent torch function. "
                        f"downstream _rewrite_method_descriptors should handle this by converting to call_method"
                    )
        return func

    @staticmethod
    def _safe_tensor_version(tensor: torch.Tensor | None) -> int | None:
        # Used to detect in-place mutations.
        if tensor is None:
            return None
        try:
            return tensor._version
        except RuntimeError:
            # Inference tensors may not expose version counters.
            return None

    @staticmethod
    def find_traced_tensor(obj):
        """Find the first TracedTensor in a supported nested structure."""
        found = None

        def visit(item):
            nonlocal found
            if found is None and isinstance(item, TracedTensor):
                found = item
            return item

        TracedData._map_structure(obj, visit)
        return found

    @staticmethod
    def unwrap_traced_tensor(obj):
        """Recursively unwrap TracedTensors to get raw tensors."""
        return TracedData._map_structure(
            obj,
            lambda item: item.tensor if isinstance(item, TracedTensor) else item,
        )

    @staticmethod
    def find_all_contexts(obj, contexts=None):
        """Recursively find all unique context names."""
        return TracedData.find_all_contexts(obj, contexts)

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
            _get_logger().fatal(
                f"source must be a TracedTensor, got {type(source)}",
                error_type=TypeError,
            )

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
    
    @staticmethod
    def _handle_TS_decomposition(func, real_args, real_kwargs):
        """Handle TorchScript decomposition."""
        # Decompose the TorchScript module into individual aten ops via
        # TS2EPConverter, then replay with TracedTensors so each op gets
        # recorded in our FX graph transparently.

        _get_logger().info(
            f"Detected TorchScript call to {type(func).__name__}. "
            f"Decomposing via TS2EPConverter for transparent tracing."
        )

        try:
            # Obtain a fully-initialized RecursiveScriptModule from the
            # ScriptMethod.  func.owner returns the C++ ScriptModule
            # (not an nn.Module).  Save → reload via BytesIO so
            # torch.jit.load performs full Python-level init.
            script_module = getattr(func, '__self__', None)
            if not isinstance(script_module, torch.nn.Module):
                cpp_module = getattr(func, 'owner', None)
                if cpp_module is not None:
                    _buf = _io.BytesIO()
                    _tmp = torch.jit.RecursiveScriptModule._construct(
                        cpp_module, lambda self: None
                    )
                    torch.jit.save(_tmp, _buf)
                    _buf.seek(0)
                    _dev = next(
                        (a.device for a in real_args
                            if isinstance(a, torch.Tensor)),
                        torch.device('cpu'),
                    )
                    script_module = torch.jit.load(
                        _buf, map_location=_dev
                    )

            #validate the script_module
            if not isinstance(script_module, torch.nn.Module):
                _get_logger().fatal(
                    f"Could not obtain nn.Module from {type(func).__name__}. "
                    f"func.__self__={type(getattr(func, '__self__', None))}, "
                    f"func.owner={type(getattr(func, 'owner', None))}",
                    error_type=RuntimeError,
                )

            # Clone+detach real_args so they are plain tensors with no
            # shared storage back to TracedTensors.  Without this,
            # as_subclass views can trigger __torch_function__ inside
            # torch.export internals.
            _clean = tuple(
                a.clone().detach() if isinstance(a, torch.Tensor) else a
                for a in real_args
            )
            _clean_kw = {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in (real_kwargs or {}).items()
            }
            ep = TS2EPConverter(
                script_module, _clean, _clean_kw
            ).convert()
            fx_module = ep.module()

        except Exception as e:
            _get_logger().warning(
                f"Failed to decompose TorchScript {type(func).__name__} via "
                f"TS2EPConverter: {e}\n"
                f"To resolve, consider one of:\n"
                f"  1. Use the regular (non-scripted) nn.Module instead "
                f"during tracing\n"
                f"  2. Extract and call the underlying operations "
                f"directly\n"
                f"  3. Break the chain by calling output_tensors() first "
                f"then annotate the scripted module usage with other "
                f"LEAPP API"
            )
            fx_module = None

        # Replay the decomposed FX module with TracedTensors.
        # Each aten op triggers __torch_function__, recording it in our graph.
        return fx_module

    # =========================================================================
    # Torch Function Interception
    # =========================================================================

    @staticmethod
    def _is_complete_slice(key):
        """Return whether key is an open ``slice(None)`` without comparing tensors."""
        return (
            isinstance(key, slice)
            and key.start is None
            and key.stop is None
            and key.step is None
        )

    @staticmethod
    def _is_full_assignment_key(key):
        """Return whether an index covers the complete destination tensor."""
        if key is Ellipsis:
            return True
        if isinstance(key, slice):
            return TracedTensor._is_complete_slice(key)
        return (
            isinstance(key, tuple)
            and key
            and all(TracedTensor._is_complete_slice(item) for item in key)
        )

    @classmethod
    def _promote_plain_tensor(cls, target, anchor, proxy):
        """Attach tracing state to an existing plain tensor object."""
        target.__class__ = cls
        target._init_tracing_state(anchor.name, anchor.context_obj, proxy)
        return target

    def _record_assignment(self, key, value, real_value):
        """Record one functional assignment and update this object's proxy."""
        # Plain full replacement may already have promoted with the source proxy.
        if (
            isinstance(value, TracedTensor)
            and self._proxy is value.proxy
            and self._is_full_assignment_key(key)
        ):
            return True

        value_proxy = value.proxy if isinstance(value, TracedTensor) else value
        return self._update_setitem_proxy(
            key, value_proxy, real_value=real_value
        )

    @classmethod
    def _handle_plain_assignment(cls, func, args=(), kwargs=None):
        """Promote a plain destination, then re-enter ``__setitem__`` / ``copy_``.

        Full replacement with a matching traced source reuses that source's
        proxy. Partial writes register the pre-write destination as a graph
        constant and use it as the promoted proxy.
        """
        kwargs = kwargs or {}
        func_name = getattr(func, "__name__", "")
        is_setitem = func_name == "__setitem__" and len(args) >= 3
        is_copy = func_name == "copy_" and len(args) >= 2
        if not (is_setitem or is_copy):
            return False, None

        target = args[0]
        if type(target) is not torch.Tensor:
            return False, None

        key = args[1] if is_setitem else Ellipsis
        value = args[2] if is_setitem else args[1]
        anchor = TracedTensor.find_traced_tensor((key, value))
        if anchor is None or not cls._is_supported_index_key(key):
            return False, None
        anchor.validate_status((key, value))

        if (
            isinstance(value, TracedTensor)
            and cls._is_full_assignment_key(key)
            and tuple(target.shape) == tuple(value.shape)
            and target.dtype == value.dtype
        ):
            dest_proxy = value.proxy
        else:
            dest_proxy = anchor._register_setitem_tensor(
                target.clone().detach(), "_setitem_destination"
            )
        cls._promote_plain_tensor(target, anchor, dest_proxy)
        if is_copy:
            return True, target.copy_(
                value, non_blocking=kwargs.get("non_blocking", False)
            )
        target[key] = value
        return True, None

    @classmethod
    def _handle_scripted_call(
        cls,
        func,
        traced_tensor,
        real_args,
        real_kwargs,
        tensor_out,
        args=(),
        kwargs=None,
    ):
        """Handle TorchScript calls by decomposing or rewrapping outputs.

        Returns:
            tuple[bool, object]: (handled, result). When handled is True, result
            should be returned directly from __torch_function__.
        """
        if kwargs is None:
            kwargs = {}

        func_type_name = type(func).__name__
        func_module = type(func).__module__ if hasattr(type(func), '__module__') else ''

        is_scripted = ('ScriptMethod' in func_type_name or 'ScriptModule' in func_type_name)

        if not is_scripted:
            if func_module.startswith('torch.jit') or func_module.startswith('torch._C'):
                if hasattr(func, '__self__') and hasattr(func.__self__, '__class__'):
                    self_class_name = func.__self__.__class__.__name__
                    is_scripted = 'Script' in self_class_name

        if not is_scripted:
            return False, None

        fx_module = TracedTensor._handle_TS_decomposition(func, real_args, real_kwargs)
        if fx_module is not None:
            return True, fx_module(*args, **kwargs)

        if isinstance(tensor_out, (tuple, list)):
            result = []
            for t in tensor_out:
                if isinstance(t, torch.Tensor):
                    result.append(traced_tensor._new(t, traced_tensor.proxy))
                else:
                    result.append(t)
            return True, type(tensor_out)(result)
        if isinstance(tensor_out, torch.Tensor):
            return True, traced_tensor._new(tensor_out, traced_tensor.proxy)
        return True, tensor_out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Intercept torch operations to record them in the graph.

        This method is called by PyTorch whenever a torch function is applied
        to this object. We execute the operation on the real tensor and record
        it in the FX graph.
        """
        if kwargs is None:
            kwargs = {}

        # TODO: revise numpy code when implementing numpy side of tracing
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


        # Handles situations where the destination is a plain tensor but a traced object 
        # is in the key or value.
        # Example:
        #   torch_tensor[0:3] = traced_tensor
        #   torch_tensor[traced_tensor] = plain_tensor

        handled, result = cls._handle_plain_assignment(func, args, kwargs)
        if handled:
            return result

        # Extract real tensors for actual computation
        real_args = tuple(TracedTensor.unwrap_traced_tensor(arg) for arg in args)
        real_kwargs = {k: TracedTensor.unwrap_traced_tensor(v) for k, v in kwargs.items()}
        receiver = args[0] if args and isinstance(args[0], TracedTensor) else None
        real_receiver = real_args[0] if receiver is not None else None
        receiver_version = cls._safe_tensor_version(real_receiver)

        # ================== EXECUTE THE ACTUAL OPERATION ==================
        tensor_out = func(*real_args, **real_kwargs)
        # ================== END OF EXECUTE THE ACTUAL OPERATION ===========
        receiver_was_mutated = (
            receiver_version is not None
            and cls._safe_tensor_version(real_receiver) != receiver_version
        )

        # Skip tracing if context is not tracing - return raw tensors
        if not traced_tensor.validate_status(args, kwargs):
            return tensor_out

        # ================== SPECIAL CASES IN HANDLING ==================
        handled, result = cls._handle_scripted_call(
            func,
            traced_tensor,
            real_args,
            real_kwargs,
            tensor_out,
            args,
            kwargs,
        )
        if handled:
            return result

        # ================== SPECIAL CASES IN HANDLING ==================

        def extract_proxy_leaf(item):
            if isinstance(item, TracedTensor):
                return item.proxy
            if isinstance(item, torch.nn.Parameter):
                # Inline parameters as constants for ONNX/JIT freezing.
                return item.data
            return item

        # Extract proxies for graph recording.
        proxy_args = TracedData._map_structure(args, extract_proxy_leaf)
        proxy_kwargs = TracedData._map_structure(kwargs, extract_proxy_leaf)

        # Record the operation in the graph
        proxy_out = traced_tensor._context.tracer.create_proxy(
            "call_function", func, proxy_args, proxy_kwargs
        )

        if (
            receiver_was_mutated
            and isinstance(tensor_out, torch.Tensor)
            and tensor_out is real_receiver
        ):
            receiver._init_tracing_state(
                receiver._name_from_proxy(proxy_out),
                receiver.context_obj,
                proxy_out,
            )
            return receiver

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

    # =========================================================================
    # Attribute Access
    # =========================================================================

    def __getattr__(self, name: str):
        """Forward unknown attributes to torch functions or tensor methods.

        This allows tensor methods like x.sum() to automatically work by
        forwarding to torch.sum(x), which then gets intercepted by
        __torch_function__ for graph recording.

        For attributes that don't have corresponding torch functions,
        we forward to the underlying tensor's attribute.
        """
        # Check __dict__ first for dynamically added attributes (e.g., leapp_tag)
        # This is needed because torch.Tensor subclasses may not follow normal
        # attribute lookup for custom attributes
        if name in self.__dict__:
            return self.__dict__[name]

        # Handle in-place operations (methods ending with _)
        if name.endswith('_') and not name.startswith('_') and len(name) > 1:
            base_name = name[:-1]  # Remove trailing underscore
            torch_func = getattr(torch, base_name, None)

            if torch_func is not None and callable(torch_func):
                def inplace_method(*args, **kwargs):
                    if not self.is_tracing:
                        # Just call the in-place method on the tensor and return raw tensor
                        underlying = self.as_subclass(torch.Tensor)
                        method = getattr(underlying, name, None)
                        if method is not None:
                            method(*TracedData.unwrap_traced_data(args), **TracedData.unwrap_traced_data(kwargs))
                        return underlying

                    # Record as functional operation
                    result = torch_func(self, *args, **kwargs)

                    # Update internal state to maintain in-place semantics
                    if isinstance(result, TracedTensor):
                        with torch.no_grad():
                            self.copy_(result.tensor)
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

        # Fall back to the underlying tensor's attribute via parent class
        try:
            attr = super().__getattr__(name)
        except AttributeError:
            attr = None

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

    # =========================================================================
    # Special Methods: reshape and permute need custom handling
    # =========================================================================

    @staticmethod
    def _normalize_shape_args(args):
        """Normalize Tensor method shape/dim args for torch function calls."""
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return args

    def reshape(self, *shape):
        """Reshape the tensor.

        Note: torch.reshape expects (tensor, shape) but tensor.reshape
        allows both reshape(shape) and reshape(*shape), so we handle both.
        """
        shape = self._normalize_shape_args(shape)
        return torch.reshape(self, shape)

    def permute(self, *dims):
        """Permute dimensions.

        Note: torch.permute expects (tensor, dims) but tensor.permute
        allows both permute(dims) and permute(*dims), so we handle both.
        """
        dims = self._normalize_shape_args(dims)
        return torch.permute(self, dims)

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other):
        """Addition operator."""
        return torch.add(self, other)

    def __radd__(self, other):
        """Reverse addition operator."""
        # Keep commutative reverse ops tensor-first for exporter compatibility.
        return torch.add(self, other)

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
        # Keep commutative reverse ops tensor-first for exporter compatibility.
        return torch.mul(self, other)

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

    # =========================================================================
    # In-place Arithmetic Operators (for +=, -=, etc.)
    # =========================================================================

    def __iadd__(self, other):
        """In-place addition (+=)."""
        if not self.is_tracing:
            # Get underlying tensor and call add_ directly on it
            # This bypasses __torch_function__ which would convert add_ to torch.add (non-in-place)
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.add_(unwrapped)
            return underlying

        # Record as functional operation in graph
        result = torch.add(self, other)

        # Update internal state to maintain in-place semantics
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy

        return self

    def __isub__(self, other):
        """In-place subtraction (-=)."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.sub_(unwrapped)
            return underlying

        result = torch.sub(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def __imul__(self, other):
        """In-place multiplication (*=)."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.mul_(unwrapped)
            return underlying

        result = torch.mul(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def __itruediv__(self, other):
        """In-place division (/=)."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.div_(unwrapped)
            return underlying

        result = torch.div(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def __ipow__(self, other):
        """In-place power (**=)."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.pow_(unwrapped)
            return underlying

        result = torch.pow(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def __imatmul__(self, other):
        """In-place matrix multiplication (@=)."""
        if not self.is_tracing:
            # matmul doesn't have a direct in-place version, compute and copy
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            result_data = torch.matmul(underlying, unwrapped)
            underlying.copy_(result_data)
            return underlying

        result = torch.matmul(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    # =========================================================================
    # In-place Methods (add_, mul_, etc.) - Must override since inherited from torch.Tensor
    # =========================================================================

    def add_(self, other, *, alpha=1):
        """In-place addition method."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.add_(unwrapped, alpha=alpha)
            return underlying
        # Record as functional operation
        result = torch.add(self, other, alpha=alpha)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def sub_(self, other, *, alpha=1):
        """In-place subtraction method."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.sub_(unwrapped, alpha=alpha)
            return underlying

        result = torch.sub(self, other, alpha=alpha)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def mul_(self, other):
        """In-place multiplication method."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.mul_(unwrapped)
            return underlying

        result = torch.mul(self, other)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def div_(self, other, *, rounding_mode=None):
        """In-place division method."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(other)
            underlying = self.as_subclass(torch.Tensor)
            underlying.div_(unwrapped, rounding_mode=rounding_mode)
            return underlying

        result = torch.div(self, other, rounding_mode=rounding_mode)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def pow_(self, exponent):
        """In-place power method."""
        if not self.is_tracing:
            unwrapped = TracedData.unwrap_traced_data(exponent)
            underlying = self.as_subclass(torch.Tensor)
            underlying.pow_(unwrapped)
            return underlying

        result = torch.pow(self, exponent)
        with torch.no_grad():
            torch.Tensor.copy_(self, result.tensor if isinstance(result, TracedTensor) else result)
        if isinstance(result, TracedTensor):
            self._proxy = result.proxy
        return self

    def copy_(self, src, non_blocking=False):
        """Copy ``src`` into this tensor and record a full functional write."""
        real_value = TracedTensor.unwrap_traced_tensor(src)
        self.tensor.copy_(real_value, non_blocking=non_blocking)

        if not self.validate_status((Ellipsis, src)):
            return self

        if not self._record_assignment(Ellipsis, src, real_value):
            _get_logger().warning(
                "TracedTensor.copy_ cannot be lowered functionally; "
                "recording raw __setitem__, which may not export."
            )
            value_proxy = src.proxy if isinstance(src, TracedTensor) else src
            self._proxy = self._context.tracer.create_proxy(
                "call_method", "__setitem__",
                (self._proxy, Ellipsis, value_proxy), {},
            )
        return self


    # =========================================================================
    # Comparison Operators
    # =========================================================================

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

    # =========================================================================
    # Indexing Operations
    # =========================================================================

    def __getitem__(self, key):
        """Indexing operator for TracedTensor.

        Supports all Python indexing operations: slicing, integer indexing,
        tensor indexing, etc. The operation is recorded in the computation graph.

        Whole-key traced masks/indices retain their specialized lowerings.
        Tuple/mixed traced indices are converted to FX proxies and recorded as
        operator.getitem so augmented assignment can write the result back.
        """
        # Check for boolean mask indexing with TracedTensor
        if isinstance(key, TracedTensor):
            result_tensor = self.tensor[key.tensor]

            # Skip tracing if context is not tracing - return raw tensor
            if not self.validate_status(args=(key,)):
                return result_tensor

            # Check if it's a boolean tensor (mask)
            if key.dtype == torch.bool:
                proxy_out = self._context.tracer.create_proxy(
                    "call_function", torch.masked_select, (self._proxy, key.proxy), {}
                )
                return self._new(result_tensor, proxy_out)

            if key.dtype not in (torch.int32, torch.int64):
                raise NotImplementedError(
                    "Advanced indexing with TracedTensor indices is only auto-lowered "
                    "for integer index tensors. "
                    f"Received dtype {key.dtype}."
                )

            flat_index_proxy = self._context.tracer.create_proxy(
                "call_method", "reshape", (key.proxy, (-1,)), {}
            )
            proxy_out = self._context.tracer.create_proxy(
                "call_function", torch.index_select, (self._proxy, 0, flat_index_proxy), {}
            )

            if key.ndim > 1:
                output_shape = tuple(key.shape) + tuple(self.shape[1:])
                proxy_out = self._context.tracer.create_proxy(
                    "call_method", "reshape", (proxy_out, output_shape), {}
                )
            return self._new(result_tensor, proxy_out)

        if TracedData.find_traced_data(key) is not None:
            if not self._is_supported_index_key(key):
                _get_logger().fatal(
                    "Mixed traced indexing supports boolean and integer "
                    "tensor indices combined with basic Python indices.",
                    error_type=NotImplementedError,
                )
            real_key = TracedData.unwrap_traced_data(key)
            result_tensor = self.tensor[real_key]
            if not self.validate_status(args=(key,)):
                return result_tensor
            proxy_out = self._create_getitem_proxy(key)
            return self._new(result_tensor, proxy_out)

        result_tensor = self.tensor[key]

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
        """Indexed assignment with functional ``index_put`` lowering.

        Plain destinations may re-enter here after
        ``_handle_plain_assignment`` promotes them and installs a destination
        constant as this tensor's proxy.
        """
        real_key = TracedData.unwrap_traced_data(key)
        real_value = TracedTensor.unwrap_traced_tensor(value)
        self.tensor[real_key] = real_value

        if not self.validate_status((key, value)):
            return

        if not self._record_assignment(key, value, real_value):
            _get_logger().warning(
                f"TracedTensor assignment with key {key!r} cannot be lowered "
                "functionally; recording raw __setitem__, which may not export."
            )
            value_proxy = value.proxy if isinstance(value, TracedTensor) else value
            self._proxy = self._context.tracer.create_proxy(
                "call_method", "__setitem__", (self._proxy, key, value_proxy), {}
            )

    # =========================================================================
    # Magic Methods
    # =========================================================================

    def __len__(self) -> int:
        """Length operator for TracedTensor."""
        return self.tensor.__len__()

    def __str__(self) -> str:
        """String representation of TracedTensor."""
        return f"TracedTensor({self.tensor})"

    def __repr__(self) -> str:
        """String representation of TracedTensor."""
        return f"TracedTensor({self.tensor})"

    def __format__(self, format_spec: str) -> str:
        """Format the TracedTensor by delegating to the underlying tensor."""
        return self.tensor.__format__(format_spec)

    # =========================================================================
    # Type Conversion Methods
    # =========================================================================

    def to(self, *args, **kwargs):
        """Convert tensor to different dtype or device.

        Note: This requires special handling because .to() is a tensor method,
        not a torch.* function, so __torch_function__ doesn't intercept it.
        """
        result_tensor = self.tensor.to(*args, **kwargs)

        # Skip tracing if context is not tracing - return raw tensor
        if not self.validate_status():
            return result_tensor

        # For type conversions, we track it as an operation
        proxy_out = self._context.tracer.create_proxy(
            "call_method", "to", (self._proxy,) + args, kwargs
        )
        return self._new(result_tensor, proxy_out)

    def numpy(self):
        """Convert to numpy array, preserving tracing as TracedNpArray.
        
        If tracing is active, returns a TracedNpArray that shares the proxy
        and context, allowing the trace to continue across the conversion.
        
        If not tracing, returns a regular numpy array.
        
        Returns:
            TracedNpArray if tracing, else np.ndarray
        """
        np_data = self.as_subclass(torch.Tensor).detach().cpu().numpy()
        
        # Skip tracing if context is not tracing - return raw numpy array
        if not self.validate_status():
            return np_data
        
        # Return TracedNpArray with inherited proxy/context
        from leapp.leapp_graph.datatypes import as_traced  # lazy import to avoid circular dependency
        return as_traced(np_data, self.name, self.context_obj, self.proxy)

    def __array__(self, dtype=None, copy=None):
        """NumPy array protocol - called by np.array(), np.asarray(), etc.
        
        This allows seamless conversion to numpy while preserving tracing.
        
        Args:
            dtype: Optional dtype for the resulting array
            copy: Optional copy flag (NumPy 2.0+)
                - None: Copy only if necessary (default)
                - True: Always copy
                - False: Never copy; raise ValueError if copy required
            
        Returns:
            TracedNpArray if tracing, else np.ndarray
        """
        result = self.numpy()
        
        # Check if dtype change requires a copy
        needs_dtype_copy = dtype is not None and dtype != result.dtype
        
        # copy=False but copy is required → raise error (NumPy 2.0 semantics)
        if copy is False and needs_dtype_copy:
            _get_logger().fatal(
                f"Unable to avoid copy while creating an array with dtype {dtype} "
                f"from array with dtype {result.dtype}.",
                error_type=ValueError,
            )
        
        if needs_dtype_copy:
            result = result.astype(dtype)
        
        if copy is True:
            result = result.copy()
        
        return result