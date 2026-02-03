#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""TracedNpArray - A numpy array subclass that records operations using torch.fx.

This class inherits from both TracedData (for tracing infrastructure) and
np.ndarray (for native array behavior). It records all operations performed
on it by maintaining a computation graph via torch.fx.Proxy. The recorded
graph uses torch equivalents of numpy operations for export compatibility.
"""

import operator
from abc import ABCMeta

import numpy as np
import torch
from torch.fx.proxy import Proxy

from leapp._logging import _get_logger
from leapp.tracing_lock import TracingLock
from .traced_data import TracedData
from .numpy_compatibility import (
    AXIS_TO_DIM_FUNCTIONS,
    get_torch_equivalent_ufunc,
    get_torch_equivalent_func,
)


# Combined metaclass to resolve conflict between ABCMeta (from TracedData)
# and np.ndarray's metaclass
class _TracedNpArrayMeta(ABCMeta, type(np.ndarray)):
    """Metaclass combining ABCMeta and np.ndarray's metaclass."""
    pass


class TracedNpArray(TracedData, np.ndarray, metaclass=_TracedNpArrayMeta):
    """A numpy array subclass that records operations using torch.fx.

    This class inherits from both TracedData (for tracing infrastructure)
    and np.ndarray (for native array behavior). It records all operations
    performed on it by maintaining a computation graph via torch.fx.Proxy.

    TracedNpArrays must be created via TraceContext.create_input().
    """

    def __new__(cls, array: np.ndarray, name: str, context, proxy: Proxy):
        """Create a new TracedNpArray instance.

        Args:
            array: The actual numpy array data
            name: Name for the array (used in export and graph)
            context: The TraceContext that owns this array
            proxy: The fx.Proxy for graph recording

        Returns:
            A new TracedNpArray instance as a view of the input array
        """
        # Create a view of the input array as our subclass
        obj = np.asarray(array).view(cls)
        # Set our custom attributes
        obj._name = name
        obj._context = context
        obj._proxy = proxy
        obj._global_tracing_lock = TracingLock()
        return obj

    def __array_finalize__(self, obj):
        """Finalize array creation - called for views and new-from-template.

        This is called whenever a new array is created from an existing
        TracedNpArray (e.g., slicing, view casting). We need to copy our
        custom attributes to the new array.
        """
        if obj is None:
            # Called from __new__, attributes already set
            return
        # Copy attributes from the source array
        self._name = getattr(obj, '_name', 'derived')
        self._context = getattr(obj, '_context', None)
        self._proxy = getattr(obj, '_proxy', None)
        self._global_tracing_lock = getattr(obj, '_global_tracing_lock', None)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def tensor(self) -> np.ndarray:
        """Get the underlying numpy array (for TracedData compatibility)."""
        return torch.from_numpy(self.view(np.ndarray))

    @property
    def data(self) -> np.ndarray:
        """Get the underlying datatype."""
        return self.view(np.ndarray)

    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy

    @property
    def name(self) -> str:
        """Get the name of the array."""
        return self._name

    @property
    def context(self) -> str:
        """Get the name of the context that owns this array."""
        if self._context is None:
            return "untraced"
        return self._context.name

    @property
    def context_obj(self):
        """Get the TraceContext that owns this array."""
        return self._context

    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the context."""
        if self._context is None:
            return False
        return self._context.is_tracing

    # =========================================================================
    # TracedData abstract method implementations
    # =========================================================================

    def _unwrap(self) -> np.ndarray:
        """Get the underlying raw array."""
        return self.view(np.ndarray)

    # =========================================================================
    # Array Properties - inherited from np.ndarray
    # shape, dtype, ndim, size, T are all inherited automatically
    # =========================================================================

    # Override T to go through our tracing
    @property
    def T(self):
        """Transpose."""
        return np.transpose(self)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _new(self, array: np.ndarray, proxy: Proxy = None) -> "TracedNpArray":
        """Create a new TracedNpArray in the same context."""
        if proxy is not None:
            intermediate_name = str(proxy.node.name)
        else:
            intermediate_name = "untraced"
        return TracedNpArray(array, intermediate_name, self._context, proxy)

    @staticmethod
    def unwrap_traced_array(obj):
        """Recursively unwrap TracedNpArrays to get underlying numpy arrays."""
        if isinstance(obj, TracedNpArray):
            return obj.view(np.ndarray)
        elif isinstance(obj, TracedData):
            return obj._unwrap()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(TracedNpArray.unwrap_traced_array(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: TracedNpArray.unwrap_traced_array(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def find_traced_array(obj):
        """Find first TracedNpArray in args."""
        if isinstance(obj, TracedNpArray):
            return obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = TracedNpArray.find_traced_array(item)
                if result is not None:
                    return result
        return None

    @staticmethod
    def find_all_contexts(obj, contexts=None):
        """Recursively find all unique context names from TracedNpArray instances."""
        if contexts is None:
            contexts = set()
        elif isinstance(obj, TracedData) and obj.is_tracing:
            contexts.add(obj.context)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                TracedNpArray.find_all_contexts(item, contexts)
        elif isinstance(obj, dict):
            for v in obj.values():
                TracedNpArray.find_all_contexts(v, contexts)
        return contexts
    def _extract_proxy(self, obj):
        """Recursively extract proxies for graph recording."""
        if isinstance(obj, TracedNpArray):
            return obj.proxy
        elif isinstance(obj, TracedData):
            return obj.proxy
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to torch tensor for the graph
            return torch.from_numpy(obj.copy())
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._extract_proxy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._extract_proxy(v) for k, v in obj.items()}
        return obj

    def _convert_numpy_kwargs_to_torch(self, proxy_kwargs, torch_func, original_kwargs=None, args=None):
        """Convert numpy kwargs to torch kwargs.
        
        Args:
            proxy_kwargs: Dict of kwargs with proxies extracted
            torch_func: The torch function being called
            original_kwargs: Original kwargs dict (before proxy extraction) for key checking
            args: Original args for getting array dimensions
        
        Handles:
        - 'axis' → 'dim' for functions in AXIS_TO_DIM_FUNCTIONS
        - 'keepdims' → 'keepdim' (numpy uses 's', torch doesn't)
        - 'keepdims' without axis: torch requires dim when keepdim is specified
        - 'axes' for transpose: None or missing → reversed dims tuple
        """
        proxy_kwargs = proxy_kwargs.copy()
        if original_kwargs is None:
            original_kwargs = {}
        
        # Get input array for dimension info
        input_arr = args[0] if args and len(args) > 0 else None
        
        # Convert axis → dim
        has_dim = False
        if torch_func in AXIS_TO_DIM_FUNCTIONS and 'axis' in proxy_kwargs:
            proxy_kwargs['dim'] = proxy_kwargs.pop('axis')
            has_dim = True
        
        # Convert keepdims → keepdim (numpy uses 's', torch doesn't)
        # Important: torch.sum/mean only support keepdim when dim is also specified
        if 'keepdims' in proxy_kwargs:
            keepdims_value = proxy_kwargs.pop('keepdims')
            if has_dim:
                # dim is specified, we can use keepdim
                proxy_kwargs['keepdim'] = keepdims_value
            elif keepdims_value and input_arr is not None and hasattr(input_arr, 'ndim'):
                # keepdims=True but no axis specified
                # torch requires dim when using keepdim, so specify all dims
                proxy_kwargs['dim'] = tuple(range(input_arr.ndim))
                proxy_kwargs['keepdim'] = True
            # If keepdims=False without axis, we can just omit both (default behavior)
        
        # Handle np.transpose with axes=None or missing
        # torch.permute requires explicit dims, but numpy reverses all dims when axes is omitted
        if torch_func == torch.permute:
            # Check if axes was passed as positional arg (2nd argument)
            axes_in_args = len(args) > 1 if args else False
            
            if 'axes' in proxy_kwargs:
                axes = proxy_kwargs.pop('axes')
                if axes is None and input_arr is not None and hasattr(input_arr, 'ndim'):
                    # axes=None means reverse all dimensions
                    proxy_kwargs['dims'] = tuple(range(input_arr.ndim - 1, -1, -1))
                elif axes is not None:
                    proxy_kwargs['dims'] = axes
            elif not axes_in_args and 'axes' not in original_kwargs:
                # axes not provided at all (neither positional nor kwarg) - numpy reverses all dims
                if input_arr is not None and hasattr(input_arr, 'ndim'):
                    proxy_kwargs['dims'] = tuple(range(input_arr.ndim - 1, -1, -1))
            # If axes_in_args is True, it will be passed as positional arg, don't add to kwargs
        
        return proxy_kwargs

    def validate_status(self, args=None, kwargs=None):
        """Validate that this TracedNpArray can be used in the current context."""
        if not self.is_tracing:
            return False
        if self.is_tracing and self._global_tracing_lock.is_active:
            _get_logger().error(
                f"Error: detected active TracedNpArray {self._name} from node {self.context} "
                f"inside of a traced function.\n"
                f"\n"
                f"This happens when you have an active TracedNpArray and it is being used "
                f"for computation inside of a traced function/block.\n"
                f"\n"
                f"You must call output_tensors() to finalize the TracedNpArray node first"
            )
            raise Exception(
                "Cannot use TracedNpArray inside of a traced function/block. "
                "Call output_tensors() first to finalize the TracedNpArray node"
            )

        # Check for multiple contexts in args/kwargs
        contexts = set()
        if args is not None:
            for arg in args:
                contexts = TracedNpArray.find_all_contexts(arg, contexts)
        if kwargs is not None:
            for kwarg in kwargs.values():
                contexts = TracedNpArray.find_all_contexts(kwarg, contexts)

        if len(contexts) > 1:
            _get_logger().error(
                f"Error: detected multiple TracedNpArray contexts: {contexts} inside of a traced function.\n"
                "\n"
                "This happens when you mix multiple active TracedNpArrays from different contexts "
                "inside of a traced function/block.\n"
                "\n"
                "You can call output_tensors() to finalize one of the TracedNpArray nodes first "
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
            raise Exception(
                "Cannot mix multiple active TracedNpArrays from different contexts inside of a traced function/block. "
                "Call output_tensors() to finalize one of the TracedNpArray nodes first "
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
        return True

    # =========================================================================
    # NumPy Ufunc Interception (__array_ufunc__)
    # =========================================================================

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept numpy ufuncs (element-wise operations).

        This is called when numpy ufuncs (np.sin, np.add, etc.) are applied
        to this TracedNpArray. We execute the numpy operation and record
        the torch equivalent in the graph.
        """
        if method != '__call__':
            # Only handle direct calls, not reduce/accumulate/etc.
            return NotImplemented

        # Get the torch equivalent
        torch_func = get_torch_equivalent_ufunc(ufunc)
        if torch_func is None:
            _get_logger().warning(
                f"No torch equivalent for numpy ufunc {ufunc.__name__}. "
                f"Operation will not be traced."
            )
            # Fall back to numpy-only execution
            unwrapped = tuple(TracedNpArray.unwrap_traced_array(inp) for inp in inputs)
            return ufunc(*unwrapped, **kwargs)

        # Execute the numpy operation
        unwrapped_inputs = tuple(TracedNpArray.unwrap_traced_array(inp) for inp in inputs)
        unwrapped_kwargs = {k: TracedNpArray.unwrap_traced_array(v) for k, v in kwargs.items()}
        result_array = ufunc(*unwrapped_inputs, **unwrapped_kwargs)

        # Find a TracedNpArray for context
        traced_array = TracedNpArray.find_traced_array(inputs)
        if traced_array is None:
            return result_array

        # Skip graph recording if not tracing
        if not traced_array.validate_status(inputs, kwargs):
            return result_array

        # Record the torch operation in the graph
        proxy_inputs = tuple(self._extract_proxy(inp) for inp in inputs)
        proxy_kwargs = self._convert_numpy_kwargs_to_torch(
            {k: self._extract_proxy(v) for k, v in kwargs.items()},
            torch_func,
            original_kwargs=kwargs,
            args=inputs
        )

        proxy_out = traced_array._context.tracer.create_proxy(
            "call_function", torch_func, proxy_inputs, proxy_kwargs
        )

        # Wrap result
        if isinstance(result_array, np.ndarray):
            return traced_array._new(result_array, proxy_out)
        return result_array

    # =========================================================================
    # NumPy Function Interception (__array_function__)
    # =========================================================================

    # List of numpy functions we implement
    HANDLED_FUNCTIONS = {}

    def __array_function__(self, func, types, args, kwargs):
        """Intercept numpy array functions.

        This is called when numpy functions (np.sum, np.concatenate, etc.)
        are applied to this TracedNpArray. We execute the numpy operation
        and record the torch equivalent in the graph.
        """
        # Get the torch equivalent
        torch_func = get_torch_equivalent_func(func)
        if torch_func is None:
            _get_logger().warning(
                f"No torch equivalent for numpy function {func.__name__}. "
                f"Operation will not be traced."
            )
            # Fall back to numpy-only execution
            unwrapped_args = tuple(TracedNpArray.unwrap_traced_array(arg) for arg in args)
            unwrapped_kwargs = {k: TracedNpArray.unwrap_traced_array(v) for k, v in kwargs.items()}
            return func(*unwrapped_args, **unwrapped_kwargs)

        # Execute the numpy operation
        unwrapped_args = tuple(TracedNpArray.unwrap_traced_array(arg) for arg in args)
        unwrapped_kwargs = {k: TracedNpArray.unwrap_traced_array(v) for k, v in kwargs.items()}
        result_array = func(*unwrapped_args, **unwrapped_kwargs)

        # Find a TracedNpArray for context
        traced_array = TracedNpArray.find_traced_array(args)
        if traced_array is None:
            return result_array

        # Skip graph recording if not tracing
        if not traced_array.validate_status(args, kwargs):
            return result_array

        # Record the torch operation in the graph
        proxy_args = tuple(self._extract_proxy(arg) for arg in args)
        proxy_kwargs = self._convert_numpy_kwargs_to_torch(
            {k: self._extract_proxy(v) for k, v in kwargs.items()},
            torch_func,
            original_kwargs=kwargs,
            args=args
        )

        proxy_out = traced_array._context.tracer.create_proxy(
            "call_function", torch_func, proxy_args, proxy_kwargs
        )

        # Handle multiple outputs (e.g., np.split returns a list)
        if isinstance(result_array, (tuple, list)):
            result = []
            for i, arr in enumerate(result_array):
                if isinstance(arr, np.ndarray):
                    item_proxy = traced_array._context.tracer.create_proxy(
                        "call_function", operator.getitem, (proxy_out, i), {}
                    )
                    result.append(traced_array._new(arr, item_proxy))
                else:
                    result.append(arr)
            return type(result_array)(result)
        elif isinstance(result_array, np.ndarray):
            return traced_array._new(result_array, proxy_out)
        return result_array

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return np.add(other, self)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return np.multiply(other, self)

    def __truediv__(self, other):
        return np.divide(self, other)

    def __rtruediv__(self, other):
        return np.divide(other, self)

    def __floordiv__(self, other):
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)

    def __mod__(self, other):
        return np.mod(self, other)

    def __rmod__(self, other):
        return np.mod(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __neg__(self):
        return np.negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return np.abs(self)

    # =========================================================================
    # In-place Arithmetic Operators
    # =========================================================================

    def __iadd__(self, other):
        result = np.add(self, other)
        if isinstance(result, TracedNpArray):
            # Copy data in-place since self IS the array
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    def __isub__(self, other):
        result = np.subtract(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    def __imul__(self, other):
        result = np.multiply(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    def __itruediv__(self, other):
        result = np.divide(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    def __ifloordiv__(self, other):
        result = np.floor_divide(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    def __ipow__(self, other):
        result = np.power(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy = result._proxy
            return self
        return result

    # =========================================================================
    # Comparison Operators
    # =========================================================================

    def __eq__(self, other):
        return np.equal(self, other)

    def __ne__(self, other):
        return np.not_equal(self, other)

    def __lt__(self, other):
        return np.less(self, other)

    def __le__(self, other):
        return np.less_equal(self, other)

    def __gt__(self, other):
        return np.greater(self, other)

    def __ge__(self, other):
        return np.greater_equal(self, other)

    # =========================================================================
    # Logical Operators
    # =========================================================================

    def __and__(self, other):
        return np.logical_and(self, other)

    def __rand__(self, other):
        return np.logical_and(other, self)

    def __or__(self, other):
        return np.logical_or(self, other)

    def __ror__(self, other):
        return np.logical_or(other, self)

    def __xor__(self, other):
        return np.logical_xor(self, other)

    def __rxor__(self, other):
        return np.logical_xor(other, self)

    def __invert__(self):
        return np.logical_not(self)

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    def __matmul__(self, other):
        return np.matmul(self, other)

    def __rmatmul__(self, other):
        return np.matmul(other, self)

    # =========================================================================
    # Indexing
    # =========================================================================

    def __getitem__(self, key):
        """Handle array indexing."""
        # Get result from underlying array
        result = self.view(np.ndarray)[key]

        if not self.validate_status():
            return result

        # Record indexing in graph
        proxy_key = self._extract_proxy(key) if not isinstance(key, (slice, type(None))) else key
        proxy_out = self._context.tracer.create_proxy(
            "call_function", operator.getitem, (self.proxy, proxy_key), {}
        )

        if isinstance(result, np.ndarray):
            return self._new(result, proxy_out)
        return result

    def __setitem__(self, key, value):
        """Handle array assignment."""
        unwrapped_value = TracedNpArray.unwrap_traced_array(value)
        # Assign to underlying array
        self.view(np.ndarray)[key] = unwrapped_value

        if not self.validate_status():
            return

        # Record setitem in graph (though this may not work well with fx)
        _get_logger().warning(
            "In-place assignment (setitem) on TracedNpArray may not trace correctly. "
            "Consider using functional operations instead."
        )

    # =========================================================================
    # Array Methods (delegate to numpy functions)
    # =========================================================================

    def sum(self, axis=None, **kwargs):
        return np.sum(self, axis=axis, **kwargs)

    def mean(self, axis=None, **kwargs):
        return np.mean(self, axis=axis, **kwargs)

    def std(self, axis=None, **kwargs):
        return np.std(self, axis=axis, **kwargs)

    def var(self, axis=None, **kwargs):
        return np.var(self, axis=axis, **kwargs)

    def min(self, axis=None, **kwargs):
        return np.min(self, axis=axis, **kwargs)

    def max(self, axis=None, **kwargs):
        return np.max(self, axis=axis, **kwargs)

    def argmin(self, axis=None):
        return np.argmin(self, axis=axis)

    def argmax(self, axis=None):
        return np.argmax(self, axis=axis)

    def cumsum(self, axis=None):
        return np.cumsum(self, axis=axis)

    def cumprod(self, axis=None):
        return np.cumprod(self, axis=axis)

    def all(self, axis=None):
        return np.all(self, axis=axis)

    def any(self, axis=None):
        return np.any(self, axis=axis)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return np.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            return np.transpose(self)
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return np.transpose(self, axes)

    def swapaxes(self, axis1, axis2):
        return np.swapaxes(self, axis1, axis2)

    def flatten(self):
        return np.reshape(self, (-1,))

    def ravel(self):
        return np.reshape(self, (-1,))

    def squeeze(self, axis=None):
        return np.squeeze(self, axis=axis)

    def clip(self, a_min, a_max):
        return np.clip(self, a_min, a_max)

    def round(self, decimals=0):
        return np.round(self, decimals)

    def astype(self, dtype):
        """Type conversion - executes but may not trace correctly."""
        result = self.view(np.ndarray).astype(dtype)
        if not self.validate_status():
            return result
        # Note: torch equivalent would be .to(dtype), but dtype mapping is complex
        _get_logger().warning(f"astype({dtype}) may not trace correctly to torch equivalent")
        return self._new(result, self.proxy)

    def copy(self):
        """Return a copy of the array."""
        result = self.view(np.ndarray).copy()
        if not self.validate_status():
            return result
        proxy_out = self._context.tracer.create_proxy(
            "call_function", torch.clone, (self.proxy,), {}
        )
        return self._new(result, proxy_out)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def __array__(self, dtype=None):
        """Convert to numpy array."""
        arr = self.view(np.ndarray)
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def tolist(self):
        """Convert to Python list."""
        return self.view(np.ndarray).tolist()

    def item(self):
        """Return scalar value."""
        return self.view(np.ndarray).item()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self):
        return f"TracedNpArray(name={self._name}, shape={self.shape}, dtype={self.dtype})"

    def __str__(self):
        return f"TracedNpArray({self.view(np.ndarray)})"
