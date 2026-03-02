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
from .traced_data import TracedData


# =============================================================================
# NumPy to Torch Mappings
# =============================================================================

# Ufuncs are element-wise operations that numpy broadcasts automatically.
# These are intercepted via __array_ufunc__ protocol.
NUMPY_UFUNC_TO_TORCH = {
    # Arithmetic operations
    np.add: torch.add,
    np.subtract: torch.sub,
    np.multiply: torch.mul,
    np.divide: torch.div,
    np.true_divide: torch.div,
    np.floor_divide: torch.floor_divide,
    np.power: torch.pow,
    np.negative: torch.neg,
    np.positive: lambda x: x,  # No-op
    np.mod: torch.remainder,
    np.remainder: torch.remainder,
    np.fmod: torch.fmod,

    # Absolute and sign
    np.absolute: torch.abs,
    np.abs: torch.abs,
    np.sign: torch.sign,

    # Powers and roots
    np.sqrt: torch.sqrt,
    np.square: torch.square,
    np.exp: torch.exp,
    np.exp2: lambda x: torch.pow(2, x),
    np.expm1: torch.expm1,

    # Logarithms
    np.log: torch.log,
    np.log2: torch.log2,
    np.log10: torch.log10,
    np.log1p: torch.log1p,

    # Trigonometric functions
    np.sin: torch.sin,
    np.cos: torch.cos,
    np.tan: torch.tan,
    np.arcsin: torch.asin,
    np.arccos: torch.acos,
    np.arctan: torch.atan,
    np.arctan2: torch.atan2,
    np.hypot: torch.hypot,

    # Hyperbolic functions
    np.sinh: torch.sinh,
    np.cosh: torch.cosh,
    np.tanh: torch.tanh,
    np.arcsinh: torch.asinh,
    np.arccosh: torch.acosh,
    np.arctanh: torch.atanh,

    # Rounding
    np.floor: torch.floor,
    np.ceil: torch.ceil,
    np.trunc: torch.trunc,
    np.round: torch.round,
    np.rint: torch.round,

    # Comparison (element-wise, return boolean tensor)
    np.greater: torch.gt,
    np.greater_equal: torch.ge,
    np.less: torch.lt,
    np.less_equal: torch.le,
    np.equal: torch.eq,
    np.not_equal: torch.ne,
    np.maximum: torch.maximum,
    np.minimum: torch.minimum,

    # Logical operations
    np.logical_and: torch.logical_and,
    np.logical_or: torch.logical_or,
    np.logical_xor: torch.logical_xor,
    np.logical_not: torch.logical_not,

    # Bitwise operations
    np.bitwise_and: torch.bitwise_and,
    np.bitwise_or: torch.bitwise_or,
    np.bitwise_xor: torch.bitwise_xor,
    np.invert: torch.bitwise_not,
    np.left_shift: torch.bitwise_left_shift,
    np.right_shift: torch.bitwise_right_shift,

    # Special values
    np.isnan: torch.isnan,
    np.isinf: torch.isinf,
    np.isfinite: torch.isfinite,

    # Clipping
    np.clip: torch.clamp,
}

# Higher-level array functions intercepted via __array_function__ protocol.
NUMPY_FUNC_TO_TORCH = {
    # Reduction operations
    np.sum: torch.sum,
    np.prod: torch.prod,
    np.mean: torch.mean,
    np.std: torch.std,
    np.var: torch.var,
    # Use amax/amin instead of max/min because torch.max/min with dim returns (values, indices)
    # while numpy just returns values. amax/amin always return just values.
    np.min: torch.amin,
    np.max: torch.amax,
    np.argmin: torch.argmin,
    np.argmax: torch.argmax,
    np.cumsum: torch.cumsum,
    np.cumprod: torch.cumprod,
    np.all: torch.all,
    np.any: torch.any,

    # Array manipulation
    np.concatenate: torch.cat,
    np.stack: torch.stack,
    np.vstack: torch.vstack,
    np.hstack: torch.hstack,
    np.split: torch.split,
    np.array_split: torch.tensor_split,
    np.squeeze: torch.squeeze,
    np.expand_dims: torch.unsqueeze,
    np.reshape: torch.reshape,
    np.transpose: torch.permute,
    np.swapaxes: torch.swapaxes,
    np.moveaxis: torch.moveaxis,
    np.flip: torch.flip,
    np.roll: torch.roll,
    np.rot90: torch.rot90,

    # Sorting and searching
    np.sort: torch.sort,
    np.argsort: torch.argsort,
    np.where: torch.where,
    np.nonzero: torch.nonzero,

    # Element-wise (also available as ufuncs)
    np.clip: torch.clamp,
    np.abs: torch.abs,
    np.absolute: torch.abs,
    np.sqrt: torch.sqrt,
    np.square: torch.square,
    np.exp: torch.exp,
    np.log: torch.log,
    np.sin: torch.sin,
    np.cos: torch.cos,
    np.tan: torch.tan,
    np.tanh: torch.tanh,

    # Linear algebra
    np.matmul: torch.matmul,
    np.dot: torch.matmul,  # Note: torch.dot is only for 1D vectors
    np.tensordot: torch.tensordot,
    np.einsum: torch.einsum,
    np.trace: torch.trace,
    np.diagonal: torch.diagonal,
    np.tril: torch.tril,
    np.triu: torch.triu,

    # Creation functions (when operating on TracedTensor)
    np.zeros_like: torch.zeros_like,
    np.ones_like: torch.ones_like,
    np.full_like: torch.full_like,
    np.empty_like: torch.empty_like,

    # Standalone creation (less commonly needed with TracedTensor)
    np.eye: torch.eye,
    np.zeros: torch.zeros,
    np.ones: torch.ones,
    np.full: torch.full,
    np.arange: torch.arange,
    np.linspace: torch.linspace,
}

# Functions that need axis -> dim conversion
AXIS_TO_DIM_FUNCTIONS = {
    torch.sum,
    torch.mean,
    torch.std,
    torch.var,
    torch.min,
    torch.max,
    torch.amin,
    torch.amax,
    torch.argmin,
    torch.argmax,
    torch.cumsum,
    torch.cumprod,
    torch.all,
    torch.any,
    torch.cat,
    torch.squeeze,
    torch.unsqueeze,
    torch.flip,
    torch.roll,
    torch.sort,
    torch.argsort,
}


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
        if isinstance(obj, TracedData):
            return obj.data
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
        
        # np.var/std default to ddof=0 (population), torch defaults to correction=1 (sample)
        if torch_func in (torch.var, torch.std):
            if 'ddof' in proxy_kwargs:
                proxy_kwargs['correction'] = proxy_kwargs.pop('ddof')
            elif 'correction' not in proxy_kwargs:
                proxy_kwargs['correction'] = 0

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
        torch_func = NUMPY_UFUNC_TO_TORCH.get(ufunc)
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

    def _patch_sort(self, torch_func, traced_array, proxy_args, args, kwargs):
        """Patch sort/argsort to use torch.topk for ONNX compatibility.

        torch.sort/argsort export to ONNX TopK with a dynamically computed K
        (via Shape+Gather) that ONNX Runtime's shape inference cannot verify.
        Using torch.topk with a concrete int k bakes K as a constant initializer
        with the correct shape [1], which ONNX Runtime accepts.

        Returns the proxy output if patched, or None if not applicable.
        """
        if torch_func not in (torch.sort, torch.argsort):
            return None

        input_array = args[0]
        dim = kwargs.get('axis', -1)
        if dim is None:
            dim = -1
        k = input_array.shape[dim]

        topk_kwargs = {'dim': dim, 'largest': False, 'sorted': True}
        proxy_out = traced_array._context.tracer.create_proxy(
            "call_function", torch.topk, (proxy_args[0], k), topk_kwargs
        )
        # topk returns (values, indices): index 0 for sort, 1 for argsort
        item_index = 0 if torch_func == torch.sort else 1
        return traced_array._context.tracer.create_proxy(
            "call_function", operator.getitem, (proxy_out, item_index), {}
        )

    def __array_function__(self, func, types, args, kwargs):
        """Intercept numpy array functions.

        This is called when numpy functions (np.sum, np.concatenate, etc.)
        are applied to this TracedNpArray. We execute the numpy operation
        and record the torch equivalent in the graph.
        """
        # Get the torch equivalent
        torch_func = NUMPY_FUNC_TO_TORCH.get(func)
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

        # Apply sort/argsort patch if applicable, otherwise record normally
        proxy_out = self._patch_sort(torch_func, traced_array, proxy_args, args, kwargs)
        if proxy_out is None:
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
        elif isinstance(result_array, np.generic):
            return traced_array._new(np.asarray(result_array), proxy_out)
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
        """Indexed assignment using functional operations for graph compatibility.
        
        Uses the shared _create_setitem_proxy helper from TracedData to convert
        __setitem__ to torch.index_put for FX/TorchScript/ONNX compatibility.
        """
        # Unwrap value if it's a TracedNpArray
        unwrapped_value = TracedNpArray.unwrap_traced_array(value)
        
        # Perform the actual assignment on the underlying array
        self.view(np.ndarray)[key] = unwrapped_value

        # Skip tracing if context is not tracing
        if not self.validate_status():
            return

        # Extract proxy from value if it's a TracedNpArray, or convert to tensor for graph
        if isinstance(value, TracedNpArray):
            value_proxy = value.proxy
        elif isinstance(value, np.ndarray):
            # Convert numpy array to torch tensor in graph
            value_proxy = self._context.tracer.create_proxy(
                "call_function", torch.as_tensor, (value.tolist(),), {}
            )
        else:
            # Scalar or list - will be handled by _create_setitem_proxy
            value_proxy = value

        # Use shared helper to create the proxy
        proxy_out = self._create_setitem_proxy(key, value_proxy)
        
        if proxy_out is not None:
            self._proxy = proxy_out
            return
        
        # Handle unsupported cases with warnings
        if isinstance(key, tuple):
            if all(isinstance(k, int) for k in key):
                _get_logger().warning(
                    "Multi-dimensional integer indexing in setitem may not export correctly. "
                    "Consider using functional operations instead."
                )
            else:
                _get_logger().warning(
                    "Complex multi-dimensional indexing in setitem may not export correctly. "
                    "Consider restructuring to use simple slices."
                )
        else:
            _get_logger().warning(
                f"Indexing with {type(key).__name__} in setitem may not export correctly. "
                "Consider using functional operations instead."
            )
        
        # Fallback: record __setitem__ directly (may not export)
        self._proxy = self._context.tracer.create_proxy(
            "call_method", "__setitem__", (self._proxy, key, value_proxy), {}
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

    def __array__(self, dtype=None, copy=None):
        """Convert to numpy array (NumPy array protocol).
        
        When tracing is active, returns self (or a dtype-converted TracedNpArray)
        to preserve tracing. When not tracing, returns a plain numpy array.
        
        Args:
            dtype: Optional dtype for the resulting array
            copy: Optional copy flag (NumPy 2.0+)
                - None: Copy only if necessary (default)
                - True: Always copy
                - False: Never copy; raise ValueError if copy required
        """
        # Check if dtype change requires a copy
        needs_dtype_copy = dtype is not None and dtype != self.dtype
        
        # copy=False but copy is required → raise error (NumPy 2.0 semantics)
        if copy is False and needs_dtype_copy:
            raise ValueError(
                f"Unable to avoid copy while creating an array with dtype {dtype} "
                f"from array with dtype {self.dtype}."
            )
        
        # If not tracing, return plain numpy array
        if not self.validate_status():
            arr = self.view(np.ndarray)
            if needs_dtype_copy:
                arr = arr.astype(dtype)
            if copy is True:
                arr = arr.copy()
            return arr
        
        # Tracing - preserve TracedNpArray
        result = self
        if needs_dtype_copy:
            result = result.astype(dtype)
        if copy is True:
            result = result.copy()
        return result

    def tolist(self):
        """Convert to Python list.
        
        Warning: During tracing, this extracts concrete values and breaks the trace.
        If used in a conditional, the graph will only capture one branch.
        """
        if self.is_tracing:
            _get_logger().warning(
                f"TracedNpArray '{self._name}'.tolist() called during tracing. "
                f"This extracts concrete Python values, breaking the trace chain. "
                f"If used in a conditional (if/while), the graph will only capture one branch."
            )
        return self.view(np.ndarray).tolist()

    def item(self):
        """Return scalar value.
        
        Warning: During tracing, this extracts a concrete scalar and breaks the trace.
        If used in a conditional, the graph will only capture one branch.
        """
        if self.is_tracing:
            _get_logger().warning(
                f"TracedNpArray '{self._name}'.item() called during tracing. "
                f"This extracts a concrete Python scalar, breaking the trace chain. "
                f"If used in a conditional (if/while), the graph will only capture one branch."
            )
        return self.view(np.ndarray).item()

    # =========================================================================
    # Representation
    # =========================================================================

    def __repr__(self):
        return f"TracedNpArray(name={self._name}, shape={self.shape}, dtype={self.dtype})"

    def __str__(self):
        return f"TracedNpArray({self.view(np.ndarray)})"
