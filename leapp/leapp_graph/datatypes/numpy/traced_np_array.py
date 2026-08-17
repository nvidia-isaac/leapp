#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from leapp.utils.logging import _get_logger
from leapp.utils.dtype import DtypeCodec, dtype_to_name, register_dtype_codec
from ..proxy_view import bind_new_view
from ..traced_data import TracedData


# numpy dtype object -> common name string. Lives with the numpy node library
# so the backend's dtype knowledge is unified with its implementation; the
# registry lets leapp core resolve dtypes without importing numpy directly.
_NUMPY_DTYPE_TO_NAME = dict(
    (dtype, name)
    for scalar, name in {
        np.float64: "float64",
        np.float32: "float32",
        np.float16: "float16",
        np.int16: "int16",
        np.int32: "int32",
        np.int64: "int64",
        np.uint8: "uint8",
        np.int8: "int8",
        np.bool_: "bool",
    }.items()
    for dtype in (scalar, np.dtype(scalar))
)

register_dtype_codec(DtypeCodec(
    backend="numpy",
    matches=lambda v: isinstance(v, np.ndarray),
    value_dtype=lambda v: v.dtype,
    dtype_to_name=_NUMPY_DTYPE_TO_NAME,
))


def _torch_dtype_for(dtype):
    """Map a numpy dtype to its torch counterpart, or None if unsupported."""
    try:
        return getattr(torch, dtype_to_name(np.dtype(dtype)))
    except (ValueError, TypeError):
        return None


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

    # ``np.copy`` is the allocating form that reaches ``__array_function__``.
    # ``.copy()`` calls ``preserve_port`` on the method itself; ``np.asanyarray``
    # returns this carrier without dispatch.
    _EQUIVALENT_COPY_NAMES = frozenset({"copy"})
    _NATIVE_TYPE = np.ndarray

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
        bind_new_view(obj, name, context, proxy)
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
        # Copy tracing state from the source array. ``getattr`` covers a plain
        # ndarray source and a source still mid-construction.
        bind_new_view(
            self,
            getattr(obj, '_name', 'derived'),
            getattr(obj, '_context', None),
            getattr(obj, 'proxy', None),
        )

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
        intermediate_name = self._name_from_proxy(proxy)
        return TracedNpArray(array, intermediate_name, self._context, proxy)

    @staticmethod
    def find_traced_array(obj):
        """Find the first TracedNpArray in a supported nested structure.

        NumPy dispatch must anchor on a NumPy carrier specifically, so this
        cannot use the backend-agnostic ``find_traced_data``.
        """
        found = None

        def visit(item):
            nonlocal found
            if found is None and isinstance(item, TracedNpArray):
                found = item
            return item

        TracedData._map_structure(obj, visit)
        return found

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
        """Recursively extract proxies and convert constant NumPy arrays."""
        def convert(item):
            if isinstance(item, TracedData):
                return item.proxy
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item.copy())
            return item

        return TracedData._map_structure(obj, convert)

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
        """Execute NumPy ufuncs and record them only while the source is active."""
        traced_array = TracedNpArray.find_traced_array([inputs, kwargs])
        unwrapped_inputs = tuple(
            TracedData.unwrap_traced_data(inp) for inp in inputs
        )
        unwrapped_kwargs = {
            key: TracedData.unwrap_traced_data(value)
            for key, value in kwargs.items()
        }

        if method != "__call__":
            # Reduction/accumulation protocols are not export mappings yet, but
            # they must remain usable after the producing node has finished.
            if traced_array is None or traced_array.validate_status(inputs, kwargs):
                return NotImplemented
            return getattr(ufunc, method)(*unwrapped_inputs, **unwrapped_kwargs)

        result_array = ufunc(*unwrapped_inputs, **unwrapped_kwargs)
        if traced_array is None:
            return result_array

        # An inactive traced array is a finished boundary value. Derived results
        # are new data that this node never published, so they stay native.
        if not traced_array.validate_status(inputs, kwargs):
            return result_array

        torch_func = NUMPY_UFUNC_TO_TORCH.get(ufunc)
        if torch_func is None:
            _get_logger().warning(
                f"No torch equivalent for numpy ufunc {ufunc.__name__}. "
                f"Operation will not be traced."
            )
            return result_array

        proxy_inputs = tuple(self._extract_proxy(inp) for inp in inputs)
        proxy_kwargs = self._convert_numpy_kwargs_to_torch(
            {key: self._extract_proxy(value) for key, value in kwargs.items()},
            torch_func,
            original_kwargs=kwargs,
            args=inputs,
        )
        proxy_out = traced_array._context.tracer.create_proxy(
            "call_function", torch_func, proxy_inputs, proxy_kwargs
        )

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
        """Execute NumPy functions and record them only while the source is active."""
        traced_array = TracedNpArray.find_traced_array([args, kwargs])
        unwrapped_args = tuple(
            TracedData.unwrap_traced_data(arg) for arg in args
        )
        unwrapped_kwargs = {
            key: TracedData.unwrap_traced_data(value)
            for key, value in kwargs.items()
        }
        result_array = func(*unwrapped_args, **unwrapped_kwargs)

        if traced_array is None:
            return result_array

        # Inactive carriers use full NumPy behavior, including functions that
        # have no Torch export mapping.
        if not traced_array.validate_status(args, kwargs):
            if self._is_equivalent_copy(func, traced_array, result_array, args):
                return traced_array.preserve_port(result_array)
            return result_array

        torch_func = NUMPY_FUNC_TO_TORCH.get(func)
        if torch_func is None:
            _get_logger().warning(
                f"No torch equivalent for numpy function {func.__name__}. "
                f"Operation will not be traced."
            )
            return result_array

        proxy_args = tuple(self._extract_proxy(arg) for arg in args)
        proxy_kwargs = self._convert_numpy_kwargs_to_torch(
            {key: self._extract_proxy(value) for key, value in kwargs.items()},
            torch_func,
            original_kwargs=kwargs,
            args=args,
        )

        proxy_out = self._patch_sort(torch_func, traced_array, proxy_args, args, kwargs)
        if proxy_out is None:
            proxy_out = traced_array._context.tracer.create_proxy(
                "call_function", torch_func, proxy_args, proxy_kwargs
            )

        if isinstance(result_array, (tuple, list)):
            result = []
            for index, array in enumerate(result_array):
                if isinstance(array, np.ndarray):
                    item_proxy = traced_array._context.tracer.create_proxy(
                        "call_function", operator.getitem, (proxy_out, index), {}
                    )
                    result.append(traced_array._new(array, item_proxy))
                else:
                    result.append(array)
            return type(result_array)(result)
        if isinstance(result_array, np.ndarray):
            return traced_array._new(result_array, proxy_out)
        if isinstance(result_array, np.generic):
            return traced_array._new(np.asarray(result_array), proxy_out)
        return result_array

    # =========================================================================
    # Arithmetic Operators
    # =========================================================================

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        # Keep commutative reverse ops array-first for exporter compatibility.
        return np.add(self, other)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        # Keep commutative reverse ops array-first for exporter compatibility.
        return np.multiply(self, other)

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
            self._proxy_view.proxy = result.proxy
            return self
        return result

    def __isub__(self, other):
        result = np.subtract(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy_view.proxy = result.proxy
            return self
        return result

    def __imul__(self, other):
        result = np.multiply(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy_view.proxy = result.proxy
            return self
        return result

    def __itruediv__(self, other):
        result = np.divide(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy_view.proxy = result.proxy
            return self
        return result

    def __ifloordiv__(self, other):
        result = np.floor_divide(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy_view.proxy = result.proxy
            return self
        return result

    def __ipow__(self, other):
        result = np.power(self, other)
        if isinstance(result, TracedNpArray):
            np.copyto(self.view(np.ndarray), result.view(np.ndarray))
            self._proxy_view.proxy = result.proxy
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
        real_key = TracedData.unwrap_traced_data(key)
        result = self.view(np.ndarray)[real_key]

        if not self.validate_status(args=(key,)):
            return result

        proxy_out = self._create_getitem_proxy(key)
        if proxy_out is None:
            proxy_key = self._extract_proxy(key)
            proxy_out = self._context.tracer.create_proxy(
                "call_function", operator.getitem,
                (self.proxy, proxy_key), {}
            )

        if isinstance(result, np.ndarray):
            return self._new(result, proxy_out)
        return result

    def _record_assignment(self, key, value, real_value):
        """Record one functional assignment and update this object's proxy."""
        value_proxy = value.proxy if isinstance(value, TracedData) else value
        if self._update_setitem_proxy(
            key, value_proxy, real_value=real_value
        ):
            return True

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
        return False

    def __setitem__(self, key, value):
        """Indexed assignment with functional ``index_put`` lowering.

        Plain ``np.ndarray`` destinations cannot be promoted into
        ``TracedNpArray``; the destination must already be traced.
        """
        real_key = TracedData.unwrap_traced_data(key)
        real_value = TracedData.unwrap_traced_data(value)
        self.view(np.ndarray)[real_key] = real_value

        if not self.validate_status(args=(key, value)):
            self.overwrite_port(key, value)
            return

        if not self._record_assignment(key, value, real_value):
            value_proxy = value.proxy if isinstance(value, TracedData) else value
            self._proxy_view.proxy = self._context.tracer.create_proxy(
                "call_method", "__setitem__", (self.proxy, key, value_proxy), {}
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
        """Type conversion, recorded as a dtype cast in the graph."""
        result = self.view(np.ndarray).astype(dtype)
        if not self.validate_status():
            return result
        torch_dtype = _torch_dtype_for(dtype)
        if torch_dtype is None:
            _get_logger().warning(
                f"astype({dtype}) has no torch equivalent and cannot be recorded; "
                f"the exported graph will keep {self.dtype}"
            )
            return self._new(result, self.proxy)
        proxy_out = self._context.tracer.create_proxy(
            "call_method", "to", (self.proxy, torch_dtype), {}
        )
        return self._new(result, proxy_out)

    def copy(self):
        """Return a copy of the array."""
        result = self.view(np.ndarray).copy()
        if not self.validate_status():
            return self.preserve_port(result)
        proxy_out = self._context.tracer.create_proxy(
            "call_function", torch.clone, (self.proxy,), {}
        )
        return self._new(result, proxy_out)

    # =========================================================================
    # Conversion Methods
    # =========================================================================

    def __array__(self, dtype=None, copy=None):
        """Convert to numpy array (NumPy array protocol).
        
        Returns self (or a converted TracedNpArray) for both active and inactive
        traced values so representation conversion cannot silently demote it.
        
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
            _get_logger().fatal(
                f"Unable to avoid copy while creating an array with dtype {dtype} "
                f"from array with dtype {self.dtype}.",
                error_type=ValueError,
            )
        
        # Preserve the carrier even after its source context stops tracing.
        # astype() and copy() already retain the complete tracing state.

        result = self
        if needs_dtype_copy:
            result = result.astype(dtype)
        # astype() already allocated, so an extra copy would only add a
        # redundant clone to the graph.
        if copy is True and not needs_dtype_copy:
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
