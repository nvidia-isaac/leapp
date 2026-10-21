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
from ..proxy_view import bind_new_view, may_adopt_view, share_view
from ..traced_data import TracedData
from .aten_adapter import capture_numpy_call


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

register_dtype_codec(
    DtypeCodec(
        backend="numpy",
        matches=lambda v: isinstance(v, np.ndarray),
        value_dtype=lambda v: v.dtype,
        dtype_to_name=_NUMPY_DTYPE_TO_NAME,
    )
)


def _torch_dtype_for(dtype):
    """Map a numpy dtype to its torch counterpart, or None if unsupported."""
    try:
        return getattr(torch, dtype_to_name(np.dtype(dtype)))
    except (ValueError, TypeError):
        return None


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
        # A view covering exactly the source's bytes is a second handle on one
        # value, so it shares that cell and a mutation through either is visible
        # through both. A slice covers fewer bytes and a freshly allocated ufunc
        # result covers different bytes, so both fail the comparison and keep a
        # root of their own. The view guard covers a source still
        # mid-construction, which has no cell to share yet.
        if getattr(obj, "_proxy_view", None) is not None and may_adopt_view(obj, self):
            share_view(self, obj)
            return
        # Copy tracing state from the source array. ``getattr`` covers a plain
        # ndarray source and a source still mid-construction.
        bind_new_view(
            self,
            getattr(obj, "_name", "derived"),
            getattr(obj, "_context", None),
            getattr(obj, "proxy", None),
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def tensor(self) -> np.ndarray:
        """Get the underlying numpy array (for TracedData compatibility)."""
        array = np.ascontiguousarray(self.view(np.ndarray))
        # NumPy considers singleton axes contiguous even with negative strides.
        if any(stride < 0 for stride in array.strides):
            array = array.copy()
        return torch.from_numpy(array)

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
            if isinstance(item, np.generic):
                return torch.as_tensor(np.asarray(item))
            return item

        return TracedData._map_structure(obj, convert)

    # =========================================================================
    # NumPy dispatch
    # =========================================================================

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Capture active calls; retain native NumPy behavior in transit."""
        source = self.find_traced_array((inputs, kwargs))
        active = source is not None and source.validate_status(inputs, kwargs)
        if method != "__call__" and active:
            return NotImplemented
        if active:
            return capture_numpy_call(source, ufunc, inputs, kwargs)
        return getattr(ufunc, method)(
            *TracedData.unwrap_traced_data(inputs),
            **TracedData.unwrap_traced_data(kwargs),
        )

    def __array_function__(self, func, types, args, kwargs):
        """Use the compatibility layer plus overrides for active calls."""
        source = self.find_traced_array((args, kwargs))
        if source is not None and source.validate_status(args, kwargs):
            return capture_numpy_call(source, func, args, kwargs)
        result = func(
            *TracedData.unwrap_traced_data(args),
            **TracedData.unwrap_traced_data(kwargs),
        )
        if source is not None and self._is_equivalent_copy(func, source, result, args):
            return source.preserve_port(result)
        return result

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
                "call_function", operator.getitem, (self.proxy, proxy_key), {}
            )

        if isinstance(result, np.ndarray):
            return self._new(result, proxy_out)
        return result

    def _record_assignment(self, key, value, real_value):
        """Record one functional assignment and update this object's proxy."""
        value_proxy = value.proxy if isinstance(value, TracedData) else value
        if self._update_setitem_proxy(key, value_proxy, real_value=real_value):
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
        return (
            f"TracedNpArray(name={self._name}, shape={self.shape}, dtype={self.dtype})"
        )

    def __str__(self):
        return f"TracedNpArray({self.view(np.ndarray)})"
