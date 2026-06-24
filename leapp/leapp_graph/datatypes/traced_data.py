#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Base class for traced data types (tensors, arrays, etc.)."""

from abc import ABC, abstractmethod
from typing import Any, Set, Optional

from torch.fx.proxy import Proxy
from leapp.utils.logging import _get_logger
from leapp.utils.dtype import dtype_to_name

import torch


class TracedData(ABC):
    """Abstract base class for traced data types.
    
    This class provides the common infrastructure for tracing operations
    on different data types (torch.Tensor, numpy.ndarray, etc.) using torch.fx.
    
    Child classes implement type-specific operation interception:
    - TracedTensor: Uses __torch_function__ for torch operations
    - TracedArray (future): Uses __array_ufunc__/__array_function__ for numpy
    
    The FX graph always records torch operations, so child classes must
    map their native operations to torch equivalents for proxy recording.
    """
    
    def __init__(self, value: Any, name: str, context, proxy: Proxy):
        """Initialize a TracedData instance.
        
        Args:
            value: The underlying data (torch.Tensor, numpy.ndarray, etc.)
            name: Name for the data (used in ONNX export and graph)
            context: The TraceContext that owns this data
            proxy: The fx.Proxy for graph recording
        """
        self._value = value
        self._init_tracing_state(name, context, proxy)
    
    # =========================================================================
    # Common Properties
    # =========================================================================

    def _init_tracing_state(self, name: str, context, proxy: Proxy) -> None:
        """Initialize common tracing metadata for TracedData subclasses."""
        self._name = name
        self._context = context
        self._proxy = proxy

    @staticmethod
    def _name_from_proxy(proxy: Proxy) -> str:
        """Return the conventional traced-data name for an operation result."""
        if proxy is not None:
            return str(proxy.node.name)
        return "untraced"
    
    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy
    
    @property
    def name(self) -> str:
        """Get the name of this traced data."""
        return self._name
    
    @property
    def context(self) -> str:
        """Get the name of the context that owns this data."""
        if self._context is None:
            return "untraced"
        return self._context.name
    
    @property
    def context_obj(self):
        """Get the TraceContext that owns this data."""
        return self._context
    
    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the context that owns this data."""
        if self._context is None:
            return False
        return self._context.is_tracing
    
    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Get the underlying data as a torch.Tensor."""
        pass

    @property
    @abstractmethod
    def data(self) -> Any:
        """Get the underlying data."""
        pass

    def get_dtype_name(self) -> str:
        """Common dtype name (e.g. "float32") of the underlying value.

        Delegates to the backend dtype-codec registry, so each backend's
        mapping lives with that backend rather than in leapp core.
        """
        return dtype_to_name(self.data.dtype)
    # =========================================================================
    # Abstract Methods - Must be implemented by child classes
    # =========================================================================
    
    @abstractmethod
    def _new(self, value: Any, proxy: Proxy = None) -> "TracedData":
        """Create a new TracedData of the same type in the same context.
        
        Args:
            value: The new underlying value
            proxy: The proxy for the new data (can be None if not tracing)
            
        Returns:
            A new TracedData instance of the same type
        """
        pass
    
    # =========================================================================
    # Common Static Methods
    # =========================================================================
    
    @staticmethod
    def find_traced_data(obj):
        """Find the first TracedData in obj (including nested in lists/tuples).
        
        Args:
            obj: Object to search (can be TracedData, list, tuple, or other)
            
        Returns:
            The first TracedData found, or None if not found
        """
        if isinstance(obj, TracedData):
            return obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                result = TracedData.find_traced_data(item)
                if result is not None:
                    return result
        return None
    
    @staticmethod
    def unwrap_traced_data(obj):
        """Recursively unwrap TracedData to get raw values.
        
        Args:
            obj: Object to unwrap (can be TracedData, list, tuple, dict, or other)
            
        Returns:
            The unwrapped value with all TracedData replaced by their raw values
        """
        if isinstance(obj, TracedData):
            return obj.data
        elif isinstance(obj, (list, tuple)):
            return type(obj)(TracedData.unwrap_traced_data(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: TracedData.unwrap_traced_data(v) for k, v in obj.items()}
        return obj
    
    @staticmethod
    def find_all_contexts(obj, contexts: Optional[Set[str]] = None) -> Set[str]:
        """Recursively find all unique context names.
        
        Args:
            obj: Object to search
            contexts: Set to accumulate context names (created if None)
            
        Returns:
            Set of context names found
        """
        if contexts is None:
            contexts = set()
        if isinstance(obj, TracedData) and obj.is_tracing:
            contexts.add(obj.context)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                TracedData.find_all_contexts(item, contexts)
        elif isinstance(obj, dict):
            for v in obj.values():
                TracedData.find_all_contexts(v, contexts)
        return contexts
    
    @staticmethod
    def extract_proxy(obj):
        """Recursively extract proxies from TracedData objects.
        
        Args:
            obj: Object to extract proxies from
            
        Returns:
            Object with TracedData replaced by their proxies
        """
        if isinstance(obj, TracedData):
            return obj.proxy
        elif isinstance(obj, (list, tuple)):
            return type(obj)(TracedData.extract_proxy(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: TracedData.extract_proxy(v) for k, v in obj.items()}
        return obj
    
    # =========================================================================
    # Common Validation
    # =========================================================================
    
    def validate_status(self, args=None, kwargs=None) -> bool:
        """Validate that this TracedData can be used in the current context.
        
        Checks:
        1. If not tracing, returns False (operation should not be recorded)
        2. If tracing inside a traced function, raises an error
        3. If mixing TracedData from different contexts, raises an error
        
        Args:
            args: Positional arguments to check for other TracedData
            kwargs: Keyword arguments to check for other TracedData
            
        Returns:
            True if tracing should proceed, False if not tracing
            
        Raises:
            Exception: If TracedData is used inside a traced function
            Exception: If TracedData from multiple contexts are mixed
        """
        if not self.is_tracing:
            return False

        contexts = set()
        if args is not None:
            for arg in args:
                contexts = TracedData.find_all_contexts(arg, contexts)
        if kwargs is not None:
            for kwarg in kwargs.values():
                contexts = TracedData.find_all_contexts(kwarg, contexts)
        
        if len(contexts) > 1:
            cls_name = self.__class__.__name__
            _get_logger().fatal(
                f"Error: detected multiple {cls_name} contexts: {contexts} inside of a traced function.\n"
                "\n"
                f"This happens when you mix multiple active {cls_name}s from different contexts "
                "inside of a traced function/block. solutions:\n"
                "\n"
                "1. call output_tensors() to finalize one of the nodes first\n"
                "2. combine both nodes into a single node by calling input_tensors() with the same node name",
                error_type=Exception)
        return True
    
    # =========================================================================
    # Common Magic Methods
    # =========================================================================
    
    def __len__(self) -> int:
        """Length operator."""
        return len(self._value)
    
    def __bool__(self) -> bool:
        """Boolean conversion for TracedNpArray.
        
        During tracing, this indicates tensor-dependent control flow which
        produces a silently incorrect graph. Logs an error to alert the user.
        """
        if self.is_tracing:
            _get_logger().error(
                f"Attempted to use {self.__class__.__name__} '{self._name}' from node '{self.context}' "
                f"in a boolean context (if/while/and/or/not) during tracing.\n"
                f"\n"
                f"This typically happens with array-value-dependent control flow:\n"
                f"  - if array.mean() > 0.9: return early   (Early Exit)\n"
                f"  - while error.sum() > eps: refine()      (Iterative Refinement)\n"
                f"  - if array.any(): ...                     (Conditional on values)\n"
                f"\n"
                f"The traced graph is a static DAG and cannot represent dynamic branches.\n"
                f"Only the branch taken during tracing would be recorded, producing a\n"
                f"silently incorrect graph for other inputs.\n"
                f"\n"
                f"Alternatives:\n"
                f"  1. Use torch.where(condition, true_val, false_val) for simple if/else\n"
                f"  2. Use fixed iteration counts instead of dynamic while loops\n"
                f"  3. Break the trace: call output_tensors() before the conditional,\n"
                f"     then start a new input_tensors() after it"
            )
        return bool(self.data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __format__(self, format_spec: str) -> str:
        """Format the TracedData by delegating to the underlying value."""
        return self._value.__format__(format_spec)

    # =========================================================================
    # Setitem Proxy Generation
    # =========================================================================
    
    def _create_setitem_proxy(self, key, value_proxy):
        """Create proxy for __setitem__ using functional torch operations.
        
        Converts indexed assignment to torch.index_put for graph compatibility.
        This is shared logic that both TracedTensor and TracedNpArray can use.
        
        Args:
            key: The indexing key (int, slice, tuple)
            value_proxy: Proxy for the value, or a constant value
            
        Returns:
            New proxy representing the modified tensor, or None if unsupported
        """
        # Convert full slice [:] to slice(0, None, 1) for uniform handling
        # We always use index_put to maintain connection to self._proxy in the graph
        if key == slice(None) or (isinstance(key, tuple) and all(k == slice(None) for k in key)):
            key = slice(0, None, 1)  # Convert to explicit slice for index_put handling
        
        # Single index: x[i] = v → index_put(x, (tensor([i]),), v)
        if isinstance(key, int):
            indices_proxy = self._context.tracer.create_proxy(
                "call_function", torch.tensor, ([key],), {"dtype": torch.long}
            )
            if not isinstance(value_proxy, Proxy):
                value_tensor_proxy = self._context.tracer.create_proxy(
                    "call_function", torch.tensor, ([value_proxy],), {}
                )
            else:
                # Reshape to ensure 1D for index_put
                value_tensor_proxy = self._context.tracer.create_proxy(
                    "call_method", "reshape", (value_proxy, (-1,)), {}
                )
            return self._context.tracer.create_proxy(
                "call_function", torch.index_put,
                (self._proxy, (indices_proxy,), value_tensor_proxy), {}
            )
        
        # Slice: x[start:end:step] = v → index_put with arange indices
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            end = key.stop
            step = key.step if key.step is not None else 1
            
            if end is None:
                # Use tensor size to determine end
                size_proxy = self._context.tracer.create_proxy(
                    "call_method", "size", (self._proxy, 0), {}
                )
                indices_proxy = self._context.tracer.create_proxy(
                    "call_function", torch.arange,
                    (start, size_proxy, step), {"dtype": torch.long}
                )
            else:
                indices_proxy = self._context.tracer.create_proxy(
                    "call_function", torch.arange,
                    (start, end, step), {"dtype": torch.long}
                )
            
            # Handle constant values - register as buffer for TorchScript compatibility
            if not isinstance(value_proxy, Proxy):
                value_tensor = torch.tensor(value_proxy) if not isinstance(value_proxy, torch.Tensor) else value_proxy
                attr_name = f"_setitem_value_{id(value_tensor)}"
                self._context.tracer.root.register_buffer(attr_name, value_tensor)
                value_proxy = self._context.tracer.create_proxy(
                    "get_attr", attr_name, (), {}
                )
            
            return self._context.tracer.create_proxy(
                "call_function", torch.index_put,
                (self._proxy, (indices_proxy,), value_proxy), {}
            )
        
        # Unsupported key type - return None to signal caller should handle
        return None
