#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Base class for traced data types (tensors, arrays, etc.)."""

from abc import ABC, abstractmethod
from typing import Any, Set, Optional

from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.tracing_lock import TracingLock

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
        self._name = name
        self._context = context
        self._proxy = proxy
        self._global_tracing_lock = TracingLock()
    
    # =========================================================================
    # Common Properties
    # =========================================================================
    
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
        return self._context.name
    
    @property
    def context_obj(self):
        """Get the TraceContext that owns this data."""
        return self._context
    
    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the context that owns this data."""
        return self._context.is_tracing
    
    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Get the underlying data as a torch.Tensor."""
        pass
    
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
    
    @abstractmethod
    def _unwrap(self) -> Any:
        """Get the underlying raw value.
        
        Returns:
            The underlying value (torch.Tensor, numpy.ndarray, etc.)
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
            return obj._unwrap()
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
        
        if self.is_tracing and self._global_tracing_lock.is_active:
            _get_logger().error(
                f"Error: detected active TracedData {self._name} from node {self.context} inside of a traced function.\n"
                f"\n"
                f"This happens when you have an active TracedData and it is being used for computation inside of a traced function/block."
                f"\n"
                f"You must call output_tensors() to finalize the TracedData node first"
            )
            raise Exception(
                "Cannot use TracedData inside of a traced function/block. "
                "Call output_tensors() first to finalize the TracedData node"
            )
        
        contexts = set()
        if args is not None:
            for arg in args:
                contexts = TracedData.find_all_contexts(arg, contexts)
        if kwargs is not None:
            for kwarg in kwargs.values():
                contexts = TracedData.find_all_contexts(kwarg, contexts)
        
        if len(contexts) > 1:
            _get_logger().error(
                f"Error: detected multiple TracedData contexts: {contexts} inside of a traced function.\n"
                "\n"
                "This happens when you mix multiple active TracedData from different contexts inside of a traced function/block."
                "\n"
                "You can call output_tensors() to finalize one of the TracedData nodes first "
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
            raise Exception(
                "Cannot mix multiple active TracedData from different contexts inside of a traced function/block. "
                "Call output_tensors() to finalize one of the TracedData nodes first "
                "or combine both nodes into a single node by calling input_tensors() with the same node name"
            )
        return True
    
    # =========================================================================
    # Common Magic Methods
    # =========================================================================
    
    def __len__(self) -> int:
        """Length operator."""
        return len(self._value)
    
    def __bool__(self) -> bool:
        """Boolean conversion."""
        return bool(self._value)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __format__(self, format_spec: str) -> str:
        """Format the TracedData by delegating to the underlying value."""
        return self._value.__format__(format_spec)

