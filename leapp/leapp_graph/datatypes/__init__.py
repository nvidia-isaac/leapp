#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Data type compatibility utilities for LEAPP tracing.

This module provides:
- TracedData base class and its implementations (TracedTensor, TracedNpArray)
- Type registry for mapping base types to traced types
- NumPy-to-PyTorch compatibility utilities
"""

from typing import Type, Union, Optional

import numpy as np
import torch as _torch

from .traced_data import TracedData
from .torch.traced_tensor import TracedTensor
from .numpy.traced_np_array import TracedNpArray
from .warp import TracedWpArray, WarpLeappCallDetector, wp

from .patching import (
    apply_traced_data_patches,
    remove_traced_data_patches,
    is_numpy_patching_enabled,
)


# =============================================================================
# Type Registry
# =============================================================================

# Mapping from base tensor types to their traced counterparts
# Order matters: more specific types should come first
TRACED_TYPE_REGISTRY: dict[type, Type[TracedData]] = {
    _torch.Tensor: TracedTensor,
    np.ndarray: TracedNpArray,
}
if wp is not None and TracedWpArray is not None:
    TRACED_TYPE_REGISTRY[wp.array] = TracedWpArray

# Tuple of all base types that can be traced (for isinstance checks)
TRACABLE_BASE_TYPES: tuple[type, ...] = tuple(TRACED_TYPE_REGISTRY.keys())

# Tuple of all traced types (for isinstance checks)
TRACED_TYPES: tuple[Type[TracedData], ...] = tuple(TRACED_TYPE_REGISTRY.values())


# =============================================================================
# Type Checking Functions
# =============================================================================

def is_tracable_tensor_type(obj) -> bool:
    """Check if an object is a type that can be traced.
    
    This includes both base types (torch.Tensor, np.ndarray) and
    their traced counterparts (TracedTensor, TracedNpArray).
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is a traceable tensor type
    """
    return isinstance(obj, TRACABLE_BASE_TYPES)


def is_traced_type(obj) -> bool:
    """Check if an object is a TracedData subclass instance.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is a TracedData instance
    """
    return isinstance(obj, TracedData)


def get_traced_class_for(obj_or_type: Union[type, object]) -> Optional[Type[TracedData]]:
    """Get the appropriate TracedData subclass for a given type or object.
    
    Args:
        obj_or_type: Either a type (e.g., torch.Tensor) or an instance
        
    Returns:
        The corresponding TracedData subclass, or None if not traceable
        
    Examples:
        >>> get_traced_class_for(torch.Tensor)
        <class 'TracedTensor'>
        >>> get_traced_class_for(np.array([1, 2, 3]))
        <class 'TracedNpArray'>
        >>> get_traced_class_for("string")
        None
    """
    # If already a traced type, return its class
    if isinstance(obj_or_type, TracedData):
        return type(obj_or_type)
    
    # If it's a type, look it up directly
    if isinstance(obj_or_type, type):
        # Check for exact match first
        if obj_or_type in TRACED_TYPE_REGISTRY:
            return TRACED_TYPE_REGISTRY[obj_or_type]
        # Check for subclass match
        for base_type, traced_type in TRACED_TYPE_REGISTRY.items():
            if issubclass(obj_or_type, base_type):
                return traced_type
        return None
    
    # It's an instance, find matching type
    for base_type, traced_type in TRACED_TYPE_REGISTRY.items():
        if isinstance(obj_or_type, base_type):
            return traced_type
    
    return None


def as_traced(data, name: str, context, proxy) -> TracedData:
    """Create a TracedData instance from a tensor or array.
    
    This is a factory function that automatically selects the appropriate
    TracedData subclass based on the input data type.
    
    Args:
        data: The tensor/array to wrap (torch.Tensor or np.ndarray)
        name: Name for the traced data (used in export and graph)
        context: The TraceContext that owns this data
        proxy: The fx.Proxy for graph recording
        
    Returns:
        A TracedData instance (TracedTensor or TracedNpArray)
        
    Raises:
        TypeError: If data is not a supported type
        
    Examples:
        >>> tensor = torch.randn(3)
        >>> traced = as_traced(tensor, "input", context, proxy)
        >>> type(traced)
        <class 'TracedTensor'>
        
        >>> array = np.array([1, 2, 3])
        >>> traced = as_traced(array, "input", context, proxy)
        >>> type(traced)
        <class 'TracedNpArray'>
    """
    # If already traced, unwrap to get the underlying tensor/array
    # We need to create a NEW traced instance with the new context
    # This erases any knowledge of previous operations operated on the data
    if isinstance(data, TracedData):
        data = data.data
    
    # Find the appropriate traced class
    traced_class = get_traced_class_for(data)
    
    if traced_class is None:
        raise TypeError(
            f"Cannot create traced data from type {type(data).__name__}. "
            f"Supported types: {', '.join(t.__name__ for t in TRACABLE_BASE_TYPES)}"
        )
    
    return traced_class(data, name, context, proxy)

def to_export_torch_tensor(data) -> _torch.Tensor:
    """Convert a traceable LEAPP value to ``torch.Tensor`` for export metadata."""
    if isinstance(data, TracedData):
        return data.tensor
    if isinstance(data, _torch.Tensor):
        return data
    if np is not None and isinstance(data, np.ndarray):
        return _torch.from_numpy(data)
    if wp is not None and isinstance(data, wp.array):
        return wp.to_torch(data)
    raise TypeError(
        f"Cannot convert type {type(data).__name__} to torch.Tensor"
    )

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core data types
    "TracedData",
    "TracedTensor",
    "TracedNpArray",
    "TracedWpArray",
    "WarpLeappCallDetector",
    # Type registry
    "TRACED_TYPE_REGISTRY",
    "TRACABLE_BASE_TYPES",
    "TRACED_TYPES",
    # Factory and type checking functions
    "as_traced",
    "is_tracable_tensor_type",
    "is_traced_type",
    "get_traced_class_for",
    "to_export_torch_tensor",
    # Patch management
    "apply_traced_data_patches",
    "remove_traced_data_patches",
    "is_numpy_patching_enabled",
]
