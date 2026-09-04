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

from leapp.utils.logging import _get_logger

from .traced_data import TracedData
from .torch.traced_tensor import TracedTensor
from .numpy.traced_np_array import TracedNpArray
from .proxy_view import (
    ProxyView,
    bind_new_view,
    bind_shared_view,
    layout_key,
    may_adopt_view,
    share_view,
    update_view_proxy,
)
from .warp import TracedWpArray, WarpPatchBackend, wp


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


def as_traced(
    data,
    name: str,
    context,
    proxy=None,
    *,
    view: Optional[ProxyView] = None,
) -> TracedData:
    """Create a TracedData instance from a tensor or array.

    Pass exactly one of ``proxy`` or ``view``:

    - ``proxy``: bind a **new** ``ProxyView`` around that FX value (private cell).
    - ``view``: attach that **existing** ``ProxyView`` (shared cell / zero-copy alias).

    Args:
        data: The tensor/array to wrap (torch.Tensor or np.ndarray)
        name: Name for the traced data (used in export and graph)
        context: The TraceContext that owns this data
        proxy: FX proxy for a private root (mutually exclusive with ``view``)
        view: Existing view to share (mutually exclusive with ``proxy``)

    Returns:
        A TracedData instance (TracedTensor or TracedNpArray)

    Raises:
        TypeError: If data is not a supported type
    """
    if view is not None and proxy is not None:
        _get_logger().fatal(
            "as_traced accepts proxy= or view=, not both",
            error_type=ValueError,
        )

    # Rewrapping an existing traced value must not rebind the producer object.
    # Consumers receive a fresh traced carrier for their own node context so the
    # original value can still fan out to other nodes.
    was_traced = isinstance(data, TracedData)
    if was_traced:
        data = data.data

    # Exact raw Warp arrays can be promoted in place. Existing TracedWpArrays
    # were unwrapped above and must instead get a fresh non-owning traced alias.
    if wp is not None and TracedWpArray is not None and isinstance(data, wp.array):
        if was_traced:
            return TracedWpArray(data, name, context, proxy, view=view)
        return TracedWpArray.make_traced_in_place(
            data, name, context, proxy, view=view
        )

    # Find the appropriate traced class
    traced_class = get_traced_class_for(data)

    if traced_class is None:
        _get_logger().fatal(
            f"Cannot create traced data from type {type(data).__name__}. "
            f"Supported types: {', '.join(t.__name__ for t in TRACABLE_BASE_TYPES)}",
            error_type=TypeError,
        )

    # Constructors always bind a private root; overwrite with the shared view
    # when the caller asked to alias an existing cell.
    seed_proxy = proxy if view is None else view.proxy
    result = traced_class(data, name, context, seed_proxy)
    if view is not None:
        bind_shared_view(result, name, context, view)
    return result


def promote_in_place(data, name: str, context, proxy) -> TracedData:
    """Bind tracing state onto ``data`` itself rather than a fresh carrier.

    This is the counterpart to :func:`as_traced`: where ``as_traced`` hands a
    consumer its own carrier so a producer can fan out, this rebinds the exact
    object the caller already holds, which is what boundary values written into
    a preallocated buffer need.

    Torch and Warp values are upgraded in place, so callers see the change
    without reassigning. A raw ``np.ndarray`` cannot be class-swapped, so NumPy
    returns a zero-copy view and callers must use the return value.

    A carrier that already holds tracing state keeps its own view and only has
    the proxy inside it replaced. Rebinding the object is the point of this
    function, so discarding its view would orphan every alias of the same buffer
    while leaving the object itself looking correct.
    """
    if isinstance(data, TracedData):
        update_view_proxy(data, name, context, proxy)
        return data

    if wp is not None and TracedWpArray is not None and isinstance(data, wp.array):
        return TracedWpArray.make_traced_in_place(data, name, context, proxy)

    if type(data) is _torch.Tensor:
        return TracedTensor._promote_plain_tensor(data, name, context, proxy)

    return as_traced(data, name, context, proxy)


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
    _get_logger().fatal(
        f"Cannot convert type {type(data).__name__} to torch.Tensor",
        error_type=TypeError,
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
    "WarpPatchBackend",
    "ProxyView",
    # View binding / alias classification
    "bind_new_view",
    "bind_shared_view",
    "update_view_proxy",
    "share_view",
    "may_adopt_view",
    "layout_key",
    # Type registry
    "TRACED_TYPE_REGISTRY",
    "TRACABLE_BASE_TYPES",
    "TRACED_TYPES",
    # Factory and type checking functions
    "as_traced",
    "promote_in_place",
    "is_tracable_tensor_type",
    "is_traced_type",
    "get_traced_class_for",
    "to_export_torch_tensor",
]
