#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patching for torch and numpy functions.

Several torch functions do early type checks in C++ before __torch_function__
can intercept them. Similarly, numpy's np.array() and np.asarray() strip
subclasses by default.

We use deferred patching to:
1. Enable TracedTensor/TracedNpArray compatibility
2. Avoid conflicts with TorchScript modules compiled at import time

Patches are applied when tracing starts and removed when tracing stops.
"""

import functools
import numpy as np
import torch


# =============================================================================
# Patch Factories
# =============================================================================

def _create_torch_patch(original_func):
    """Create a patched version of a torch function that preserves tracing.
    
    Flow: unwrap TracedData → call original → rewrap result with as_traced()
    
    Used for: torch.from_numpy, torch.as_tensor, torch.tensor
    """
    @functools.wraps(original_func)
    def patched(data, *args, **kwargs):
        from . import as_traced, is_traced_type
        
        is_traced_and_tracing = is_traced_type(data) and data.is_tracing
        
        if is_traced_and_tracing:
            name, context, proxy = data.name, data.context_obj, data.proxy
            raw_data = data.data
        else:
            raw_data = data
        
        result = original_func(raw_data, *args, **kwargs)
        
        if is_traced_and_tracing:
            # If same object returned, preserve original TracedData
            if result is raw_data:
                return data
            result = as_traced(result, name, context, proxy)
        
        return result
    
    return patched


def _create_numpy_patch(original_func):
    """Create a patched version of a numpy array function that preserves tracing.
    
    Flow: call TracedData.__array__() directly (bypasses PyTorch's __array__)
    
    Used for: np.array, np.asarray
    
    Why separate? PyTorch's torch.Tensor.__array__ doesn't handle NumPy 2.0's
    copy= parameter, causing deprecation warnings. Our __array__ handles it.
    """
    @functools.wraps(original_func)
    def patched(data, *args, **kwargs):
        from . import is_traced_type
        
        is_traced_and_tracing = is_traced_type(data) and data.is_tracing
        
        if is_traced_and_tracing:
            # Use our __array__ which handles dtype/copy correctly
            dtype = kwargs.get('dtype')
            copy = kwargs.get('copy')
            return data.__array__(dtype=dtype, copy=copy)
        
        # Not traced - use original function
        return original_func(data, *args, **kwargs)
    
    return patched


# =============================================================================
# Patch Registry
# =============================================================================

# Functions to patch: (module, function_name, factory)
_TORCH_FUNCTIONS = [
    (torch, 'from_numpy'),
    (torch, 'as_tensor'),
    (torch, 'tensor'),
]

_NUMPY_FUNCTIONS = [
    (np, 'array'),
    (np, 'asarray'),
]

# Store originals and patches
_originals = {}  # (module, name) -> original_func
_patches = {}    # (module, name) -> patched_func


def _init_patches():
    """Initialize the patch registry. Called once at module load."""
    # Torch functions use the torch factory
    for module, name in _TORCH_FUNCTIONS:
        original = getattr(module, name)
        key = (module, name)
        _originals[key] = original
        _patches[key] = _create_torch_patch(original)
    
    # NumPy functions use the numpy factory
    for module, name in _NUMPY_FUNCTIONS:
        original = getattr(module, name)
        key = (module, name)
        _originals[key] = original
        _patches[key] = _create_numpy_patch(original)


# Initialize patches at module load
_init_patches()


# =============================================================================
# Patch Application
# =============================================================================

_patches_applied = False


def apply_traced_tensor_patches():
    """Apply patches for torch and numpy functions.
    
    Call this when tracing starts to enable TracedTensor/TracedNpArray
    compatibility with functions like:
    - torch.as_tensor, torch.tensor, torch.from_numpy
    - np.array, np.asarray
    """
    global _patches_applied
    if not _patches_applied:
        for (module, name), patched in _patches.items():
            setattr(module, name, patched)
        _patches_applied = True


def remove_traced_tensor_patches():
    """Remove patches for torch and numpy functions.
    
    Call this when tracing stops to restore original function behavior.
    This prevents conflicts with TorchScript compilation.
    """
    global _patches_applied
    if _patches_applied:
        for (module, name), original in _originals.items():
            setattr(module, name, original)
        _patches_applied = False


def is_patching_enabled():
    """Check if patching is currently enabled.
    
    Returns:
        bool: True if patches are applied, False otherwise.
    """
    return _patches_applied


# Keep old name for backward compatibility
is_numpy_patching_enabled = is_patching_enabled
