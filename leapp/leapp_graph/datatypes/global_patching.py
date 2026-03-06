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
3. Disable torch.jit.script during tracing so new @torch.jit.script
   decorators return the original Python function (traceable by TracedTensor)

Patches are applied when tracing starts and removed when tracing stops.
"""

import sys
import functools
import numpy as np
import torch

from leapp.utils.logging import _get_logger


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


def apply_traced_data_patches():
    """Apply patches for torch and numpy functions.
    
    Call this when tracing starts to enable TracedTensor/TracedNpArray
    compatibility with functions like:
    - torch.as_tensor, torch.tensor, torch.from_numpy
    - np.array, np.asarray
    
    Also disables torch.jit.script via torch.jit._state so that any new
    @torch.jit.script decorators during tracing return the original Python
    function, allowing TracedTensor.__torch_function__ to trace through them.
    """
    global _patches_applied
    if not _patches_applied:
        for (module, name), patched in _patches.items():
            setattr(module, name, patched)
        #TODO: this is exparamental, if it causes issues, we should remove it
        torch.jit._state.disable()
        _patches_applied = True


def remove_traced_data_patches():
    """Remove patches for torch and numpy functions.
    
    Call this when tracing stops to restore original function behavior.
    This prevents conflicts with TorchScript compilation.
    Also re-enables torch.jit.script for normal JIT usage (e.g. export).
    """
    global _patches_applied
    if _patches_applied:
        for (module, name), original in _originals.items():
            setattr(module, name, original)
        torch.jit._state.enable()
        _patches_applied = False


def is_patching_enabled():
    """Check if patching is currently enabled.
    
    Returns:
        bool: True if patches are applied, False otherwise.
    """
    return _patches_applied


# Keep old name for backward compatibility
is_numpy_patching_enabled = is_patching_enabled


# =============================================================================
# TorchScript Detection Utilities
# =============================================================================

def warn_if_script_functions_in_scope():
    """Scan the caller's locals and globals for TorchScript ScriptFunctions.

    Called from input_tensors/output_tensors to warn users about pre-compiled
    ScriptFunctions that will silently break TracedTensor tracing (because
    ScriptFunction.__call__ bypasses __torch_function__).

    Only emits a warning — does not raise. Scans the immediate caller's
    frame (2 levels up: this function → input/output_tensors → user code).
    """
    frame = sys._getframe(2)  # skip this function + input/output_tensors
    suspects = []

    # Check locals
    for name, val in frame.f_locals.items():
        if isinstance(val, torch._C.ScriptFunction):
            suspects.append(f"{name} (local)")

    # Check globals (skip duplicates already found in locals)
    local_names = set(frame.f_locals.keys())
    for name, val in frame.f_globals.items():
        if name not in local_names and isinstance(val, torch._C.ScriptFunction):
            suspects.append(f"{name} (global)")

    if suspects:
        _get_logger().warning(
            f"Detected pre-compiled TorchScript ScriptFunction(s) in scope: {suspects}\n"
            "ScriptFunction.__call__ bypasses TracedTensor tracing — outputs will be "
            "plain torch.Tensors, breaking the trace chain.\n"
            "To fix this, either:\n"
            "  1. Call torch.jit._state.disable() before the @torch.jit.script "
            "decorators run (e.g. before importing the module that defines them)\n"
            "  2. Run the scripting code between leapp.start() and leapp.stop() signals"
            "(leapp disables JIT during tracing automatically)"
        )
