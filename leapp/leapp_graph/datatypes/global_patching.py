#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patching for torch functions that bypass __torch_function__.

Several torch functions do early type checks in C++ before __torch_function__
can intercept them. We use deferred patching to avoid conflicts with TorchScript
modules that are compiled at import time.

Patches are applied when tracing starts (annotate.start()) and removed when
tracing stops (annotate.stop()).
"""

import torch

# Store original torch functions before patching
_original_torch_from_numpy = torch.from_numpy
_original_torch_as_tensor = torch.as_tensor
_original_torch_tensor = torch.tensor

_patches_applied = False


def _patched_from_numpy(arr):
    """Patched torch.from_numpy that handles TracedTensor.
    
    If the input is a TracedTensor that is actively tracing, return it directly
    (it's already a traced torch tensor). Otherwise, call the original torch.from_numpy.
    """
    # Lazy import to avoid circular dependency
    from .traced_tensor import TracedTensor
    
    if isinstance(arr, TracedTensor) and arr.is_tracing:
        return arr
    return _original_torch_from_numpy(arr)


def _patched_as_tensor(data, dtype=None, device=None):
    """Patched torch.as_tensor that handles TracedTensor.
    
    If the input is a TracedTensor that is actively tracing, return it directly
    (preserving tracing). Otherwise, call the original torch.as_tensor.
    """
    # Lazy import to avoid circular dependency
    from .traced_tensor import TracedTensor
    
    if isinstance(data, TracedTensor) and data.is_tracing:
        # If dtype or device conversion is requested, we need to trace that
        if dtype is not None or device is not None:
            return data.to(dtype=dtype, device=device)
        return data
    return _original_torch_as_tensor(data, dtype=dtype, device=device)


def _patched_tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """Patched torch.tensor that handles TracedTensor.
    
    If the input is a TracedTensor that is actively tracing, return it directly
    (preserving tracing). Otherwise, call the original torch.tensor.
    
    Note: Uses explicit signature (not *args, **kwargs) to be TorchScript-compatible.
    """
    # Lazy import to avoid circular dependency
    from .traced_tensor import TracedTensor
    
    if isinstance(data, TracedTensor) and data.is_tracing:
        # torch.tensor always copies, but for TracedTensor we want to preserve tracing
        # Handle dtype/device if specified
        if dtype is not None or device is not None:
            return data.to(dtype=dtype, device=device)
        return data
    return _original_torch_tensor(data, dtype=dtype, device=device,
                                   requires_grad=requires_grad, pin_memory=pin_memory)


def apply_traced_tensor_patches():
    """Apply patches for torch functions that bypass __torch_function__.
    
    Call this when tracing starts to enable TracedTensor compatibility with
    functions like torch.as_tensor, torch.tensor, and torch.from_numpy.
    """
    global _patches_applied
    if not _patches_applied:
        torch.from_numpy = _patched_from_numpy
        torch.as_tensor = _patched_as_tensor
        torch.tensor = _patched_tensor
        _patches_applied = True


def remove_traced_tensor_patches():
    """Remove patches for torch functions.
    
    Call this when tracing stops to restore original torch function behavior.
    This prevents conflicts with TorchScript compilation.
    """
    global _patches_applied
    if _patches_applied:
        torch.from_numpy = _original_torch_from_numpy
        torch.as_tensor = _original_torch_as_tensor
        torch.tensor = _original_torch_tensor
        _patches_applied = False


def is_numpy_patching_enabled():
    """Check if numpy patching is currently enabled.
    
    Returns:
        bool: True if numpy patches are applied, False otherwise.
    """
    return _patches_applied

