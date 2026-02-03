#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""NumPy to PyTorch compatibility layer for LEAPP tracing.

This module provides mappings and utilities to convert numpy operations to their
PyTorch equivalents, enabling transparent tracing through code that uses numpy.

The key insight is that TracedTensor can intercept numpy operations via:
- __array_ufunc__: For element-wise operations (np.sin, np.add, etc.)
- __array_function__: For array operations (np.sum, np.concatenate, etc.)

By mapping these to torch equivalents, we can trace through numpy code seamlessly.
"""

import numpy as np
import torch


# =============================================================================
# NumPy Ufunc to Torch Mapping
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


# =============================================================================
# NumPy Function to Torch Mapping
# =============================================================================
# These are higher-level array functions intercepted via __array_function__ protocol.

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


# =============================================================================
# Functions that need axis -> dim conversion
# =============================================================================

AXIS_TO_DIM_FUNCTIONS = {
    torch.sum,
    torch.mean,
    torch.std,
    torch.var,
    torch.min,
    torch.max,
    torch.amin,  # Used for np.min (returns values only, unlike torch.min with dim)
    torch.amax,  # Used for np.max (returns values only, unlike torch.max with dim)
    torch.argmin,
    torch.argmax,
    torch.cumsum,
    torch.cumprod,
    torch.all,
    torch.any,
    torch.cat,  # concatenate
    torch.squeeze,
    torch.unsqueeze,
    torch.flip,
    torch.roll,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_torch_equivalent_ufunc(ufunc):
    """Get the torch equivalent of a numpy ufunc.
    
    Args:
        ufunc: A numpy ufunc (e.g., np.sin, np.add)
        
    Returns:
        The equivalent torch function, or None if not found
    """
    return NUMPY_UFUNC_TO_TORCH.get(ufunc)


def get_torch_equivalent_func(func):
    """Get the torch equivalent of a numpy function.
    
    Args:
        func: A numpy function (e.g., np.sum, np.concatenate)
        
    Returns:
        The equivalent torch function, or None if not found
    """
    return NUMPY_FUNC_TO_TORCH.get(func)


def convert_numpy_arg_to_torch(arg, device=None):
    """Convert a numpy array or nested structure to torch tensors.
    
    Args:
        arg: A numpy array, list, tuple, or other value
        device: Target device for converted tensors (optional)
        
    Returns:
        Converted value with numpy arrays replaced by torch tensors
    """
    if isinstance(arg, np.ndarray):
        tensor = torch.from_numpy(arg.copy())
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    elif isinstance(arg, (list, tuple)):
        converted = [convert_numpy_arg_to_torch(a, device) for a in arg]
        return type(arg)(converted)
    elif isinstance(arg, dict):
        return {k: convert_numpy_arg_to_torch(v, device) for k, v in arg.items()}
    return arg
