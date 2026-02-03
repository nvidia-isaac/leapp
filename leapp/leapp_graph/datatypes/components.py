#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Data type component wrappers for LEAPP tracing compatibility.

This module provides wrapper classes that bridge differences between
PyTorch and NumPy data types, enabling TracedTensor to work with
libraries that expect NumPy-style attributes.
"""

import torch


class TorchDtypeWrapper:
    """Wrapper around torch.dtype that provides numpy-compatible attributes.
    
    This allows TracedTensor to work with libraries like TensorDict that
    check numpy-style dtype attributes like .kind.
    
    The wrapper passes isinstance(wrapper, torch.dtype) checks by overriding
    __class__, allowing it to be used transparently where torch.dtype is expected.
    
    Example:
        wrapper = TorchDtypeWrapper(torch.float32)
        wrapper.kind  # Returns 'f' (float)
        wrapper == torch.float32  # Returns True
        isinstance(wrapper, torch.dtype)  # Returns True
    
    Attributes:
        kind: NumPy-style kind character ('f', 'i', 'u', 'b', 'c', 'O')
        itemsize: Size of one element in bytes
        name: Dtype name string (e.g., 'float32')
        str: NumPy-style string representation (e.g., '<f4')
    """
    
    # Mapping from torch dtype to numpy kind character
    # See: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
    TORCH_TO_NUMPY_KIND = {
        # Floating point
        torch.float16: 'f',
        torch.float32: 'f',
        torch.float64: 'f',
        torch.bfloat16: 'f',
        # Signed integers
        torch.int8: 'i',
        torch.int16: 'i',
        torch.int32: 'i',
        torch.int64: 'i',
        # Unsigned integers
        torch.uint8: 'u',
        torch.uint16: 'u',
        torch.uint32: 'u',
        torch.uint64: 'u',
        # Boolean
        torch.bool: 'b',
        # Complex
        torch.complex64: 'c',
        torch.complex128: 'c',
    }
    
    # Mapping from torch dtype to numpy itemsize (bytes)
    TORCH_TO_ITEMSIZE = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.uint16: 2,
        torch.uint32: 4,
        torch.uint64: 8,
        torch.bool: 1,
        torch.complex64: 8,
        torch.complex128: 16,
    }
    
    def __init__(self, torch_dtype):
        """Initialize with a torch dtype.
        
        Args:
            torch_dtype: A torch.dtype object (e.g., torch.float32)
        """
        self._torch_dtype = torch_dtype
    
    @property
    def __class__(self):
        """Return torch.dtype to pass isinstance checks.
        
        This allows isinstance(wrapper, torch.dtype) to return True,
        enabling transparent use where torch.dtype is expected.
        """
        return torch.dtype
    
    @property
    def kind(self):
        """Return numpy-style kind character.
        
        Returns:
            str: Single character indicating dtype kind:
                'f' = floating-point
                'i' = signed integer
                'u' = unsigned integer
                'b' = boolean
                'c' = complex
                'O' = object (fallback)
        """
        return self.TORCH_TO_NUMPY_KIND.get(self._torch_dtype, 'O')
    
    @property
    def itemsize(self):
        """Return size of one element in bytes (numpy compatibility)."""
        return self.TORCH_TO_ITEMSIZE.get(self._torch_dtype, 0)
    
    @property
    def name(self):
        """Return dtype name string (numpy compatibility)."""
        return str(self._torch_dtype).replace('torch.', '')
    
    @property 
    def str(self):
        """Return dtype string representation (numpy compatibility)."""
        return '<' + self.kind + str(self.itemsize)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying torch dtype."""
        return getattr(self._torch_dtype, name)
    
    def __eq__(self, other):
        """Compare with other dtypes."""
        if type(other).__name__ == 'TorchDtypeWrapper':
            return self._torch_dtype == other._torch_dtype
        return self._torch_dtype == other
    
    def __ne__(self, other):
        """Not equal comparison."""
        return not self.__eq__(other)
    
    def __hash__(self):
        """Hash based on underlying torch dtype."""
        return hash(self._torch_dtype)
    
    def __repr__(self):
        """String representation."""
        return repr(self._torch_dtype)
    
    def __str__(self):
        """String conversion."""
        return str(self._torch_dtype)
