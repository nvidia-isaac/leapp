#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""NumPy to PyTorch compatibility utilities.

NOTE: The main numpy-to-torch mappings (NUMPY_UFUNC_TO_TORCH, NUMPY_FUNC_TO_TORCH,
AXIS_TO_DIM_FUNCTIONS) have been moved to traced_np_array.py for direct access.

This module now only contains utility functions for numpy argument conversion.
"""

import numpy as np
import torch


def convert_numpy_arg_to_torch(arg, device=None):
    """Convert numpy arrays to torch tensors recursively.
    
    Args:
        arg: A value that may be a numpy array, list/tuple of arrays, or dict
        device: Optional torch device to move tensors to
        
    Returns:
        The same structure with numpy arrays converted to torch tensors
    """
    if isinstance(arg, np.ndarray):
        tensor = torch.from_numpy(arg)
        return tensor.to(device) if device else tensor
    elif isinstance(arg, (list, tuple)):
        converted = [convert_numpy_arg_to_torch(a, device) for a in arg]
        return type(arg)(converted)
    elif isinstance(arg, dict):
        return {k: convert_numpy_arg_to_torch(v, device) for k, v in arg.items()}
    return arg
