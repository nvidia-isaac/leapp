#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Data type compatibility utilities for LEAPP tracing."""

from .numpy_compatibility import (
    NUMPY_UFUNC_TO_TORCH,
    NUMPY_FUNC_TO_TORCH,
    AXIS_TO_DIM_FUNCTIONS,
    convert_numpy_arg_to_torch,
    get_torch_equivalent_ufunc,
    get_torch_equivalent_func,
)

from .components import (
    TorchDtypeWrapper,
)

from .traced_tensor import (
    TracedTensor,
    apply_traced_tensor_patches,
    remove_traced_tensor_patches,
)

__all__ = [
    # Core data types
    "TracedTensor",
    # Patch management
    "apply_traced_tensor_patches",
    "remove_traced_tensor_patches",
    # NumPy compatibility
    "NUMPY_UFUNC_TO_TORCH",
    "NUMPY_FUNC_TO_TORCH",
    "AXIS_TO_DIM_FUNCTIONS",
    "convert_numpy_arg_to_torch",
    "get_torch_equivalent_ufunc",
    "get_torch_equivalent_func",
    # Components
    "TorchDtypeWrapper",
]
