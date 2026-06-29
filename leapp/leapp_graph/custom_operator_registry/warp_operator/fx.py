#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""FX / eager implementations for ``leapp::warp_runner``."""

from __future__ import annotations

import torch

from .schema import (
    QUALIFIED_NAME,
    decode_output_dtypes,
    decode_output_mask,
    decode_output_shapes,
    _resolve_dtype,
)


def warp_runner_fake(inputs, output_shapes, output_dtypes, output_mask, bundle):
    """Abstract impl: produce correctly-shaped meta outputs from the spec."""
    shape_lists = decode_output_shapes(output_shapes)
    dtype_lists = decode_output_dtypes(output_dtypes)
    mask_lists = decode_output_mask(output_mask) if output_mask else []

    if len(shape_lists) != len(dtype_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(shape_lists)}) and "
            f"output_dtypes ({len(dtype_lists)}) must have equal length"
        )
    if mask_lists and len(mask_lists) != len(shape_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_mask ({len(mask_lists)}) must match the "
            f"number of outputs ({len(shape_lists)}) when provided"
        )

    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.empty(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(shape_lists, dtype_lists)
    ]


def warp_runner_eager(inputs, output_shapes, output_dtypes, output_mask, bundle):
    """Eager kernel: allocate shape-correct (zeros) outputs from the spec."""
    shape_lists = decode_output_shapes(output_shapes)
    dtype_lists = decode_output_dtypes(output_dtypes)

    if len(shape_lists) != len(dtype_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_shapes ({len(shape_lists)}) and "
            f"output_dtypes ({len(dtype_lists)}) must have equal length"
        )
    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.zeros(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(shape_lists, dtype_lists)
    ]
