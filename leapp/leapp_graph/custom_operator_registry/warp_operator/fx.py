#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""FX / eager implementations for ``leapp::warp_runner``."""

from __future__ import annotations

import torch

from .metadata import decode_runtime_metadata, output_dtypes, output_shapes
from .schema import QUALIFIED_NAME, _resolve_dtype


def _decode_output_spec(runtime_metadata: str) -> tuple[list[list[int]], list[str]]:
    metadata = decode_runtime_metadata(runtime_metadata)
    shape_lists = output_shapes(metadata)
    dtype_lists = output_dtypes(metadata)
    if len(shape_lists) != len(dtype_lists):
        raise ValueError(
            f"{QUALIFIED_NAME}: runtime_metadata output shapes ({len(shape_lists)}) "
            f"and dtypes ({len(dtype_lists)}) must have equal length"
        )
    return shape_lists, dtype_lists


def warp_runner_fake(inputs, runtime_metadata, bundle):
    """Abstract impl: produce correctly-shaped meta outputs from runtime metadata."""
    shape_lists, dtype_lists = _decode_output_spec(runtime_metadata)
    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.empty(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(shape_lists, dtype_lists)
    ]


def warp_runner_eager(inputs, runtime_metadata, bundle):
    """Eager kernel: allocate shape-correct zero outputs from runtime metadata."""
    shape_lists, dtype_lists = _decode_output_spec(runtime_metadata)
    device = inputs[0].device if len(inputs) > 0 else torch.device("cpu")
    return [
        torch.zeros(list(shape), dtype=_resolve_dtype(name), device=device)
        for shape, name in zip(shape_lists, dtype_lists)
    ]
