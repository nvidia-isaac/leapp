#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp tracing backend — single optional-import gate for warp-lang."""

from .warp_segment import WarpSegment, WarpTensorRef

try:
    import warp as wp  # noqa: F401

    from .patching import WarpPatchBackend
    from .traced_wp_array import TracedWpArray
except ImportError:
    wp = None
    TracedWpArray = None
    WarpPatchBackend = None

__all__ = [
    "WarpSegment",
    "WarpTensorRef",
    "TracedWpArray",
    "WarpPatchBackend",
    "wp",
]
