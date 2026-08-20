#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp tracing backend — single optional-import gate for warp-lang."""

from .warp_segment import WarpSegment, WarpTensorRef

try:
    import warp as wp  # noqa: F401

    # APIC export bundles embed compiled module binaries. Force PTX at sm75 so
    # saved .wrp segments are portable across CUDA-capable GPUs (PTX is JIT'd
    # at load time). Must be set before kernel compilation / wp.init().
    wp.config.cuda_output = "ptx"
    wp.config.ptx_target_arch = 75

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
