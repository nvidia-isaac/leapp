#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for numpy functions used during LEAPP tracing."""

import functools

import numpy as np

_NUMPY_FUNCTIONS = [
    (np, "array"),
    (np, "asarray"),
]


def _create_numpy_patch(original_func):
    """Create a patched numpy array function that preserves tracing."""

    @functools.wraps(original_func)
    def patched(data, *args, **kwargs):
        from leapp.leapp_graph.datatypes import is_traced_type

        is_traced_and_tracing = is_traced_type(data) and data.is_tracing

        if is_traced_and_tracing:
            dtype = kwargs.get("dtype")
            copy = kwargs.get("copy")
            return data.__array__(dtype=dtype, copy=copy)

        return original_func(data, *args, **kwargs)

    return patched


def build_numpy_patches() -> tuple[dict, dict]:
    """Return ``(originals, patches)`` keyed by ``(module, name)``."""
    originals = {}
    patches = {}
    for module, name in _NUMPY_FUNCTIONS:
        original = getattr(module, name)
        key = (module, name)
        originals[key] = original
        patches[key] = _create_numpy_patch(original)
    return originals, patches
