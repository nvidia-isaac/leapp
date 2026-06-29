#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for torch functions used during LEAPP tracing."""

import functools

import torch

_TORCH_FUNCTIONS = [
    (torch, "from_numpy"),
    (torch, "as_tensor"),
    (torch, "tensor"),
]


def _create_torch_patch(original_func):
    """Create a patched version of a torch function that preserves tracing."""

    @functools.wraps(original_func)
    def patched(data, *args, **kwargs):
        from leapp.leapp_graph.datatypes import as_traced, is_traced_type

        is_traced_and_tracing = is_traced_type(data) and data.is_tracing

        if is_traced_and_tracing:
            name, context, proxy = data.name, data.context_obj, data.proxy
            raw_data = data.data
        else:
            raw_data = data

        result = original_func(raw_data, *args, **kwargs)

        if is_traced_and_tracing:
            if result is raw_data:
                return data
            result = as_traced(result, name, context, proxy)

        return result

    return patched


def build_torch_patches() -> tuple[dict, dict]:
    """Return ``(originals, patches)`` keyed by ``(module, name)``."""
    originals = {}
    patches = {}
    for module, name in _TORCH_FUNCTIONS:
        original = getattr(module, name)
        key = (module, name)
        originals[key] = original
        patches[key] = _create_torch_patch(original)
    return originals, patches
