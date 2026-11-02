#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for numpy functions used during LEAPP tracing."""

import functools

from ..primitives.traced_int import find_recording_int

import numpy as np

from .._attribute_patching import AttributePatchRegistry
from leapp.utils.shape_values import trace_numpy_factory


class NumpyPatchBackend:
    """Apply and restore numpy conversion patches for a tracing session."""

    _FUNCTIONS = [
        (np, "array"),
        (np, "asarray"),
    ]

    def __init__(self) -> None:
        self._patches = AttributePatchRegistry()
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        try:
            for module, name in self._FUNCTIONS:
                original = getattr(module, name)
                # np.array copies by default; np.asarray copies only when needed.
                patched = self._create_patch(
                    original,
                    copy_default=True if name == "array" else None,
                )
                self._patches.install(module, name, original, patched)
            for name in ("zeros", "ones", "full", "arange"):
                original = getattr(np, name)
                self._patches.install(np, name, original, self._shape_patch(original))
        except Exception:
            self._patches.restore()
            raise
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._patches.restore()
        self._installed = False

    @staticmethod
    def _create_patch(original_func, *, copy_default):
        """Create a patched numpy array function that preserves tracing."""

        @functools.wraps(original_func)
        def patched(data, *args, **kwargs):
            from leapp.leapp_graph.datatypes import is_traced_type

            if not is_traced_type(data):
                return original_func(data, *args, **kwargs)

            # dtype is the only positional argument np.array and np.asarray share.
            dtype = args[0] if args else kwargs.get("dtype")
            return data.__array__(dtype=dtype, copy=kwargs.get("copy", copy_default))

        return patched

    @staticmethod
    def _shape_patch(original):
        @functools.wraps(original)
        def patched(*args, **kwargs):
            if (shape := find_recording_int((args, kwargs))) is not None:
                return trace_numpy_factory(original, args, kwargs, shape=shape)
            return original(*args, **kwargs)

        return patched
