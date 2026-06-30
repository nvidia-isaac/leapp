#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for numpy functions used during LEAPP tracing."""

import functools

import numpy as np


class NumpyPatchBackend:
    """Apply and restore numpy conversion patches for a tracing session."""

    _FUNCTIONS = [
        (np, "array"),
        (np, "asarray"),
    ]

    def __init__(self) -> None:
        self._originals: dict[tuple[object, str], object] = {}
        self._patches: dict[tuple[object, str], object] = {}
        self._installed = False
        for module, name in self._FUNCTIONS:
            original = getattr(module, name)
            key = (module, name)
            self._originals[key] = original
            self._patches[key] = self._create_patch(original)

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        for (module, name), patched in self._patches.items():
            setattr(module, name, patched)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for (module, name), original in self._originals.items():
            setattr(module, name, original)
        self._installed = False

    @staticmethod
    def _create_patch(original_func):
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
