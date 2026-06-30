#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for torch functions used during LEAPP tracing."""

import functools

import torch


class TorchPatchBackend:
    """Apply and restore torch conversion patches for a tracing session."""

    _FUNCTIONS = [
        (torch, "from_numpy"),
        (torch, "as_tensor"),
        (torch, "tensor"),
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
        torch.jit._state.disable()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for (module, name), original in self._originals.items():
            setattr(module, name, original)
        torch.jit._state.enable()
        self._installed = False

    @staticmethod
    def _create_patch(original_func):
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
