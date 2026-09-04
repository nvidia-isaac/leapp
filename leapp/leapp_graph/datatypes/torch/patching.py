#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for torch functions used during LEAPP tracing."""

import functools

import torch

from .._attribute_patching import AttributePatchRegistry


class TorchPatchBackend:
    """Apply and restore torch conversion patches for a tracing session."""

    _FUNCTIONS = [
        (torch, "from_numpy"),
        (torch, "as_tensor"),
        (torch, "tensor"),
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
                patched = self._create_patch(original)
                self._patches.install(module, name, original, patched)
            torch.jit._state.disable()
        except Exception:
            self._patches.restore()
            raise
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._patches.restore()
        torch.jit._state.enable()
        self._installed = False

    @staticmethod
    def _create_patch(original_func):
        """Create a patched version of a torch function that preserves tracing."""

        @functools.wraps(original_func)
        def patched(data, *args, **kwargs):
            from leapp.leapp_graph.datatypes import as_traced, is_traced_type

            is_traced = is_traced_type(data)

            if is_traced:
                name, context, proxy = data.name, data.context_obj, data.proxy
                raw_data = data.data
            else:
                raw_data = data

            result = original_func(raw_data, *args, **kwargs)

            if is_traced:
                if result is raw_data:
                    return data
                result = as_traced(result, name, context, proxy)
                # A conversion that keeps shape and dtype presents the same
                # data, so it also presents the source's boundary identity.
                if (
                    tuple(result.shape) == tuple(data.shape)
                    and result.get_dtype_name() == data.get_dtype_name()
                ):
                    return data.preserve_port(result)

            return result

        return patched
