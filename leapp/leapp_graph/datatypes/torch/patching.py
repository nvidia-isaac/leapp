#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deferred patches for torch functions used during LEAPP tracing."""

import functools

import torch

from .._attribute_patching import AttributePatchRegistry
from ..proxy_view import conversion_proxy


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
            from leapp.leapp_graph.datatypes import (
                TracedData,
                TracedInt,
                TracedTensor,
                as_traced,
                is_traced_type,
            )
            from leapp.leapp_graph.datatypes.primitives.traced_int import (
                find_recording_int,
            )

            # A tensor built from traced integers, e.g. torch.tensor(x.shape),
            # must read them at run time, so the call itself is recorded.
            size = find_recording_int(data)
            if size is not None and size.validate_status((data,)):
                result = original_func(data, *args, **kwargs)
                proxy = _record_tensor_from_ints(
                    size.context_obj.tracer, data, result.dtype, result.device
                )
                traced = TracedTensor(
                    result, TracedData._name_from_proxy(proxy), size.context_obj, proxy
                )
                traced._data_dependent_shape = True
                return traced
            if isinstance(data, TracedInt):
                # An integer that no longer records is a plain value.
                data = int(data)

            is_traced = is_traced_type(data)

            if is_traced:
                name, context = data.name, data.context_obj
                raw_data = data.data
            else:
                raw_data = data

            result = original_func(raw_data, *args, **kwargs)

            if is_traced:
                if result is raw_data:
                    return data
                result = as_traced(
                    result, name, context, conversion_proxy(data, result)
                )
                # A conversion that keeps shape and dtype presents the same
                # data, so it also presents the source's boundary identity.
                if (
                    tuple(result.shape) == tuple(data.shape)
                    and result.get_dtype_name() == data.get_dtype_name()
                ):
                    return data.preserve_port(result)

            return result

        return patched


def _record_tensor_from_ints(tracer, data, dtype, device):
    """Record ``torch.tensor(data)`` for nested traced integers.

    Exporters that trace the FX graph (TorchScript) keep ``torch.tensor`` of
    traced integers as a constant, but they do trace ``scalar_tensor`` and
    ``stack``; the tensor is rebuilt from those.
    """
    from leapp.leapp_graph.datatypes.primitives.traced_int import graph_value

    if isinstance(data, (list, tuple)):
        items = [_record_tensor_from_ints(tracer, item, dtype, device) for item in data]
        return tracer.create_proxy("call_function", torch.stack, (items,), {})
    return tracer.create_proxy(
        "call_function",
        torch.scalar_tensor,
        (graph_value(data),),
        {"dtype": dtype, "device": device},
    )
