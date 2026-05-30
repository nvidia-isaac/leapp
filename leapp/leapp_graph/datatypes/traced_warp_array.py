#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""TracedWarpArray - A warp array subclass for LEAPP tracing."""

from abc import ABCMeta

import torch
from torch.fx.proxy import Proxy

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "traced_warp_array requires warp-lang (pip install warp-lang)."
    ) from exc

from .traced_data import TracedData


class _TracedWarpArrayMeta(ABCMeta, type(wp.array)):
    """Metaclass combining ABCMeta and wp.array's metaclass."""

    pass


class TracedWarpArray(TracedData, wp.array, metaclass=_TracedWarpArrayMeta):
    """A warp array subclass that records operations using torch.fx.

    TracedWarpArrays must be created via TraceContext.create_input() / as_traced().
    """

    def __new__(cls, array: wp.array, name: str, context, proxy: Proxy):
        """Share storage with an existing wp.array."""
        obj = wp.array.__new__(cls)
        wp.array.__init__(
            obj,
            dtype=array.dtype,
            shape=array.shape,
            ptr=array.ptr,
            device=array.device,
            copy=False,
        )
        obj._name = name
        obj._context = context
        obj._proxy = proxy
        return obj

    def __init__(self, array: wp.array, name: str, context, proxy: Proxy):
        """Attributes are initialized in __new__."""

    @property
    def tensor(self) -> torch.Tensor:
        """Get the underlying data as a torch.Tensor."""
        return torch.from_numpy(self.numpy())

    @property
    def data(self) -> wp.array:
        """Get the underlying wp.array."""
        return self

    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy

    @property
    def name(self) -> str:
        """Get the name of the array."""
        return self._name

    @property
    def context(self) -> str:
        """Get the name of the context that owns this array."""
        if self._context is None:
            return "untraced"
        return self._context.name

    @property
    def context_obj(self):
        """Get the TraceContext that owns this array."""
        return self._context

    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the context."""
        if self._context is None:
            return False
        return self._context.is_tracing

    def _new(self, array: wp.array, proxy: Proxy = None) -> "TracedWarpArray":
        """Create a new TracedWarpArray in the same context."""
        if proxy is not None:
            intermediate_name = str(proxy.node.name)
        else:
            intermediate_name = "untraced"
        return TracedWarpArray(array, intermediate_name, self._context, proxy)
