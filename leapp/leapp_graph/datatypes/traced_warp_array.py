#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""TracedWarpArray - A warp array subclass for LEAPP tracing."""

from abc import ABCMeta
import inspect
from typing import Callable, Optional

import torch
from torch.fx.proxy import Proxy

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "traced_warp_array requires warp-lang (pip install warp-lang)."
    ) from exc

from .traced_data import TracedData


_ArrayInterfaceAccessHook = Callable[["TracedWarpArray", str, object], None]
_array_interface_access_hook: Optional[_ArrayInterfaceAccessHook] = None


def leapp_warp_launch(metadata: dict, traced_arrays: tuple = ()):  # pragma: no cover - FX marker only
    """FX marker for a trace-time Warp kernel launch.

    Runtime replay/export code should read ``metadata`` and bind ``traced_arrays``
    by the ``traced_index`` values embedded in the metadata. The Python function
    itself is a side-effect marker and is not intended to execute as computation.
    """

    return None


def set_array_interface_access_hook(hook: Optional[_ArrayInterfaceAccessHook]):
    """Install a legacy debug hook for traced Warp array interface access.

    The production trace-time design now uses global Warp function profiling.
    Array-interface access no longer records FX markers automatically; this hook
    is retained only for old experiments that explicitly opt into debug prints.
    """

    global _array_interface_access_hook
    previous = _array_interface_access_hook
    _array_interface_access_hook = hook
    return previous


def _notify_array_interface_access(array: "TracedWarpArray", interface_name: str) -> None:
    hook = _array_interface_access_hook
    if hook is None:
        return

    frame = inspect.currentframe()
    caller_frame = None
    try:
        caller_frame = frame.f_back if frame is not None else None
        hook(array, interface_name, caller_frame)
    finally:
        del caller_frame
        del frame



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
    def __array_interface__(self):
        """Expose CPU storage and notify optional trace-time consumers."""
        _notify_array_interface_access(self, "__array_interface__")
        return wp.array.__array_interface__.fget(self)

    @property
    def __cuda_array_interface__(self):
        """Expose CUDA storage and notify optional trace-time consumers."""
        _notify_array_interface_access(self, "__cuda_array_interface__")
        return wp.array.__cuda_array_interface__.fget(self)

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
