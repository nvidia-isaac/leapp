#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ProxyView cells and the free functions that bind, share, and classify them.

Traced carriers hold a ``ProxyView`` rather than an ``fx.Proxy`` so the proxy a
carrier reports can change without every reader holding a stale reference.
Torch, NumPy, and Warp can expose several logical arrays over one allocation
while FX proxies are immutable graph values; sharing one view is how a mutation
through one alias becomes visible through the others.

Binding primitives
------------------
``bind_new_view``
    Private root: new ``ProxyView(proxy)``.
``bind_shared_view``
    Attach an existing view object (zero-copy alias).
``update_view_proxy``
    Mutate the proxy inside the carrier's existing view (in-place change).
``share_view``
    ``bind_shared_view`` using ``source``'s name, context, and view.
``may_adopt_view``
    Whether two values cover the same bytes and policy allows sharing.
``layout_key``
    Framework-neutral description of the bytes a value covers, used both by
    ``may_adopt_view`` and by the per-node index that finds aliases created
    before tracing began.
"""

from typing import Any, Optional, Union

import numpy as np
import torch
from torch.fx.proxy import Proxy

from leapp.utils.dtype import value_to_name_and_shape


class ProxyView:
    """One mutable reference to the FX proxy that currently represents a value.

    ``proxy`` is ``None`` when the value carries no graph provenance, which is
    the normal state outside an active trace.

    Nested views over a single mutable root are how a mutation through one
    alias becomes visible to the others; parent links and forward/backward
    projection are added on this class later. Keeping every carrier behind it
    now means those additions change this class and the sites that create
    aliases, not every proxy reader.

    Assigning ``proxy`` means this value was mutated in place. An out-of-place
    operation produces an independent value and therefore a new ``ProxyView``.

    The stored cell may itself be a ``ProxyView`` so later nesting can redirect
    through another view without rewriting every carrier that already holds this
    object. Reading ``proxy`` always walks to the underlying ``fx.Proxy``.
    """

    __slots__ = ("_proxy",)

    def __init__(self, proxy: Optional[Union[Proxy, "ProxyView"]]):
        self._proxy = proxy

    @property
    def proxy(self) -> Optional[Proxy]:
        value = self._proxy
        while isinstance(value, ProxyView):
            value = value._proxy
        return value

    @proxy.setter
    def proxy(self, value: Optional[Union[Proxy, "ProxyView"]]) -> None:
        self._proxy = value


def bind_new_view(carrier: Any, name: str, context, proxy: Optional[Proxy]) -> None:
    """Attach a new root ``ProxyView`` holding ``proxy``."""
    carrier._name = name
    carrier._context = context
    carrier._proxy_view = ProxyView(proxy)
    carrier._output_port = None


def bind_shared_view(carrier: Any, name: str, context, view: ProxyView) -> None:
    """Attach an existing ``ProxyView`` so ``carrier`` shares that cell."""
    carrier._name = name
    carrier._context = context
    carrier._proxy_view = view
    carrier._output_port = None


def update_view_proxy(
    carrier: Any, name: str, context, proxy: Optional[Proxy]
) -> None:
    """Write a new proxy into ``carrier``'s existing view after an in-place change."""
    carrier._name = name
    carrier._context = context
    carrier._proxy_view.proxy = proxy
    carrier._output_port = None


def share_view(carrier: Any, source: Any) -> None:
    """Make ``carrier`` use ``source``'s view, name, and context."""
    bind_shared_view(carrier, source.name, source.context_obj, source.proxy_view)


def _contiguous_byte_strides(shape: tuple, dtype_name: str) -> tuple:
    """Row-major byte strides for ``shape`` at ``dtype_name``'s element size."""
    itemsize = np.dtype(dtype_name).itemsize
    strides = []
    step = itemsize
    for dim in reversed(shape):
        strides.append(step)
        step *= dim
    return tuple(reversed(strides))


def layout_key(value):
    """Framework-neutral description of the bytes ``value`` covers.

    Returns ``None`` when the layout cannot be established, which callers treat
    as "not an alias" so an unrecognized value fails closed rather than
    comparing equal to something it does not alias.
    """
    # Lazy: this module must not import TracedData at load time (TracedData
    # imports ProxyView from here).
    from .traced_data import TracedData

    native = value.data if isinstance(value, TracedData) else value

    if isinstance(native, torch.Tensor):
        try:
            pointer = native.data_ptr()
            element_size = native.element_size()
            byte_strides = tuple(
                stride * element_size for stride in native.stride()
            )
            contiguous = native.is_contiguous()
        except RuntimeError:
            # Layouts without strides or an addressable buffer, e.g. sparse.
            return None
        device = str(native.device)
    elif isinstance(native, np.ndarray):
        pointer = native.ctypes.data
        byte_strides = tuple(native.strides)
        contiguous = native.flags["C_CONTIGUOUS"]
        device = "cpu"
    elif hasattr(native, "ptr") and hasattr(native, "is_contiguous"):
        # Warp arrays, duck-typed so this module never imports warp-lang. A real
        # import here would run Warp's initialization during core datatype
        # import, and warp is an optional dependency.
        pointer = native.ptr
        byte_strides = tuple(native.strides or ())
        contiguous = bool(native.is_contiguous)
        device = str(native.device)
    else:
        return None

    # A null pointer means nothing was allocated, and an allocator may hand the
    # same address to several zero-byte requests, so unrelated values would
    # compare equal.
    if not pointer:
        return None

    try:
        dtype_name, storage_shape = value_to_name_and_shape(native)
    except ValueError:
        return None
    storage_shape = tuple(int(dim) for dim in storage_shape)
    if 0 in storage_shape:
        return None

    if len(byte_strides) != len(storage_shape):
        # A compound element dtype such as ``wp.vec3`` reports strides for its
        # outer dimensions only, while the scalar shape includes the component
        # dimensions. Only a contiguous layout can be restated in scalar terms.
        if not contiguous:
            return None
        byte_strides = _contiguous_byte_strides(storage_shape, dtype_name)

    return (device, dtype_name, storage_shape, byte_strides, pointer)


def may_adopt_view(source, result) -> bool:
    """Whether ``result`` may share ``source``'s view instead of a new root.

    True when the two cover exactly the same bytes with the same interpretation
    and policy allows sharing. Sharing an allocation is not enough on its own:
    ``view`` and storage-sharing ``reshape`` keep the pointer and change the
    shape, and letting those share a root would leave a carrier reporting a
    proxy of the wrong shape, so the shape and stride comparisons are what keep
    them out.

    A source is excluded even when the layouts match if it is:

    - a published value, because it has to stay available to fan out to other
      nodes and sharing would let a consumer of one alias rewrite the
      provenance the boundary depends on;
    - a value whose node has finished, because rewrapping it at a boundary
      hands a consumer its own carrier rather than a second handle on a value
      still being built.
    """
    from .traced_data import TracedData

    if not isinstance(source, TracedData):
        return False
    if source.output_port is not None:
        return False
    if not source.is_tracing:
        return False
    source_key = layout_key(source)
    if source_key is None:
        return False
    return source_key == layout_key(result)
