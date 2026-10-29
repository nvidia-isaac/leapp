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

from typing import Any, NamedTuple, Optional, Union

import numpy as np
import torch
from torch.fx.proxy import Proxy
from torch.multiprocessing.reductions import StorageWeakRef

from leapp.utils.dtype import value_to_name_and_shape


class ProxyView:
    """One mutable reference to the FX proxy that currently represents a value.

    ``proxy`` is ``None`` when the value carries no graph provenance, which is
    the normal state outside an active trace.

    During capture the proxy is a stable native tensor handle. Assigning a
    functional update records a copy into that handle, preserving relationships
    with existing views until PyTorch functionalizes the complete graph.
    An independent allocation gets a new cell instead of mutating this one.

    The stored cell may itself be a ``ProxyView`` so later nesting can redirect
    through another view without rewriting every carrier that already holds this
    object. Reading ``proxy`` always walks to the underlying ``fx.Proxy``.
    """

    __slots__ = ("_proxy", "pending")

    def __init__(self, proxy: Optional[Union[Proxy, "ProxyView"]]):
        self._proxy = proxy
        self.pending = False

    def rebind(self, proxy: Optional[Proxy]) -> None:
        """Replace provenance at a declaration rather than record a data write."""
        self._proxy = proxy
        self.pending = False

    @property
    def proxy(self) -> Optional[Proxy]:
        value = self._proxy
        while isinstance(value, ProxyView):
            value = value._proxy
        return value

    @proxy.setter
    def proxy(self, value: Optional[Union[Proxy, "ProxyView"]]) -> None:
        if self.pending:
            # A new Warp allocation borrows provenance until its segment
            # closes; that borrowed handle is not its storage destination.
            self._proxy = value
            self.pending = False
            return
        old = self.proxy
        new = value.proxy if isinstance(value, ProxyView) else value
        if (
            old is not None
            and new is not None
            and old.node.graph is new.node.graph
            and getattr(old.node.graph, "recording", False)
        ):
            if old.node is not new.node:
                mutation = old.tracer.create_proxy(
                    "call_method", "copy_", (old, new), {}
                )
                mutation.node.meta["leapp_mutation"] = True
                old.node.graph.requires_functionalization = True
            return
        self._proxy = value


STORAGE_LAYOUT_META_KEY = "leapp_storage_layout"


class TorchStorageLayout(NamedTuple):
    """Capture-pass allocation identity and layout, without retaining its tensor."""

    identity: StorageWeakRef
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype


def _record_storage_layout(carrier: Any, proxy: Optional[Proxy]) -> None:
    # Warp APIC can bind the allocation behind a partial Torch view. Record
    # covering graph values so the boundary can recover that allocation later.
    # Native tensors must not be retained here: every intermediate would stay
    # alive for the duration of capture. Weak storage identity also avoids
    # confusing an allocator-reused address with the original allocation.
    if proxy is None or not isinstance(carrier, torch.Tensor):
        return
    value = carrier.tensor
    storage = value.untyped_storage()
    if (
        value.storage_offset() == 0
        and value.is_contiguous()
        and value.numel() * value.element_size() == storage.nbytes()
    ):
        proxy.node.meta.setdefault(
            STORAGE_LAYOUT_META_KEY,
            TorchStorageLayout(
                StorageWeakRef(storage),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
            ),
        )


def bind_new_view(carrier: Any, name: str, context, proxy: Optional[Proxy]) -> None:
    """Attach a new root ``ProxyView`` holding ``proxy``."""
    carrier._name = name
    carrier._context = context
    carrier._proxy_view = ProxyView(proxy)
    carrier._output_port = None
    _record_storage_layout(carrier, proxy)


def bind_shared_view(carrier: Any, name: str, context, view: ProxyView) -> None:
    """Attach an existing ``ProxyView`` so ``carrier`` shares that cell."""
    carrier._name = name
    carrier._context = context
    carrier._proxy_view = view
    carrier._output_port = None
    _record_storage_layout(carrier, view.proxy)


def update_view_proxy(
    carrier: Any, name: str, context, proxy: Optional[Proxy], *, rebind=False
) -> None:
    """Write a new proxy into ``carrier``'s existing view after an in-place change."""
    carrier._name = name
    carrier._context = context
    if rebind:
        carrier._proxy_view.rebind(proxy)
        _record_storage_layout(carrier, proxy)
    else:
        carrier._proxy_view.proxy = proxy
    carrier._output_port = None


def share_view(carrier: Any, source: Any) -> None:
    """Make ``carrier`` use ``source``'s view, name, and context."""
    bind_shared_view(carrier, source.name, source.context_obj, source.proxy_view)


def conversion_proxy(source, result):
    """Preserve independent storage in representation-only conversions.

    Two separate proxy cells may initially hold the same graph value. Native
    mutation needs an actual graph copy as well when the eager conversion
    allocated, even if device, dtype and numerical values did not change.
    """
    proxy = source.proxy
    if proxy is None or not source.is_tracing:
        return proxy
    before, after = layout_key(source), layout_key(result)
    if (
        before is not None
        and after is not None
        and (before[0] != after[0] or before[-1] != after[-1])
    ):
        return source.context_obj.tracer.create_proxy(
            "call_function", torch.clone, (proxy,), {}
        )
    return proxy


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
            byte_strides = tuple(stride * element_size for stride in native.stride())
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
