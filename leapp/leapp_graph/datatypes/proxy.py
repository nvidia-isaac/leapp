#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""FX tensor handles, physical allocation snapshots, and conversion semantics.

Carriers store FX proxies directly. Numerical writes record copy_ against the
existing handle; PyTorch functionalization propagates them through native views.
Construction and input declarations bind provenance without a data write.
TracedTensorNode owns input redeclarations and its per-pass layout index;
TracedData.proxy resolves these bindings when retained aliases are next used.
Ports remain per carrier and are independent of its numerical storage.

TorchStorageLayout is a weak allocation snapshot on an FX node. Warp uses it
to bind the covering allocation behind partial views; graph finalization removes
it before export. It neither owns tensors nor propagates numerical mutations.
Layout comparisons also distinguish independent copying conversions, which must
allocate in the graph even when their sample values and dtypes are identical.
"""

from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from torch.fx.proxy import Proxy
from torch.multiprocessing.reductions import StorageWeakRef

from leapp.utils.dtype import value_to_name_and_shape


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
        # A pending Warp output only borrows its donor's node. Keep its weak
        # layout on the proxy until publication identifies the actual producer.
        metadata = (
            proxy.__dict__
            if proxy.__dict__.get("_leapp_pending_warp_output", False)
            else proxy.node.meta
        )
        metadata.setdefault(
            STORAGE_LAYOUT_META_KEY,
            TorchStorageLayout(
                StorageWeakRef(storage),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
            ),
        )


def bind_proxy(carrier: Any, name: str, context, proxy: Optional[Proxy]) -> None:
    """Bind provenance at construction or declaration, without a numerical write."""
    carrier._name = name
    carrier._context = context
    carrier._proxy = proxy
    carrier._output_port = None
    _record_storage_layout(carrier, proxy)


def update_proxy(carrier: Any, name: str, context, proxy: Optional[Proxy]) -> None:
    """Record a write against the stable FX tensor handle during capture."""
    old = carrier.proxy
    if (
        old is not None
        and proxy is not None
        and old.node.graph is proxy.node.graph
        and getattr(old.node.graph, "recording", False)
    ):
        if old.node is not proxy.node:
            mutation = old.tracer.create_proxy("call_method", "copy_", (old, proxy), {})
            mutation.node.meta["leapp_mutation"] = True
            old.node.graph.requires_functionalization = True
    else:
        carrier._proxy = proxy
    carrier._name = name
    carrier._context = context
    carrier._output_port = None


def conversion_proxy(source, result):
    """Preserve independent storage in representation-only conversions.

    Two carriers may initially hold the same graph value. Native
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
    # imports binding helpers from here).
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


def is_same_value(source, result) -> bool:
    """Whether an active, unpublished source describes exactly this layout."""
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
