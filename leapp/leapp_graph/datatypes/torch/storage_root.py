#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Track which traced Torch tensor covers the storage behind a partial view.

Warp APIC records a Torch-backed array as its whole storage allocation, so a
segment reading ``wp.from_torch(traced[5:10])`` binds the full buffer, not the
five-element view. The runner therefore has to receive a value covering the
whole storage. A view remembers the traced tensor that covered its storage when
it was taken, together with both proxies at that moment, so the segment can
substitute that tensor and detect when either side changed since.

Mutations through one alias do not update the proxies of the others, so the
substitution is only valid while neither proxy changed. Callers must fail
rather than export a stale buffer when :func:`valid_storage_root` returns
``None`` for a partial view.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.fx.proxy import Proxy

_ATTRIBUTE = "_leapp_storage_root"


@dataclass(frozen=True)
class StorageRoot:
    # Traced tensor covering every byte of the view's storage.
    tensor: Any
    # ``tensor``'s proxy when the view was taken.
    tensor_proxy: Proxy
    # The view's proxy when it was taken.
    view_proxy: Proxy


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    # Layout queries on a traced tensor would be recorded as graph operations.
    return tensor.as_subclass(torch.Tensor)


def covers_storage(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is a contiguous view of its entire storage."""
    tensor = _plain(tensor)
    return (
        tensor.storage_offset() == 0
        and tensor.is_contiguous()
        and tensor.numel() * tensor.element_size()
        == tensor.untyped_storage().nbytes()
    )


def shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    first, second = _plain(first), _plain(second)
    return (
        first.device == second.device
        and first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    )


def get_storage_root(carrier: Any) -> Optional[StorageRoot]:
    return carrier.__dict__.get(_ATTRIBUTE)


def set_storage_root(carrier: Any, root: Optional[StorageRoot]) -> None:
    if root is None:
        carrier.__dict__.pop(_ATTRIBUTE, None)
    else:
        carrier.__dict__[_ATTRIBUTE] = root


def valid_storage_root(carrier: Any) -> Optional[StorageRoot]:
    """The carrier's storage root if neither side changed since the view was taken."""
    root = get_storage_root(carrier)
    if root is None:
        return None
    if carrier.proxy is not root.view_proxy or root.tensor.proxy is not root.tensor_proxy:
        return None
    return root


def record_view(view: Any, source: Any) -> None:
    """Remember the storage root of ``view``, a traced result taken from ``source``.

    ``source`` is the traced tensor the operation read. Results that own their
    storage, or that cover all of it, need no root.
    """
    if covers_storage(view) or not shares_storage(view, source):
        return
    if covers_storage(source):
        root_tensor = source
    else:
        source_root = valid_storage_root(source)
        if source_root is None:
            return
        root_tensor = source_root.tensor
    set_storage_root(
        view, StorageRoot(root_tensor, root_tensor.proxy, view.proxy)
    )
