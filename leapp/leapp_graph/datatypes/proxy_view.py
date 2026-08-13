#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Indirection between a traced carrier and the FX proxy representing its value."""

from typing import Optional

from torch.fx.proxy import Proxy


class ProxyView:
    """One mutable reference to the FX proxy that currently represents a value.

    Traced carriers hold a ``ProxyView`` rather than an ``fx.Proxy`` so that the
    proxy a carrier reports can change without every reader holding a stale
    reference. ``proxy`` is ``None`` when the value carries no graph provenance,
    which is the normal state outside an active trace.

    The reason for the indirection is shared memory. Torch, NumPy, and Warp can
    expose several logical arrays over one allocation, while FX proxies are
    immutable graph values. Nested views over a single mutable root are how a
    mutation through one alias becomes visible to the others, so this class is
    where parent links and forward/backward projection are added later. Keeping
    every carrier behind it now means those additions change this class and the
    sites that create aliases, not every proxy reader.

    Assigning ``proxy`` means this value was mutated in place. An out-of-place
    operation produces an independent value and therefore a new ``ProxyView``.
    """

    __slots__ = ("proxy",)

    def __init__(self, proxy: Optional[Proxy]):
        self.proxy = proxy
