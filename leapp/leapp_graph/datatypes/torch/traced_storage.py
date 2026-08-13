"""Experimental versioned storage provenance for ``TracedTensor``.

This prototype intentionally supports only contiguous, same-dtype views that
cover the complete canonical tensor. It proves the storage-owned proxy model
without taking on partial-view writeback yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import torch
from torch.fx.proxy import Proxy


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    storage_id = getattr(storage, "_cdata", None)
    if storage_id is None:
        storage_id = (int(storage.data_ptr()), int(storage.nbytes()))
    device = tensor.device
    return (device.type, device.index, storage_id)


@dataclass(frozen=True)
class TorchViewSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    covers_canonical: bool

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        canonical_numel: int,
        canonical_storage_offset: int,
    ) -> "TorchViewSpec":
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=tensor.dtype,
            covers_canonical=(
                tensor.is_contiguous()
                and tensor.numel() == canonical_numel
                and int(tensor.storage_offset()) == canonical_storage_offset
            ),
        )


class TracedStorage:
    """One versioned FX value for a live Torch storage allocation."""

    def __init__(self, tensor: torch.Tensor, context: Any, proxy: Proxy) -> None:
        self.context = context
        self.tracer = getattr(context, "tracer", None)
        self.storage_key = _storage_key(tensor)
        self.canonical_shape = tuple(tensor.shape)
        self.canonical_stride = tuple(tensor.stride())
        self.canonical_storage_offset = int(tensor.storage_offset())
        self.canonical_dtype = tensor.dtype
        self.canonical_numel = tensor.numel()
        self.supports_complete_views = (
            tensor.is_contiguous()
            and self.canonical_storage_offset == 0
            and tensor.untyped_storage().nbytes()
            == tensor.numel() * tensor.element_size()
        )
        self.current_proxy = proxy
        self.version = 0
        self._projection_cache: dict[tuple[Any, ...], Proxy] = {}

    def view_spec(self, tensor: torch.Tensor) -> TorchViewSpec:
        return TorchViewSpec.from_tensor(
            tensor,
            canonical_numel=self.canonical_numel,
            canonical_storage_offset=self.canonical_storage_offset,
        )

    def can_share_complete_view(self, tensor: torch.Tensor) -> bool:
        if not self.supports_complete_views:
            return False
        if _storage_key(tensor) != self.storage_key:
            return False
        if tensor.dtype != self.canonical_dtype:
            return False
        return self.view_spec(tensor).covers_canonical

    def _cache_key(self, spec: TorchViewSpec) -> tuple[Any, ...]:
        return (
            self.version,
            spec.shape,
            spec.stride,
            spec.storage_offset,
            spec.dtype,
        )

    def remember_projection(self, spec: TorchViewSpec, proxy: Proxy) -> None:
        self._projection_cache[self._cache_key(spec)] = proxy

    def effective_proxy(self, spec: TorchViewSpec, fallback: Proxy) -> Proxy:
        if not spec.covers_canonical:
            return fallback
        if spec.shape == self.canonical_shape and spec.stride == self.canonical_stride:
            return self.current_proxy

        key = self._cache_key(spec)
        cached = self._projection_cache.get(key)
        if cached is not None:
            return cached

        projected = self.context.tracer.create_proxy(
            "call_method",
            "reshape",
            (self.current_proxy, spec.shape),
            {},
        )
        self._projection_cache[key] = projected
        return projected

    def commit_mutation(self, spec: TorchViewSpec, updated_proxy: Proxy) -> None:
        if not spec.covers_canonical:
            raise RuntimeError(
                "Experimental TracedStorage cannot propagate mutation through "
                "a partial or non-contiguous Torch view."
            )

        if spec.shape == self.canonical_shape and spec.stride == self.canonical_stride:
            canonical_proxy = updated_proxy
        else:
            canonical_proxy = self.context.tracer.create_proxy(
                "call_method",
                "reshape",
                (updated_proxy, self.canonical_shape),
                {},
            )

        self.current_proxy = canonical_proxy
        self.version += 1
        self._projection_cache.clear()
        self._projection_cache[self._cache_key(spec)] = updated_proxy


__all__ = ["TorchViewSpec", "TracedStorage"]
