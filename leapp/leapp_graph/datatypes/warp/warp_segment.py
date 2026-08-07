##
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import collections.abc
from dataclasses import dataclass, field
from typing import Any, Literal

from torch.fx.proxy import Proxy

from leapp.utils.dtype import value_to_name_and_shape


@dataclass
class WarpTensorRef:
    # Segment-local canonical name used for APIC params / FX output labels.
    name: str
    # Live trace-time object, usually a wp.array, kept for capture/replay work.
    array: Any
    # FX proxy that represents this value in the owning TracedTensorNode graph.
    proxy: Proxy | None = None
    # Runtime array shape observed during tracing, used for validation/export metadata.
    shape: tuple | None = None
    # Runtime dtype observed during tracing, stored as text for lightweight metadata.
    dtype: str | None = None
    # Fixed scalar dimensions belonging to one Warp element, e.g. (3,) for vec3.
    component_shape: tuple = ()
    # Torch-facing scalar storage shape, including compound component dimensions.
    storage_shape: tuple | None = None
    # Torch-facing scalar dtype name used by FX/export/runtime metadata.
    storage_dtype: str | None = None
    # Device/host pointer when available; helps dedupe view-like wp.array objects.
    ptr: int | None = None

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
    ) -> WarpTensorRef:
        ptr = getattr(value, "ptr", None)
        try:
            ptr = int(ptr) if ptr else None
        except Exception:
            ptr = None

        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                shape = tuple(shape)
            except TypeError:
                pass

        dtype = getattr(value, "dtype", None)
        dtype_name = getattr(dtype, "__name__", None)
        storage_dtype, storage_shape = value_to_name_and_shape(value)
        storage_shape = tuple(storage_shape)
        component_shape = ()
        if isinstance(shape, tuple) and storage_shape[:len(shape)] == shape:
            component_shape = storage_shape[len(shape):]

        return cls(
            name=name,
            array=value,
            proxy=getattr(value, "proxy", None),
            shape=shape,
            dtype=dtype_name or (str(dtype) if dtype is not None else None),
            component_shape=component_shape,
            storage_shape=storage_shape,
            storage_dtype=storage_dtype,
            ptr=ptr,
        )


@dataclass
class WarpSegment:
    # Owning LEAPP node name. A segment should not span multiple node graphs.
    node_name: str
    # Discovery open-site anchor used to reject mismatched capture reentry.
    open_call_stack: Any | None = None
    # Discovery close-site anchor, kept for future boundary tracing.
    close_call_stack: Any | None = None
    # Warp call sequence observed during discovery.
    call_qualnames: tuple[str, ...] = ()
    # Lifecycle state; invalid segments fail closed instead of silently exporting.
    status: Literal["open", "closed", "invalid"] = "open"
    # Detector-recorded top-level Warp events/calls that make up the segment.
    events: list[Any] = field(default_factory=list)
    # Runtime APIC/FX inputs, keyed by segment-local canonical name.
    input_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # Runtime APIC/FX outputs that will get marker-derived proxies.
    output_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # FX proxy for the single segment marker node.
    marker_proxy: Proxy | None = None
    # Stable FX runner base name assigned when the segment is discovered.
    runner_name: str | None = None
    # FX marker name, e.g. ``warp_segment_0``.
    proxy_name: str | None = None
    # Live APIC graph object during trace/export; intentionally not serialized.
    apic_graph: Any | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def is_pending_capture(self) -> bool:
        return self.apic_graph is None

    @property
    def is_empty(self) -> bool:
        return not (
            self.events
            or self.input_refs
            or self.output_refs
        )

    def add_event(self, event: Any) -> None:
        self.events.append(event)

    def add_input_ref(
        self,
        value: Any,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("input", len(self.input_refs)),
        )
        existing = self._find_ref(ref, self.input_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref)

        self.input_refs[ref.name] = ref
        return ref

    def add_output_ref(
        self,
        value: Any,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("output", len(self.output_refs)),
        )
        existing = self._find_ref(ref, self.output_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref)

        self.output_refs[ref.name] = ref
        return ref

    def _coerce_ref(
        self,
        value: Any,
        *,
        name: str,
    ) -> WarpTensorRef:
        if isinstance(value, WarpTensorRef):
            ref = value
            ref.name = name
            return ref

        return WarpTensorRef.from_value(name, value)

    @staticmethod
    def _default_ref_name(prefix: str, index: int) -> str:
        return f"{prefix}_{index}"

    @staticmethod
    def _ref_key(ref: WarpTensorRef) -> tuple[int, int | None]:
        return (id(ref.array), ref.ptr)

    def _find_ref(
        self, ref: WarpTensorRef, existing_refs: collections.abc.Iterable
    ) -> WarpTensorRef | None:
        ref_key = self._ref_key(ref)
        for existing in existing_refs:
            if self._ref_key(existing) == ref_key:
                return existing
        return None

    @staticmethod
    def _merge_ref(
        existing: WarpTensorRef,
        incoming: WarpTensorRef,
    ) -> WarpTensorRef:
        if existing.proxy is None:
            existing.proxy = incoming.proxy
        return existing

    def invalidate(self) -> None:
        self.status = "invalid"
