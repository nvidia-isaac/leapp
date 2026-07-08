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


@dataclass
class WarpTensorRef:
    # Segment-local canonical name used for APIC params / FX output labels.
    name: str
    # Live trace-time object, usually a wp.array, kept for capture/replay work.
    array: Any
    # FX proxy that represents this value in the owning TracedTensorNode graph.
    proxy: Proxy | None = None
    # Owning LEAPP trace context, normally the TracedTensorNode instance.
    context: Any | None = None
    # Detector path showing where the value was found, e.g. args[0] or kwargs['out'].
    path: str | None = None
    # Runtime array shape observed during tracing, used for validation/export metadata.
    shape: tuple | None = None
    # Runtime dtype observed during tracing, stored as text for lightweight metadata.
    dtype: str | None = None
    # Runtime device observed during tracing, stored as text for lightweight metadata.
    device: str | None = None
    # Device/host pointer when available; helps dedupe view-like wp.array objects.
    ptr: int | None = None
    # True when this ref is a runtime input to the Warp segment.
    is_input: bool = False
    # True when this ref is a runtime output of the segment.
    is_output: bool = False
    # Index of the detector event that first produced/wrote this ref, when known.
    produced_by_event_index: int | None = None
    # Extra detector/export annotations that do not deserve first-class fields yet.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        path: str | None = None,
        is_input: bool = False,
        is_output: bool = False,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
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
        device = getattr(value, "device", None)

        return cls(
            name=name,
            array=value,
            proxy=getattr(value, "proxy", None),
            context=getattr(value, "context_obj", None),
            path=path,
            shape=shape,
            dtype=str(dtype) if dtype is not None else None,
            device=str(device) if device is not None else None,
            ptr=ptr,
            is_input=is_input,
            is_output=is_output,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata or {},
        )


@dataclass
class WarpSegment:
    # Owning LEAPP node name. A segment should not span multiple node graphs.
    node_name: str
    # Lifecycle state; invalid segments fail closed instead of silently exporting.
    status: Literal["open", "closed", "invalid"] = "open"
    # Detector-recorded top-level Warp events/calls that make up the segment.
    events: list[Any] = field(default_factory=list)
    # Runtime APIC/FX inputs, keyed by segment-local canonical name.
    input_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # Runtime APIC/FX outputs that will get marker-derived proxies.
    output_refs: dict[str, WarpTensorRef] = field(default_factory=dict)
    # Conservative possible outputs seen by the detector, not yet confirmed.
    output_candidates: list[WarpTensorRef] = field(default_factory=list)
    # FX proxy for the single segment marker node.
    marker_proxy: Proxy | None = None
    # Per-output FX proxies derived from marker_proxy.
    output_proxies: dict[str, Proxy] = field(default_factory=dict)
    # Extra segment annotations such as capture strategy or detector details.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Warp device used for capture/replay when known.
    device: str | None = None
    # FX marker name, e.g. ``warp_segment_0``.
    proxy_name: str | None = None
    # Live APIC graph object during trace/export; intentionally not serialized.
    apic_graph: Any | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def is_empty(self) -> bool:
        return not (
            self.events
            or self.input_refs
            or self.output_refs
            or self.output_candidates
        )

    def add_event(self, event: Any) -> None:
        self.events.append(event)

    def add_input_ref(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("input", len(self.input_refs)),
            path=path,
            is_input=True,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.input_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref, is_input=True)

        self.input_refs[ref.name] = ref
        return ref

    def add_output_ref(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name("output", len(self.output_refs)),
            path=path,
            is_output=True,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.output_refs.values())
        if existing is not None:
            return self._merge_ref(existing, ref, is_output=True)

        self.output_refs[ref.name] = ref
        return ref

    def add_output_candidate(
        self,
        value: Any,
        *,
        path: str | None = None,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        ref = self._coerce_ref(
            value,
            name=self._default_ref_name(
                "output_candidate", len(self.output_candidates)
            ),
            path=path,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )
        existing = self._find_ref(ref, self.output_candidates)
        if existing is not None:
            return self._merge_ref(existing, ref)

        self.output_candidates.append(ref)
        return ref

    def _coerce_ref(
        self,
        value: Any,
        *,
        name: str,
        path: str | None = None,
        is_input: bool = False,
        is_output: bool = False,
        produced_by_event_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WarpTensorRef:
        if isinstance(value, WarpTensorRef):
            ref = value
            ref.name = name
            if path is not None:
                ref.path = path
            if metadata:
                ref.metadata.update(metadata)
            ref.is_input = ref.is_input or is_input
            ref.is_output = ref.is_output or is_output
            if produced_by_event_index is not None:
                ref.produced_by_event_index = produced_by_event_index
            return ref

        return WarpTensorRef.from_value(
            name,
            value,
            path=path,
            is_input=is_input,
            is_output=is_output,
            produced_by_event_index=produced_by_event_index,
            metadata=metadata,
        )

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
        *,
        is_input: bool = False,
        is_output: bool = False,
    ) -> WarpTensorRef:
        existing.is_input = existing.is_input or is_input or incoming.is_input
        existing.is_output = existing.is_output or is_output or incoming.is_output
        if existing.path is None:
            existing.path = incoming.path
        if existing.proxy is None:
            existing.proxy = incoming.proxy
        if existing.context is None:
            existing.context = incoming.context
        if existing.produced_by_event_index is None:
            existing.produced_by_event_index = incoming.produced_by_event_index
        existing.metadata.update(incoming.metadata)
        return existing

    def invalidate(self) -> None:
        self.status = "invalid"
