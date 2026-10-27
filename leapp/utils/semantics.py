#
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

"""Tensor semantics, axis descriptors, and experimental quantity helpers."""

import math
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from leapp.leapp_graph.datatypes import is_tracable_tensor_type, TRACABLE_BASE_TYPES
from .dtype import value_to_name_and_shape
from .enums import InputKindEnum, OutputKindEnum, Kind, AxisKind, ExpressionFrame
from .logging import _get_logger
from .yaml_utils import CompactYamlList


TEMPORAL_AXIS_SENTINEL = "__temporal_axis__"


@dataclass(frozen=True)
class TemporalAxis:
    """Deprecated temporal marker; use Axis(AxisKind.TIME, period_ms=...)."""

    period_ms: float

    def __post_init__(self):
        if self.period_ms <= 0:
            _get_logger().fatal(
                "TemporalAxis period_ms must be positive",
                error_type=ValueError,
            )
        warnings.warn(
            "TemporalAxis is deprecated; use Axis(AxisKind.TIME, period_ms=...) instead.",
            FutureWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class Axis:
    """Experimental meaning and labels for one tensor dimension."""

    kind: AxisKind | str | None = None
    names: list[str] | None = None
    period_ms: float | None = None

    def __post_init__(self):
        if self.kind is not None and (not isinstance(self.kind, str) or not self.kind):
            raise TypeError("Axis.kind must be an AxisKind, non-empty string, or None")
        if self.names is not None and (
            not isinstance(self.names, list)
            or any(not isinstance(name, str) or not name for name in self.names)
        ):
            raise TypeError("Axis.names must be a list of non-empty strings or None")
        if self.period_ms is not None:
            if self.kind != AxisKind.TIME:
                raise ValueError("period_ms is only valid for a time axis")
            if (
                isinstance(self.period_ms, bool)
                or not isinstance(self.period_ms, (int, float))
                or not math.isfinite(self.period_ms)
                or self.period_ms <= 0
            ):
                raise ValueError("Axis.period_ms must be finite and positive")
        elif self.kind == AxisKind.TIME:
            raise ValueError("A time axis requires period_ms")

    def to_dict(self):
        result = {}
        if self.kind is not None:
            result["kind"] = (
                self.kind.value if isinstance(self.kind, AxisKind) else self.kind
            )
        if self.names is not None:
            result["names"] = list(self.names)
        if self.period_ms is not None:
            result["period_ms"] = self.period_ms
        return result


@dataclass
class TensorSemantics:
    """User-facing semantic metadata for a tensor.

    ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` are
    experimental keyword-only fields. Existing kinds and element names retain
    their original behavior.

    Add new semantic fields here — they automatically propagate to
    TensorDescription, YAML serialization, and the unwrap/apply flow.

    Convention:
        - Internal fields (ref, name) are listed in _INTERNAL_FIELDS and excluded from serialization.
        - Public fields (kind, element_names, ...) are semantic data that gets serialized to YAML.
    """

    # Fields excluded from serialization (internal use only)
    _INTERNAL_FIELDS = frozenset({"ref", "name"})

    name: str = None
    ref: Any = None

    # Semantic fields
    kind: Optional[InputKindEnum | OutputKindEnum | Kind | str] = None
    element_names: Optional[List] = None  # deprecated
    temporal_period_ms: Optional[float] = field(default=None, init=False)
    extra: Optional[Dict[str, Any]] = None

    # Experimental fields are keyword-only to preserve the existing constructor.
    axes: list[Axis | None] | None = field(default=None, kw_only=True)
    is_setpoint: bool | None = field(default=None, kw_only=True)
    reference: str | None = field(default=None, kw_only=True)
    expressed_in: ExpressionFrame | str | None = field(default=None, kw_only=True)

    @property
    def _uses_experimental_semantics(self):
        return (
            isinstance(self.kind, Kind)
            or self.axes is not None
            or self.is_setpoint is not None
            or self.reference is not None
            or self.expressed_in is not None
        )

    def __post_init__(self):
        """error checking, auto conditioning"""
        existing_period_ms = self.temporal_period_ms
        self.temporal_period_ms = None
        if not is_tracable_tensor_type(
            self.ref
        ):  # this checks for both base types and traced types
            _get_logger().fatal(
                f"TensorSemantics 'ref' must be a traceable tensor type "
                f"accepted types are: {TRACABLE_BASE_TYPES}"
                f"got {type(self.ref).__name__}",
                error_type=TypeError,
            )
        if (
            isinstance(self.kind, (InputKindEnum, OutputKindEnum))
            or self.element_names is not None
        ):
            warnings.warn(
                "InputKindEnum, OutputKindEnum, and element_names are deprecated; "
                "use quantity helpers or Kind with Axis instead. "
                "Existing annotations retain their YAML representation.",
                FutureWarning,
                stacklevel=3,
            )
        if self.element_names is not None:
            self.element_names, detected_period_ms = self._normalize_element_names(
                self.element_names,
                allow_temporal_sentinel=existing_period_ms is not None,
            )
            self.temporal_period_ms = detected_period_ms or existing_period_ms

        if not self._uses_experimental_semantics:
            return
        if self.axes is not None and self.element_names is not None:
            raise ValueError("Use either element_names or axes, not both")
        if (
            self.kind is not None
            and not isinstance(self.kind, (InputKindEnum, OutputKindEnum))
            and (not isinstance(self.kind, str) or not self.kind)
        ):
            raise TypeError(
                "kind must be an existing kind enum, Kind, non-empty string, or None"
            )
        if self.is_setpoint is not None and not isinstance(self.is_setpoint, bool):
            raise TypeError("is_setpoint must be a boolean or None")
        if self.reference is not None and (
            not isinstance(self.reference, str) or not self.reference
        ):
            raise TypeError("reference must be a non-empty frame name or None (world)")
        if self.expressed_in is not None and not isinstance(
            self.expressed_in, ExpressionFrame
        ):
            if not isinstance(self.expressed_in, str) or not self.expressed_in:
                raise TypeError(
                    "expressed_in must be an ExpressionFrame or non-empty frame name"
                )
        if (
            self.kind in (Kind.FRAME_WRENCH, Kind.FORCE_APPLICATION_POSITION)
            and self.expressed_in is ExpressionFrame.REFERENCE
        ):
            raise ValueError("A wrench cannot use REFERENCE")
        if self.extra and set(self.extra) & {
            "axes",
            "is_setpoint",
            "reference",
            "expressed_in",
            "spatial",
        }:
            raise ValueError(
                "Use TensorSemantics fields rather than extra for axes and spatial metadata"
            )
        if self.axes is not None:
            if not isinstance(self.axes, list):
                raise TypeError("axes must be a list or None")
            _, shape = value_to_name_and_shape(self.ref)
            if len(self.axes) != len(shape):
                raise ValueError("axes must have one entry per tensor dimension")
            for size, axis in zip(shape, self.axes):
                if axis is None:
                    continue
                if not isinstance(axis, Axis):
                    raise TypeError("axes entries must be Axis or None")
                if axis.names is not None and len(axis.names) != size:
                    raise ValueError("Axis label count must match its tensor dimension")
            if sum(a is not None and a.kind == AxisKind.TIME for a in self.axes) > 1:
                raise ValueError("A tensor may contain at most one time axis")

    def to_dict(self) -> Dict[str, Any]:
        """Return all non-None semantic fields, excluding internal fields.

        Extra fields are flattened directly into the result dict rather than
        nested under an 'extra' key.
        """
        result = {}
        for f in fields(self):
            if f.name in self._INTERNAL_FIELDS or f.name in (
                "extra",
                "reference",
                "expressed_in",
            ):
                continue
            value = getattr(self, f.name)
            if f.name == "axes" and value is not None:
                value = [axis.to_dict() if axis is not None else None for axis in value]
            if value is not None:
                result[f.name] = value
        spatial = {}
        if (
            self._uses_experimental_semantics
            and isinstance(self.kind, str)
            and self.kind.startswith("frame/")
        ):
            if self.kind not in (Kind.FRAME_WRENCH, Kind.FORCE_APPLICATION_POSITION):
                spatial["reference"] = self.reference
            if self.kind not in (
                Kind.FRAME_POSE,
                Kind.FRAME_POSITION,
                Kind.FRAME_ORIENTATION,
            ):
                spatial["expressed_in"] = None
        # Preserve supplied metadata regardless of the quantity's conventions.
        if self.reference is not None:
            spatial["reference"] = self.reference
        if self.expressed_in is not None:
            spatial["expressed_in"] = (
                {"selector": self.expressed_in.value}
                if isinstance(self.expressed_in, ExpressionFrame)
                else {"frame": self.expressed_in}
            )
        if spatial:
            result["spatial"] = spatial
        if self.extra:
            result.update(self.extra)
        return result

    def update(self, values: Dict[str, Any]):
        """Set semantic fields from a dict.

        Known dataclass fields are set directly. Unknown keys are stored
        in the ``extra`` dict so they still appear in the serialized YAML.
        """
        valid_fields = {f.name for f in fields(self) if f.init} - self._INTERNAL_FIELDS
        # Existing callers used these names as arbitrary extra metadata. Only
        # interpret them as fields after opting into experimental semantics.
        if not self._uses_experimental_semantics and not isinstance(
            values.get("kind"), Kind
        ):
            valid_fields -= {"axes", "is_setpoint", "reference", "expressed_in"}
        for key, value in values.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                if self.extra is None:
                    self.extra = {}
                self.extra[key] = value
        self.__post_init__()

    @staticmethod
    def _normalize_element_names(element_names, allow_temporal_sentinel=False):
        """Normalize element_names and extract temporal axis metadata."""
        if isinstance(element_names, str):
            if element_names == TEMPORAL_AXIS_SENTINEL:
                if allow_temporal_sentinel:
                    return CompactYamlList([element_names]), None
                _get_logger().fatal(
                    f"{TEMPORAL_AXIS_SENTINEL!r} is reserved for TemporalAxis",
                    error_type=ValueError,
                )
            return CompactYamlList([CompactYamlList([element_names])]), None

        if not isinstance(element_names, list):
            return element_names, None

        temporal_period_ms = None
        normalized = CompactYamlList()
        has_axis_descriptors = False
        has_temporal_axis = False

        for item in element_names:
            if isinstance(item, TemporalAxis):
                if has_temporal_axis:
                    _get_logger().fatal(
                        "element_names can contain at most one TemporalAxis",
                        error_type=ValueError,
                    )
                has_axis_descriptors = True
                has_temporal_axis = True
                temporal_period_ms = item.period_ms
                normalized.append(TEMPORAL_AXIS_SENTINEL)
            elif isinstance(item, str):
                if item == TEMPORAL_AXIS_SENTINEL:
                    if not allow_temporal_sentinel:
                        _get_logger().fatal(
                            f"{TEMPORAL_AXIS_SENTINEL!r} is reserved for TemporalAxis",
                            error_type=ValueError,
                        )
                    has_axis_descriptors = True
                    normalized.append(TEMPORAL_AXIS_SENTINEL)
                else:
                    normalized.append(item)
            elif isinstance(item, list):
                if any(isinstance(child, TemporalAxis) for child in item):
                    _get_logger().fatal(
                        "TemporalAxis must be an axis item, not nested in a list",
                        error_type=ValueError,
                    )
                if any(child == TEMPORAL_AXIS_SENTINEL for child in item):
                    _get_logger().fatal(
                        f"{TEMPORAL_AXIS_SENTINEL!r} must be a bare axis item, not nested in a list",
                        error_type=ValueError,
                    )
                has_axis_descriptors = True
                normalized.append(CompactYamlList(item))
            elif item is None:
                has_axis_descriptors = True
                normalized.append(None)
            else:
                _get_logger().warning(
                    "element_names has mixed types, expected names or axis descriptors"
                )
                return element_names, temporal_period_ms

        if has_axis_descriptors:
            return normalized, temporal_period_ms

        return CompactYamlList([CompactYamlList(normalized)]), None


_COMPONENTS = {
    Kind.FORCE_APPLICATION_POSITION: ("x", "y", "z"),
    Kind.FRAME_POSE: ("x", "y", "z", "qx", "qy", "qz", "qw"),
    Kind.FRAME_POSITION: ("x", "y", "z"),
    Kind.FRAME_ORIENTATION: ("qx", "qy", "qz", "qw"),
    Kind.FRAME_TWIST: ("vx", "vy", "vz", "wx", "wy", "wz"),
    Kind.FRAME_LINEAR_VELOCITY: ("vx", "vy", "vz"),
    Kind.FRAME_ANGULAR_VELOCITY: ("wx", "wy", "wz"),
}


def _annotate(kind, name, ref, *, axes, is_setpoint, **spatial):
    annotation = TensorSemantics(
        name=name, ref=ref, kind=kind, axes=axes, is_setpoint=is_setpoint, **spatial
    )
    components = _COMPONENTS.get(kind)
    if components is not None and annotation.axes is not None:
        for axis in annotation.axes:
            if (
                axis is not None
                and axis.kind == AxisKind.COMPONENT
                and axis.names is not None
            ):
                if len(axis.names) != len(components) or set(axis.names) != set(
                    components
                ):
                    raise ValueError(
                        f"{kind.value} components must be a permutation of {components}"
                    )
    return annotation


def joint_position(
    name, ref, *, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(Kind.JOINT_POSITION, name, ref, axes=axes, is_setpoint=is_setpoint)


def joint_velocity(
    name, ref, *, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(Kind.JOINT_VELOCITY, name, ref, axes=axes, is_setpoint=is_setpoint)


def joint_effort(
    name, ref, *, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(Kind.JOINT_EFFORT, name, ref, axes=axes, is_setpoint=is_setpoint)


def vector3d(
    name, ref, *, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(Kind.VECTOR3D, name, ref, axes=axes, is_setpoint=is_setpoint)


def kp(name, ref, *, axes=None, is_setpoint: bool | None = None) -> TensorSemantics:
    return _annotate(Kind.KP, name, ref, axes=axes, is_setpoint=is_setpoint)


def kd(name, ref, *, axes=None, is_setpoint: bool | None = None) -> TensorSemantics:
    return _annotate(Kind.KD, name, ref, axes=axes, is_setpoint=is_setpoint)


def image(name, ref, *, axes=None, is_setpoint: bool | None = None) -> TensorSemantics:
    return _annotate(Kind.IMAGE, name, ref, axes=axes, is_setpoint=is_setpoint)


def frame_pose(
    name, ref, *, reference=None, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_POSE,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
    )


def frame_position(
    name, ref, *, reference=None, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_POSITION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
    )


def frame_orientation(
    name, ref, *, reference=None, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_ORIENTATION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
    )


def frame_twist(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_TWIST,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_linear_velocity(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_LINEAR_VELOCITY,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_angular_velocity(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_ANGULAR_VELOCITY,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_acceleration(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_ACCELERATION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_linear_acceleration(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_LINEAR_ACCELERATION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_angular_acceleration(
    name,
    ref,
    *,
    reference=None,
    expressed_in=None,
    axes=None,
    is_setpoint: bool | None = None,
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_ANGULAR_ACCELERATION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        reference=reference,
        expressed_in=expressed_in,
    )


def frame_wrench(
    name, ref, *, expressed_in=None, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    return _annotate(
        Kind.FRAME_WRENCH,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        expressed_in=expressed_in,
    )


def force_application_position(
    name, ref, *, expressed_in=None, axes=None, is_setpoint: bool | None = None
) -> TensorSemantics:
    """Position where a force acts, paired by element name with its wrench."""
    return _annotate(
        Kind.FORCE_APPLICATION_POSITION,
        name,
        ref,
        axes=axes,
        is_setpoint=is_setpoint,
        expressed_in=expressed_in,
    )
