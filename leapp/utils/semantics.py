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

from .enums import Kind, AxisKind, ExpressionFrame
from .tensor_description import TensorSemantics


_COMPONENTS = {
    Kind.FRAME_POSE: ("x", "y", "z", "qx", "qy", "qz", "qw"),
    Kind.FRAME_POSITION: ("x", "y", "z"),
    Kind.FRAME_ORIENTATION: ("qx", "qy", "qz", "qw"),
    Kind.FRAME_TWIST: ("vx", "vy", "vz", "wx", "wy", "wz"),
    Kind.FRAME_LINEAR_VELOCITY: ("vx", "vy", "vz"),
    Kind.FRAME_ANGULAR_VELOCITY: ("wx", "wy", "wz"),
}


def _annotate(kind, name, ref, *, axes, is_setpoint, **spatial):
    if "reference" in spatial and spatial["reference"] is None:
        raise ValueError("A spatial reference frame is required")
    if "expressed_in" in spatial and spatial["expressed_in"] is None:
        raise ValueError("An expression frame is required")
    annotation = TensorSemantics(name=name, ref=ref, kind=kind, axes=axes,
                                 is_setpoint=is_setpoint, **spatial)
    components = _COMPONENTS.get(kind)
    if components is not None and annotation.axes is not None:
        for axis in annotation.axes:
            if axis is not None and axis.kind == AxisKind.COMPONENT and axis.names is not None:
                if len(axis.names) != len(components) or set(axis.names) != set(components):
                    raise ValueError(f"{kind.value} components must be a permutation of {components}")
    return annotation


def joint_position(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.JOINT_POSITION, name, ref, axes=axes, is_setpoint=is_setpoint)


def joint_velocity(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.JOINT_VELOCITY, name, ref, axes=axes, is_setpoint=is_setpoint)


def joint_effort(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.JOINT_EFFORT, name, ref, axes=axes, is_setpoint=is_setpoint)


def vector3d(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.VECTOR3D, name, ref, axes=axes, is_setpoint=is_setpoint)


def kp(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.KP, name, ref, axes=axes, is_setpoint=is_setpoint)


def kd(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.KD, name, ref, axes=axes, is_setpoint=is_setpoint)


def image(name, ref, *, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.IMAGE, name, ref, axes=axes, is_setpoint=is_setpoint)


def frame_pose(name, ref, *, reference, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_POSE, name, ref, axes=axes, is_setpoint=is_setpoint, reference=reference)


def frame_position(name, ref, *, reference, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_POSITION, name, ref, axes=axes, is_setpoint=is_setpoint, reference=reference)


def frame_orientation(name, ref, *, reference, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_ORIENTATION, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference)


def frame_twist(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_TWIST, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_linear_velocity(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_LINEAR_VELOCITY, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_angular_velocity(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_ANGULAR_VELOCITY, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_acceleration(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_ACCELERATION, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_linear_acceleration(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_LINEAR_ACCELERATION, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_angular_acceleration(name, ref, *, reference,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_ANGULAR_ACCELERATION, name, ref,
                     axes=axes, is_setpoint=is_setpoint, reference=reference, expressed_in=expressed_in)


def frame_wrench(name, ref, *,
                expressed_in=ExpressionFrame.SELF, axes=None, is_setpoint=False) -> TensorSemantics:
    return _annotate(Kind.FRAME_WRENCH, name, ref,
                     axes=axes, is_setpoint=is_setpoint, expressed_in=expressed_in)
