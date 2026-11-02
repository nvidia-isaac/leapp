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

from enum import Enum


class InputKindEnum(Enum):
    """Deprecated input kinds; use Kind or quantity helpers."""

    JOINT_POSITION = "state/joint/position"
    JOINT_VELOCITY = "state/joint/velocity"
    JOINT_EFFORT = "state/joint/effort"
    BODY_POSE = "state/body/pose"
    BODY_VEL = "state/body/velocity"
    BODY_ACC = "state/body/acceleration"
    BODY_LINEAR_ACCELERATION = "state/body/linear_acceleration"
    BODY_LINEAR_VELOCITY = "state/body/linear_velocity"
    BODY_ANGULAR_ACCELERATION = "state/body/angular_acceleration"
    BODY_ANGULAR_VELOCITY = "state/body/angular_velocity"
    BODY_ROTATION = "state/body/rotation"
    BODY_POSITION = "state/body/position"
    WRENCH = "state/wrench"
    VECTOR3D = "state/vector3d"
    COMMAND_JOINT_POSITION = "command/joint/position"
    COMMAND_JOINT_VELOCITY = "command/joint/velocity"
    COMMAND_JOINT_TORQUES = "command/joint/torques"
    COMMAND_BODY_ROTATION = "command/body/rotation"
    COMMAND_BODY_VELOCITY = "command/body/velocity"
    COMMAND_POSE = "command/body/pose"


class OutputKindEnum(Enum):
    """Deprecated output kinds; use Kind or quantity helpers."""

    KP = "kp"
    KD = "kd"
    JOINT_POSITION = "target/joint/position"
    JOINT_VELOCITY = "target/joint/velocity"
    JOINT_TORQUES = "target/joint/torques"
    JOINT_EFFORT = "target/joint/effort"
    BODY_POSITION = "target/body/position"
    BODY_LINEAR_ACCELERATION = "target/body/linear_acceleration"
    BODY_ORIENTATION = "target/body/orientation"
    BODY_LINEAR_VELOCITY = "target/body/linear_velocity"
    BODY_ANGULAR_ACCELERATION = "target/body/angular_acceleration"


# Experimental quantity and axis vocabulary; existing enums remain unchanged.
class Kind(str, Enum):
    """Experimental quantities independent of input/output direction."""

    JOINT_POSITION = "joint/position"
    JOINT_VELOCITY = "joint/velocity"
    JOINT_EFFORT = "joint/effort"
    FRAME_POSE = "frame/pose"
    FRAME_POSITION = "frame/position"
    FRAME_ORIENTATION = "frame/orientation"
    FRAME_TWIST = "frame/twist"
    FRAME_LINEAR_VELOCITY = "frame/linear_velocity"
    FRAME_ANGULAR_VELOCITY = "frame/angular_velocity"
    FRAME_ACCELERATION = "frame/acceleration"
    FRAME_LINEAR_ACCELERATION = "frame/linear_acceleration"
    FRAME_ANGULAR_ACCELERATION = "frame/angular_acceleration"
    FORCE_APPLICATION_POSITION = "frame/force_application_position"
    FRAME_WRENCH = "frame/wrench"
    VECTOR3D = "vector3d"
    KP = "kp"
    KD = "kd"
    IMAGE = "image"


class AxisKind(str, Enum):
    """Experimental meanings for individual tensor dimensions."""

    SYSTEM = "system"
    ELEMENT = "element"
    COMPONENT = "component"
    TIME = "time"


class ExpressionFrame(Enum):
    """Experimental frame selectors, distinct from literal frame names."""

    SELF = "self"
    REFERENCE = "reference"
