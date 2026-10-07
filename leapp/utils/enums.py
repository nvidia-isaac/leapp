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


class Kind(str, Enum):
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
    FRAME_WRENCH = "frame/wrench"
    VECTOR3D = "vector3d"
    KP = "kp"
    KD = "kd"
    IMAGE = "image"


class AxisKind(str, Enum):
    ROBOT = "robot"
    ELEMENT = "element"
    COMPONENT = "component"
    TIME = "time"


class ExpressionFrame(Enum):
    SELF = "self"
    REFERENCE = "reference"
