#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class MergeCfgEnum(Enum):
    NO_MERGE = "no_merge"
    AUTOMATIC = "automatic"
    ALL = "all" # planned but not implemented yet
    SIGNATURE = "signature"  # planned but not implemented yet

class inputKindEnum(Enum):
    JOINT_POSITION= "state/joint/position"
    JOINT_VELOCITY= "state/joint/velocity"
    BODY_LINEAR_ACCELERATION= "state/body/linear_acceleration"
    BODY_LINEAR_VELOCITY= "state/body/linear_velocity"
    BODY_ANGULAR_ACCELERATION= "state/body/angular_acceleration"
    BODY_ANGULAR_VELOCITY= "state/body/angular_velocity"
    BODY_ROTATION= "state/body/rotation"
    COMMAND_JOINT_POSITION= "command/joint/position"
    COMMAND_JOINT_VELOCITY= "command/joint/velocity"
    COMMAND_JOINT_TORQUES= "command/joint/torques"
    COMMAND_BODY_ROTATION= "command/body/rotation"
    COMMAND_BODY_VELOCITY= "command/body/velocity"

class outputKindEnum(Enum):
    KP="kp"
    KD="kd"
    JOINT_POSITION="target/joint/position"
    JOINT_VELOCITY="target/joint/velocity"
    JOINT_TORQUES="target/joint/torques"
    BODY_POSITION="target/body/position"
    BODY_LINEAR_ACCELERATION="target/body/linear_acceleration"
    BODY_ORIENTATION="target/body/orientation"
    BODY_LINEAR_VELOCITY="target/body/linear_velocity"
    BODY_ANGULAR_ACCELERATION="target/body/angular_acceleration"
