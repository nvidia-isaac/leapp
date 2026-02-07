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

import torch
import sys
import os
from leapp import annotate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def quat_to_rot_matrix(quat:
                       torch.Tensor) -> torch.Tensor:
    dtype = quat.dtype
    device = quat.device
    q = quat.clone().to(dtype=dtype, device=device)
    nq = torch.dot(q, q)

    # Smooth transition without conditionals
    # Convert scalar boolean to float and ensure proper broadcasting
    # 1.0 when nq is near zero, 0.0 otherwise
    weight_scalar = (nq < 1e-10).to(dtype)

    # Identity matrix
    identity = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype, device=device
    )

    # Regular calculation
    q_scaled = q * torch.sqrt(torch.tensor(2.0, dtype=dtype, device=q.device) /
                              torch.clamp(nq, min=1e-10))
    q_outer = torch.outer(q_scaled, q_scaled)

    # Create rotation matrix using torch.stack (TorchScript-friendly)
    row0 = torch.stack([1.0 - q_outer[2, 2] - q_outer[3, 3],
                        q_outer[1, 2] - q_outer[3, 0],
                        q_outer[1, 3] + q_outer[2, 0]])

    row1 = torch.stack([q_outer[1, 2] + q_outer[3, 0],
                        1.0 - q_outer[1, 1] - q_outer[3, 3],
                        q_outer[2, 3] - q_outer[1, 0]])

    row2 = torch.stack([q_outer[1, 3] - q_outer[2, 0],
                        q_outer[2, 3] + q_outer[1, 0],
                        1.0 - q_outer[1, 1] - q_outer[2, 2]])

    regular = torch.stack([row0, row1, row2])

    # Use broadcasting to apply the weight to each element of the matrices
    return weight_scalar * identity + (1 - weight_scalar) * regular


class WBC:
    def __init__(self):
        self.action_scaling = torch.tensor(0.5)

        self.default_pos = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        model_path = os.path.join(os.path.dirname(
            __file__), "models", "isaac_velocity_flat_h1_v0.pt")
        self.model = torch.jit.load(model_path, map_location=DEVICE)
        self.previous_actions = torch.zeros(19, device=DEVICE, dtype=DTYPE)

    def quat_to_rot_matrix(self, quat:
                           torch.Tensor) -> torch.Tensor:
        dtype = quat.dtype
        device = quat.device
        q = quat.clone().to(dtype=dtype, device=device)
        nq = torch.dot(q, q)

        # Smooth transition without conditionals
        # Convert scalar boolean to float and ensure proper broadcasting
        # 1.0 when nq is near zero, 0.0 otherwise
        weight_scalar = (nq < 1e-10).to(dtype)

        # Identity matrix
        identity = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=dtype, device=device
        )

        # Regular calculation
        q_scaled = q * torch.sqrt(torch.tensor(2.0, dtype=dtype, device=q.device) /
                                  torch.clamp(nq, min=1e-10))
        q_outer = torch.outer(q_scaled, q_scaled)

        # Create rotation matrix using torch.stack (TorchScript-friendly)
        row0 = torch.stack([1.0 - q_outer[2, 2] - q_outer[3, 3],
                            q_outer[1, 2] - q_outer[3, 0],
                            q_outer[1, 3] + q_outer[2, 0]])

        row1 = torch.stack([q_outer[1, 2] + q_outer[3, 0],
                            1.0 - q_outer[1, 1] - q_outer[3, 3],
                            q_outer[2, 3] - q_outer[1, 0]])

        row2 = torch.stack([q_outer[1, 3] - q_outer[2, 0],
                            q_outer[2, 3] + q_outer[1, 0],
                            1.0 - q_outer[1, 1] - q_outer[2, 2]])

        regular = torch.stack([row0, row1, row2])

        # Use broadcasting to apply the weight to each element of the matrices
        return weight_scalar * identity + (1 - weight_scalar) * regular

    @annotate.method(export_with="jit")
    def post_process_actions(self, actions):
        action_scaling = self.action_scaling.to(
            actions.dtype).to(actions.device)
        default_pos = self.default_pos.to(
            actions.dtype).to(actions.device)
        actions = actions.squeeze()
        actions = actions * action_scaling + default_pos
        return actions

    @annotate.method(export_with="jit")
    def process_odom(self, lin_vel_I: torch.Tensor, ang_vel_I: torch.Tensor, q_IB: torch.Tensor):
        R_IB = self.quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose(0, 1)
        lin_vel_b = torch.matmul(R_BI, lin_vel_I)
        ang_vel_b = torch.matmul(R_BI, ang_vel_I)
        downward_tensor = torch.tensor([0.0, 0.0, -1.0],
                                       dtype=R_BI.dtype, device=R_BI.device)
        projected_gravity_b = torch.matmul(R_BI, downward_tensor)

        return lin_vel_b, ang_vel_b, projected_gravity_b

    @annotate.method(export_with="jit")
    def process_joint_pos(self, joint_pos):
        return joint_pos - self.default_pos.to(joint_pos.dtype).to(joint_pos.device)

    def run_model(self, joint_pos, joint_vel, velocity_commands, lin_vel_I, ang_vel_I, q_IB):
        lin_vel_b, ang_vel_b, gravity_b = self.process_odom(
            lin_vel_I, ang_vel_I, q_IB)
        processed_joint_pos = self.process_joint_pos(joint_pos)

        with annotate.block("concatenate_and_run_model",
                            inputs=["lin_vel_b", "ang_vel_b",
                                    "gravity_b", "velocity_commands",
                                    "processed_joint_pos", "joint_vel", "self.previous_actions"],
                            outputs=["actions"],
                            environment_constants=['self.model'],
                            register_buffers=['self.previous_actions'],
                            export_with="jit"):
            concatenated_tensor = torch.cat([lin_vel_b, ang_vel_b,
                                            gravity_b, velocity_commands,
                                            processed_joint_pos, joint_vel, self.previous_actions])
            transformed = concatenated_tensor.view(
                1, -1).float().to(dtype=torch.float32)
            if len(transformed.shape) == 1:
                transformed = transformed.unsqueeze(0)
            actions = self.model(transformed).detach().view(-1)
            self.previous_actions = actions

        actions = self.post_process_actions(actions)
        return actions


def main():
    wbc = WBC()
    mock_observation_data = {
        "joint_pos": torch.randn(19, device=DEVICE, dtype=DTYPE),
        "joint_vel": torch.randn(19, device=DEVICE, dtype=DTYPE),
        "velocity_commands": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "lin_vel_I": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "ang_vel_I": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "q_IB": torch.randn(4, device=DEVICE, dtype=DTYPE)}

    actions = wbc.run_model(**mock_observation_data)
    print(actions)


if __name__ == "__main__":
    annotate.start(name="sample_wbc_obj", verbose=True)
    main()
    annotate.stop()
    annotate.compile_graph()
