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
from leapp import annotate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def get_robot_params():
    joint_names = [
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'torso_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_ankle_joint',
        'right_ankle_joint',
        'left_elbow_joint',
        'right_elbow_joint',
    ]

    observations = {
        'base_lin_vel': (3,),
        'base_ang_vel': (3,),
        'projected_gravity': (3,),
        'velocity_commands': (3,),
        'joint_pos': (19,),
        'joint_vel': (19,),
        'actions': (19,),
    }

    actions = {
        'joint_pos': (19,),
    }

    action_scaling = torch.tensor(0.5)

    default_pos = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return joint_names, observations, actions, action_scaling, default_pos


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


@annotate.method(export_with="jit")
def post_process_actions(actions):
    _, _, _, action_scaling, default_pos = get_robot_params()
    action_scaling = action_scaling.to(actions.dtype).to(actions.device)
    default_pos = default_pos.to(actions.dtype).to(actions.device)
    actions = actions.squeeze()
    actions = actions * action_scaling + default_pos
    return actions


@annotate.method(export_with="jit")
def process_odom(lin_vel_I: torch.Tensor, ang_vel_I: torch.Tensor, q_IB: torch.Tensor):
    R_IB = quat_to_rot_matrix(q_IB)
    R_BI = R_IB.transpose(0, 1)
    lin_vel_b = torch.matmul(R_BI, lin_vel_I)
    ang_vel_b = torch.matmul(R_BI, ang_vel_I)
    gravity_I = torch.tensor([0.0, 0.0, -9.8], dtype=lin_vel_I.dtype,
                             device=lin_vel_I.device)
    projected_gravity_b = torch.matmul(R_BI, gravity_I)

    return lin_vel_b, ang_vel_b, projected_gravity_b


def run_model(model, joint_pos, joint_vel, velocity_commands, lin_vel_I, ang_vel_I, q_IB, previous_actions):
    _, _, _, _, default_pos = get_robot_params()
    default_pos = default_pos.to(joint_pos.dtype).to(joint_pos.device)

    with torch.no_grad():
        # process odom
        lin_vel_b, ang_vel_b, gravity_b = process_odom(
            lin_vel_I, ang_vel_I, q_IB)

        with annotate.block("process_joint_pos", inputs=["joint_pos"], outputs=["processed_joint_pos"],
                            environment_constants=['default_pos'], export_with="jit"):
            processed_joint_pos = joint_pos - \
                default_pos.to(joint_pos.dtype).to(joint_pos.device)

        with annotate.block("concatenate_and_run_model",
                            inputs=["lin_vel_b", "ang_vel_b",
                                    "gravity_b", "velocity_commands",
                                    "processed_joint_pos", "joint_vel", "previous_actions"],
                            outputs=["actions"],
                            environment_constants=['model'],
                            export_with="onnx-torchscript", backend_params={"prescript": True}):
            concatenated_tensor = torch.cat([lin_vel_b, ang_vel_b,
                                            gravity_b, velocity_commands,
                                            processed_joint_pos, joint_vel, previous_actions])
            transformed = concatenated_tensor.view(
                1, -1).float().to(dtype=torch.float32)
            if len(transformed.shape) == 1:
                transformed = transformed.unsqueeze(0)
            actions = model(transformed).detach().view(-1)

        post_processed_actions = post_process_actions(actions)

    return post_processed_actions, actions


def get_model(model_path):
    import os
    model_path = os.path.join(os.path.dirname(
        __file__), "models", "isaac_velocity_flat_h1_v0.pt")
    model = torch.jit.load(model_path, map_location=DEVICE)
    return model.eval()


def main():
    mock_observation_data = {
        "joint_pos": torch.randn(19, device=DEVICE, dtype=DTYPE),
        "joint_vel": torch.randn(19, device=DEVICE, dtype=DTYPE),
        "velocity_commands": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "lin_vel_I": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "ang_vel_I": torch.randn(3, device=DEVICE, dtype=DTYPE),
        "q_IB": torch.randn(4, device=DEVICE, dtype=DTYPE),
        "previous_actions": torch.zeros(19, device=DEVICE, dtype=DTYPE)}

    # get model
    model = get_model("models/isaac_velocity_flat_h1_v0.pt")
    # run model with data
    for i in range(5):
        final_actions, raw_actions = run_model(model, **mock_observation_data)
        print(final_actions)
        mock_observation_data["previous_actions"] = raw_actions
    # show results
    print("finished running model")


if __name__ == "__main__":
    annotate.start("sample_wbc_graph", verbose=True)
    main()
    annotate.stop()
    annotate.compile_graph()
