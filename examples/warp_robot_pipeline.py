#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Warp preprocessing into a torch policy.

This example mirrors the Warp tensor-libraries guide. Conditioning stays in
Warp; the learned policy stays in torch. The annotated path runs twice:
discovery, then APIC capture.

Pipeline:
  [preprocess] -> [policy]

Run:
    python examples/warp_robot_pipeline.py
"""

import os

import torch
import warp as wp

import leapp
from leapp import annotate

VEL_SCALE = 4.0


@wp.kernel
def scale_and_clip(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
    scale: float,
    limit: float,
):
    i = wp.tid()
    dst[i] = wp.clamp(src[i] * scale, -limit, limit)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(12, 6)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


def preprocess(joint_pos, joint_vel) -> torch.Tensor:
    pos, vel = annotate.input_tensors("preprocess", {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    })

    pos_n = wp.empty_like(pos)
    vel_n = wp.empty_like(vel)
    wp.launch(scale_and_clip, dim=pos.size,
              inputs=[pos, pos_n, 1.0, 5.0], device=pos.device)
    wp.launch(scale_and_clip, dim=vel.size,
              inputs=[vel, vel_n, 1.0 / VEL_SCALE, 1.0], device=vel.device)

    obs = torch.cat([wp.to_torch(pos_n), wp.to_torch(vel_n)])
    annotate.output_tensors("preprocess", {"obs": obs}, export_with="onnx")
    return obs


def run_policy(policy: Policy, obs: torch.Tensor) -> torch.Tensor:
    traced_obs = annotate.input_tensors("policy", {"obs": obs})
    action = policy(traced_obs.unsqueeze(0)).squeeze(0)
    annotate.output_tensors("policy", {"action": action}, export_with="onnx")
    return action


class Sim:
    def joint_state(self):
        return (
            wp.zeros(6, dtype=wp.float32, device="cuda"),
            wp.zeros(6, dtype=wp.float32, device="cuda"),
        )


def main(sim):
    policy = Policy().eval().cuda()

    leapp.start(name="warp_robot_pipeline")
    for _ in range(2):  # discovery pass, then capture pass
        pos, vel = sim.joint_state()
        obs = preprocess(pos, vel)
        run_policy(policy, obs)
    leapp.stop()
    leapp.compile_graph()

    out_dir = "warp_robot_pipeline"
    print(f"\nExported to {out_dir}/")
    for name in sorted(os.listdir(out_dir)):
        print(f"  {name}")


if __name__ == "__main__":
    main(Sim())
