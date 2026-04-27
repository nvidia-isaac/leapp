#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Getting Started with LEAPP.

This example mirrors the getting-started guide and shows two important ideas:
1. traced-tensor-first node annotation with input_tensors()/output_tensors()
2. multiple input_tensors() calls contributing inputs to the same node

Pipeline:
  [obs_processor] -> [policy]

Run:
    python examples/getting_started.py
"""

import os
import torch
import leapp
from leapp import annotate


# ---- Small helper functions (plain Python, no nn.Module needed) ----
_POS_MEAN = torch.zeros(6)
_POS_STD = torch.ones(6) * 0.5
_VEL_SCALE = 4.0


def normalize_joints(pos: torch.Tensor, vel: torch.Tensor):
    pos_norm = (pos - _POS_MEAN) / (_POS_STD + 1e-6)
    vel_norm = vel / _VEL_SCALE
    return pos_norm, vel_norm


def capture_joint_observations(joint_pos: torch.Tensor, joint_vel: torch.Tensor):
    """First input_tensors() call for the obs_processor node."""
    return annotate.input_tensors("obs_processor", {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    })


def capture_context(orientation: torch.Tensor, cmd_vel: torch.Tensor):
    """Second input_tensors() call for the same obs_processor node."""
    return annotate.input_tensors("obs_processor", {
        "orientation": orientation,
        "cmd_vel": cmd_vel,
    })


def project_gravity(quat: torch.Tensor) -> torch.Tensor:
    """Project world gravity into body frame from a (w,x,y,z) quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return torch.stack([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


_ACTION_SCALE = 0.25
_JOINT_LIMIT = 1.0


def scale_and_clip(raw: torch.Tensor) -> torch.Tensor:
    return torch.clamp(raw * _ACTION_SCALE, min=-_JOINT_LIMIT, max=_JOINT_LIMIT)


# ---- Tiny hand-written policy weights (simulating checkpoint constants) ----
torch.manual_seed(42)
_W, _b = torch.randn(18, 6) * 0.05, torch.zeros(6)


def main():
    # Example robot state.
    joint_pos = torch.randn(6)
    joint_vel = torch.randn(6)
    orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cmd_vel = torch.tensor([0.5, 0.0, 0.1])

    leapp.start(name="sample_pipeline")

    # Node 1: observation preprocessing.
    # These two calls contribute inputs to the same traced node.
    pos, vel = capture_joint_observations(joint_pos, joint_vel)
    quat, cmd = capture_context(orientation, cmd_vel)

    pos_norm, vel_norm = normalize_joints(pos, vel)
    gravity_vec = project_gravity(quat)  # helper function calls are traced too
    obs_features = torch.cat([pos_norm, vel_norm, gravity_vec, cmd])
    annotate.output_tensors(
        "obs_processor",
        {"obs_features": obs_features},
        export_with="jit",
    )

    # Node 2: tiny policy + post-processing.
    feat = annotate.input_tensors("policy", {
        "obs_features": obs_features,
    })
    raw_action = feat @ _W + _b
    joint_targets = scale_and_clip(raw_action)

    annotate.output_tensors(
        "policy",
        {"joint_targets": joint_targets},
        export_with="jit",
    )

    leapp.stop()
    leapp.compile_graph(visualize=True)

    out_dir = "sample_pipeline"
    print(f"\nExported to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
