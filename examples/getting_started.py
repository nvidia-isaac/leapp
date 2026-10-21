#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Getting Started with LEAPP.

This example mirrors the getting-started guide and shows two important ideas:
1. traced-tensor-first node annotation with input_tensors()/output_tensors()
2. multiple input_tensors() calls contributing inputs to the same node

Pipeline:
  joint state + orientation + command -> [control] -> joint targets

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


def capture_joint_observations():
    """First input_tensors() call for the control node."""
    # Simulates reading live joint sensors each time this function is called.
    JOINT_POS = torch.randn(6)
    JOINT_VEL = torch.randn(6)
    return annotate.input_tensors("control", {
        "joint_pos": JOINT_POS,
        "joint_vel": JOINT_VEL,
    })


def capture_context():
    """Second input_tensors() call for the same control node."""
    # Simulates reading live orientation and command context each call.
    ORIENTATION = torch.tensor([1.0, 0.0, 0.0, 0.0])
    CMD_VEL = torch.tensor([0.5, 0.0, 0.1])
    return annotate.input_tensors("control", {
        "orientation": ORIENTATION,
        "cmd_vel": CMD_VEL,
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
    leapp.start(name="sample_pipeline")

    # One node captures preprocessing, policy, and action processing.
    # These two calls contribute inputs at different call sites.
    pos, vel = capture_joint_observations()
    quat, cmd = capture_context()

    pos_norm, vel_norm = normalize_joints(pos, vel)
    gravity_vec = project_gravity(quat)  # helper function calls are traced too
    obs_features = torch.cat([pos_norm, vel_norm, gravity_vec, cmd])
    raw_action = obs_features @ _W + _b
    joint_targets = scale_and_clip(raw_action)

    annotate.output_tensors(
        "control",
        {"joint_targets": joint_targets},
        export_with="onnx",
    )

    leapp.stop()
    leapp.compile_graph(visualize=True, validate=True)

    out_dir = "sample_pipeline"
    print(f"\nExported to {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
