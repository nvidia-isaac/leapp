#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Verify the tensor semantics example's exported metadata and replay."""

import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from leapp import InferenceManager


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "tensor_semantics.py"


def test_tensor_semantics_example():
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], cwd=workdir,
            capture_output=True, text=True, check=True, timeout=120,
        )
        assert "Nominal row offsets (ms): [0, 20, 40, 60]" in result.stdout
        yaml_path = Path(workdir) / "tensor_semantics_example" / "tensor_semantics_example.yaml"
        with yaml_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        model = config["models"]["joint_controller"]
        inputs = {item["name"]: item for item in model["inputs"]}
        for name, kind in (
            ("joint_pos", "state/joint/position"),
            ("joint_vel", "state/joint/velocity"),
            ("joint_command", "command/joint/position"),
        ):
            assert inputs[name]["kind"] == kind
            assert inputs[name]["element_names"] == [["hip", "knee", "ankle"]]

        output = model["outputs"][0]
        assert output["shape"] == [4, 3]
        assert output["kind"] == "target/joint/position"
        assert output["element_names"] == ["__temporal_axis__", ["hip", "knee", "ankle"]]
        assert output["temporal_period_ms"] == 20
        assert config["pipeline"]["configs"]["frequency"] == 50

        manager = InferenceManager(str(yaml_path))
        values = {
            "joint_pos": [0.2, -0.1, 0.4],
            "joint_vel": [0.1, -0.2, 0.3],
            "joint_command": [0.5, 0.2, -0.2],
        }
        replay_inputs = manager.get_mock_input()
        for name, data in values.items():
            key = f"joint_controller/{name}"
            replay_inputs[key] = replay_inputs[key].new_tensor(data)
        actual = manager.run_policy(replay_inputs)["joint_controller/actions"]
        position = replay_inputs["joint_controller/joint_pos"]
        velocity = replay_inputs["joint_controller/joint_vel"]
        command = replay_inputs["joint_controller/joint_command"]
        target = command - 0.05 * velocity
        expected = torch.stack([
            position + (target - position) * fraction
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ])
        torch.testing.assert_close(actual, expected)
