#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Compare the exported FunctionPatch example against OpenCV on new frames."""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from leapp import InferenceManager


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "function_patch.py"


def test_function_patch_example():
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(
            [sys.executable, str(EXAMPLE)], cwd=workdir,
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "Patch restored: True" in result.stdout

        graph_dir = Path(workdir) / "function_patch_example"
        assert (graph_dir / "camera_preprocess.pt").is_file()
        manager = InferenceManager(str(graph_dir / "function_patch_example.yaml"))
        inputs = manager.get_mock_input()
        assert set(inputs) == {
            "camera_preprocess/left_bgr", "camera_preprocess/right_bgr",
        }
        rng = np.random.default_rng(17)
        frames = {
            name: rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
            for name in ("left_bgr", "right_bgr")
        }
        # Change each camera independently to catch frozen OpenCV results.
        for changed_camera in ("left_bgr", "right_bgr"):
            frames[changed_camera] = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
            expected_views = []
            for name, bgr in frames.items():
                key = f"camera_preprocess/{name}"
                inputs[key] = inputs[key].new_tensor(bgr)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                expected_views.append(
                    torch.nn.functional.adaptive_avg_pool2d(image, (2, 2)).flatten()
                )
            actual = manager.run_policy(inputs)["camera_preprocess/stereo_features"]
            expected = torch.cat(expected_views).to(actual.device)
            torch.testing.assert_close(actual, expected)
