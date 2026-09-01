#!/usr/bin/env python3

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

import unittest

import torch

from tests.warp_support import WarpTestCase

from .base_example_test import BaseExampleTest


class TestWarpRobotPipeline(WarpTestCase, BaseExampleTest):
    """Unit tests for examples/warp_robot_pipeline.py."""

    @classmethod
    def setUpClass(cls):
        BaseExampleTest.setUpClass()
        WarpTestCase.setUpClass()

    def test_warp_robot_pipeline_execution(self):
        """Test that warp_robot_pipeline.py runs and generates expected artifacts."""
        if not torch.cuda.is_available():
            self.skipTest("Warp example requires CUDA")

        expected_files = [
            "preprocess.onnx",
            "policy.onnx",
            "warp_robot_pipeline.png",
            "warp_robot_pipeline.yaml",
        ]

        self._run_and_verify_example(
            script_name="warp_robot_pipeline.py",
            output_dir_name="warp_robot_pipeline",
            expected_files=expected_files,
            test_description="Testing warp_robot_pipeline.py",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
