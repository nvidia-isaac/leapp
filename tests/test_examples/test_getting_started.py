#!/usr/bin/env python3

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

import unittest
from .base_example_test import BaseExampleTest


class TestGettingStarted(BaseExampleTest):
    """Unit tests for examples/getting_started.py."""

    def test_getting_started_execution(self):
        """Test that getting_started.py runs and generates expected artifacts."""
        expected_files = [
            "obs_processor.pt",
            "policy.pt",
            "sample_robot_pipeline.png",
            "sample_robot_pipeline.yaml",
        ]

        self._run_and_verify_example(
            script_name="getting_started.py",
            output_dir_name="sample_robot_pipeline",
            expected_files=expected_files,
            test_description="Testing getting_started.py",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
