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


class TestCompassNativePython(BaseExampleTest):
    """Unit tests for examples/compass_native_python.py"""

    def test_compass_native_python_execution(self):
        """Test that compass_native_python.py runs without errors and generates expected files."""

        # Expected output files based on the sample_compass_navigation_pipeline directory
        expected_files = [
            'compass_goal_checker.pt',
            'compass_image_processor.pt',
            'compass_odometry_processor.pt',
            'compass_route_calculator.pt',
            'post_process_commands.pt',
            'process_and_run_inference.pt',
            'sample_compass_navigation_pipeline.png',
            'sample_compass_navigation_pipeline.yaml'
        ]

        # Run the test
        self._run_and_verify_example(
            script_name='compass_native_python.py',
            output_dir_name='sample_compass_navigation_pipeline',
            expected_files=expected_files,
            test_description='Testing compass_native_python.py'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
