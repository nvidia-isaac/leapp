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


class TestWBCPlain(BaseExampleTest):
    """Unit tests for examples/wbc_plain.py"""

    def test_wbc_plain_execution(self):
        """Test that wbc_plain.py runs without errors and generates expected files."""

        # Expected output files based on the sample_wbc_graph directory
        expected_files = [
            'concatenate_and_run_model.pt',
            'post_process_actions.pt',
            'process_joint_pos.pt',
            'process_odom.pt',
            'sample_wbc_graph.png',
            'sample_wbc_graph.yaml'
        ]

        # Run the test
        self._run_and_verify_example(
            script_name='wbc_plain.py',
            output_dir_name='sample_wbc_graph',
            expected_files=expected_files,
            test_description='Testing wbc_plain.py'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
