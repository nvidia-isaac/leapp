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
from .base_example_test import BaseExampleTest


class TestFeedbackExample(BaseExampleTest):
    """Unit tests for examples/feedback_example.py."""

    def test_feedback_example_execution(self):
        """Test that feedback_example.py runs and generates expected artifacts."""
        expected_files = [
            "policy_step.pt",
            "feedback_update.pt",
            "sample_feedback_graph.svg",
            "sample_feedback_graph.png",
            "sample_feedback_graph.yaml",
            "sample_feedback_graph_initial_values.safetensors",
        ]

        self._run_and_verify_example(
            script_name="feedback_example.py",
            output_dir_name="sample_feedback_graph",
            expected_files=expected_files,
            test_description="Testing feedback_example.py",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
