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

import torch
import unittest
from leapp.inference_manager import InferenceManager
from .base_example_test import BaseExampleTest

def run_example():
    from examples.compass_native_python import CompassNavigationModel, create_test_data
    test_image, test_odom, test_transform, goal_pose, route_transform = create_test_data()
    compass_native_python = CompassNavigationModel(mobility_model_path="examples/models/digit_mobility.jit", device = 'cuda')
    inputs = {
        'compass_goal_checker/goal': goal_pose.clone(),
        'compass_image_processor/raw_image': test_image.clone(),
        'compass_odometry_processor/odom_msg': test_odom.clone(),
        'compass_odometry_processor/transform': test_transform.clone(),
        'compass_route_calculator/goal_pose': goal_pose.clone(),
        'compass_route_calculator/transform': route_transform.clone(),
    }

    final_commands = compass_native_python.run_navigation_pipeline(test_image, test_odom, goal_pose, test_transform, route_transform)
    return final_commands, inputs

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



        outputs, inputs = run_example()

        # Load and run the exported pipeline
        exported_example = InferenceManager('sample_compass_navigation_pipeline/sample_compass_navigation_pipeline.yaml')
        # Initialize feedback state values to match native model's initial state
        exported_example.set_input_value(
            "compass_odometry_processor", "prev_transform",
            torch.tensor([0.05, 0.02, 0.01, 0.99, 0.0, 0.01, 0.0, 999.0, 0.0],
                         dtype=torch.float32, device='cuda'))
        exported_example.set_input_value(
            "compass_odometry_processor", "ego_speed",
            torch.zeros(1, dtype=torch.float32, device='cuda'))
        exported_example.set_input_value(
            "compass_odometry_processor", "position_2d",
            torch.zeros(3, dtype=torch.float32, device='cuda'))
        exported_outputs = exported_example.run_policy(inputs)
        exported_action = exported_outputs['post_process_commands/cmd']

        try:
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            self.assertTrue(torch.allclose(outputs, exported_action, rtol=1e-6, atol=1e-6))
        except AssertionError as e:
            print("Outputs do not match:")
            print(f"Outputs: {outputs}")
            print(f"Exported action: {exported_action}")
            raise e


if __name__ == '__main__':
    unittest.main(verbosity=2)
