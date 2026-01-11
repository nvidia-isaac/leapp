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


def run_example(**inputs):
    from examples.wbc_plain import run_model, get_model
    model = get_model("models/isaac_velocity_flat_h1_v0.pt")
    return run_model(model, **inputs)

class TestWBCPlain(BaseExampleTest):
    """Unit tests for examples/wbc_plain.py"""

    def test_wbc_plain_execution(self):
        """Test that wbc_plain.py runs without errors and generates expected files."""

        # Expected output files based on the sample_wbc_graph directory
        expected_files = [
            'concatenate_and_run_model.onnx',
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

        inputs = {
            'joint_pos': torch.randn(19, device='cuda', dtype=torch.float32),
            'joint_vel': torch.randn(19, device='cuda', dtype=torch.float32),
            'velocity_commands': torch.randn(3, device='cuda', dtype=torch.float32),
            'lin_vel_I': torch.randn(3, device='cuda', dtype=torch.float32),
            'ang_vel_I': torch.randn(3, device='cuda', dtype=torch.float32),
            'q_IB': torch.randn(4, device='cuda', dtype=torch.float32),
            'previous_actions': torch.zeros(19, device='cuda', dtype=torch.float32),
        }
        exported_pipeline_inputs = {
            'concatenate_and_run_model/velocity_commands': inputs['velocity_commands'].clone(),
            'concatenate_and_run_model/joint_vel': inputs['joint_vel'].clone(),
            'process_joint_pos/joint_pos': inputs['joint_pos'].clone(),
            'process_odom/lin_vel_I': inputs['lin_vel_I'].clone(),
            'process_odom/ang_vel_I': inputs['ang_vel_I'].clone(),
            'process_odom/q_IB': inputs['q_IB'].clone(),
        }

        run_example(**inputs)
        outputs, _ = run_example(**inputs)

        exported_example = InferenceManager('sample_wbc_graph/sample_wbc_graph.yaml')

        exported_outputs = exported_example.run_policy(exported_pipeline_inputs)
        exported_action = exported_outputs['post_process_actions/actions']

        try:
            torch.cuda.synchronize()  # Wait for GPU operations to complete
            self.assertTrue(torch.allclose(outputs, exported_action, rtol=1e-3, atol=1e-6))
        except AssertionError as e:
            print("Outputs do not match:")
            print(f"Outputs: {outputs}")
            print(f"Exported action: {exported_action}")
            raise e

        # self.assertEqual(outputs.shape, (19,))
        # self.assertEqual(previous_actions.shape, (19,))


if __name__ == '__main__':
    unittest.main(verbosity=2)
