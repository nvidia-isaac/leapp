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
    from examples.wbc_obj import WBC
    wbc = WBC()
    return wbc.run_model(**inputs)


class TestWBCObj(BaseExampleTest):
    """Unit tests for examples/wbc_obj.py"""

    def test_wbc_obj_execution(self):
        """Test that wbc_obj.py runs without errors and generates expected files."""

        # Expected output files - these should be similar to wbc_plain but for sample_wbc_obj
        expected_files = [
            'concatenate_and_run_model.pt',
            'process_odom.pt',
            'process_joint_pos.pt',
            'post_process_actions.pt',
            'sample_wbc_obj.png',
            'sample_wbc_obj.yaml'
        ]

        # Run the test
        self._run_and_verify_example(
            script_name='wbc_obj.py',
            output_dir_name='sample_wbc_obj',
            expected_files=expected_files,
            test_description='Testing wbc_obj.py'
        )

        # Test that the exported pipeline produces the same outputs as the original
        inputs = {
            'joint_pos': torch.randn(19, device='cuda', dtype=torch.float32),
            'joint_vel': torch.randn(19, device='cuda', dtype=torch.float32),
            'velocity_commands': torch.randn(3, device='cuda', dtype=torch.float32),
            'lin_vel_I': torch.randn(3, device='cuda', dtype=torch.float32),
            'ang_vel_I': torch.randn(3, device='cuda', dtype=torch.float32),
            'q_IB': torch.randn(4, device='cuda', dtype=torch.float32),
        }
        exported_pipeline_inputs = {
            'concatenate_and_run_model/velocity_commands': inputs['velocity_commands'].clone(),
            'concatenate_and_run_model/joint_vel': inputs['joint_vel'].clone(),
            'process_joint_pos/joint_pos': inputs['joint_pos'].clone(),
            'process_odom/lin_vel_I': inputs['lin_vel_I'].clone(),
            'process_odom/ang_vel_I': inputs['ang_vel_I'].clone(),
            'process_odom/q_IB': inputs['q_IB'].clone(),
        }


        outputs = run_example(**inputs)

        # Load and run the exported pipeline
        exported_example = InferenceManager('sample_wbc_obj/sample_wbc_obj.yaml')
        exported_outputs = exported_example.run_policy(exported_pipeline_inputs)
        exported_action = exported_outputs['post_process_actions/actions']

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
