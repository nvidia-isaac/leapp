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
from leapp.utils.tensor_description import describe_io


class TestPackedTensorExpr(unittest.TestCase):
    """Tests for ParameterFormat.packed_tensor_expr property."""

    def test_simple_list_of_tensors(self):
        """TEST 1: Simple list of tensors"""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([4.0, 5.0, 6.0])
        data = [t1, t2]
        io_desc, param_fmt = describe_io("inputA", "inputA", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], ['inputA_0', 'inputA_1'])
        
        # Verify packing expression
        expected = "inputA = [inputA_0, inputA_1]"
        self.assertEqual(param_fmt.packed_tensor_expr, expected)

    def test_single_tensor_trivial(self):
        """TEST 2: Single tensor (trivial - no packing needed)"""
        data = torch.tensor([1.0, 2.0, 3.0])
        io_desc, param_fmt = describe_io("single", "single", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], ['single'])
        
        # Verify packing expression is empty (trivial case)
        self.assertEqual(param_fmt.packed_tensor_expr, "")




    def test_dict_of_tensors(self):
        """TEST 5: Dict of tensors"""
        data = {"pose": torch.tensor([1.0, 2.0]), "velocity": torch.tensor([3.0, 4.0])}
        io_desc, param_fmt = describe_io("state", "state", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], ['state_pose', 'state_velocity'])
        
        # Verify packing expression
        expected = 'state = {"pose": state_pose, "velocity": state_velocity}'
        self.assertEqual(param_fmt.packed_tensor_expr, expected)

    def test_nested_list_of_lists(self):
        """TEST 6: Nested list of lists"""
        data = [[torch.tensor([1.0]), torch.tensor([2.0])], 
                [torch.tensor([3.0]), torch.tensor([4.0])]]
        io_desc, param_fmt = describe_io("nested", "nested", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], 
                         ['nested_0_0', 'nested_0_1', 'nested_1_0', 'nested_1_1'])
        
        # Verify packing expression
        expected = "nested = [[nested_0_0, nested_0_1], [nested_1_0, nested_1_1]]"
        self.assertEqual(param_fmt.packed_tensor_expr, expected)

    def test_dict_containing_list(self):
        """TEST 7: Dict containing list"""
        data = {"positions": [torch.tensor([1.0]), torch.tensor([2.0])],
                "velocities": [torch.tensor([3.0]), torch.tensor([4.0])]}
        io_desc, param_fmt = describe_io("physics", "physics", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], 
                         ['physics_positions_0', 'physics_positions_1', 
                          'physics_velocities_0', 'physics_velocities_1'])
        
        # Verify packing expression
        expected = 'physics = {"positions": [physics_positions_0, physics_positions_1], "velocities": [physics_velocities_0, physics_velocities_1]}'
        self.assertEqual(param_fmt.packed_tensor_expr, expected)

    def test_list_of_dicts(self):
        """TEST 8: List of dicts"""
        data = [{"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
                {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}]
        io_desc, param_fmt = describe_io("points", "points", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], 
                         ['points_0_x', 'points_0_y', 'points_1_x', 'points_1_y'])
        
        # Verify packing expression
        expected = 'points = [{"x": points_0_x, "y": points_0_y}, {"x": points_1_x, "y": points_1_y}]'
        self.assertEqual(param_fmt.packed_tensor_expr, expected)

    def test_multiple_parameters_with_mixed_transformations(self):
        """TEST 9: Multiple parameters (with and without dots in raw names)"""
        data1 = [torch.tensor([1.0]), torch.tensor([2.0])]
        data2 = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
        io_desc1, param_fmt1 = describe_io("self_data", "self.data", data1)
        io_desc2, param_fmt2 = describe_io("config", "config", data2)
        combined_io = io_desc1 + io_desc2
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in combined_io], 
                         ['self_data_0', 'self_data_1', 'config_a', 'config_b'])
        
        # Verify packing expressions for each parameter (uses name_str)
        expected1 = "self_data = [self_data_0, self_data_1]"
        expected2 = 'config = {"a": config_a, "b": config_b}'
        self.assertEqual(param_fmt1.packed_tensor_expr, expected1)
        self.assertEqual(param_fmt2.packed_tensor_expr, expected2)

    def test_deeply_nested_structure(self):
        """TEST 10: Deeply nested structure"""
        data = {"level1": {"level2": [torch.tensor([1.0]), torch.tensor([2.0])]}}
        io_desc, param_fmt = describe_io("deep", "deep", data)
        
        # Verify io_descriptions
        self.assertEqual([d.name for d in io_desc], 
                         ['deep_level1_level2_0', 'deep_level1_level2_1'])
        
        # Verify packing expression
        expected = 'deep = {"level1": {"level2": [deep_level1_level2_0, deep_level1_level2_1]}}'
        self.assertEqual(param_fmt.packed_tensor_expr, expected)


class TestUnpackedTensorExpr(unittest.TestCase):
    """Tests for ParameterFormat.unpacked_tensor_expr property."""

    def test_simple_list_of_tensors(self):
        """TEST 1: Simple list of tensors"""
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([4.0, 5.0, 6.0])
        data = [t1, t2]
        io_desc, param_fmt = describe_io("inputA", "inputA", data)
        
        # Verify unpacking expression
        expected = "inputA_0 = inputA[0]\ninputA_1 = inputA[1]"
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)

    def test_single_tensor_trivial(self):
        """TEST 2: Single tensor (trivial - no unpacking needed)"""
        data = torch.tensor([1.0, 2.0, 3.0])
        io_desc, param_fmt = describe_io("single", "single", data)
        
        # Verify unpacking expression is empty (trivial case)
        self.assertEqual(param_fmt.unpacked_tensor_expr, "")



    def test_dict_of_tensors(self):
        """TEST 5: Dict of tensors"""
        data = {"pose": torch.tensor([1.0, 2.0]), "velocity": torch.tensor([3.0, 4.0])}
        io_desc, param_fmt = describe_io("state", "state", data)
        
        # Verify unpacking expression
        expected = 'state_pose = state["pose"]\nstate_velocity = state["velocity"]'
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)

    def test_nested_list_of_lists(self):
        """TEST 6: Nested list of lists"""
        data = [[torch.tensor([1.0]), torch.tensor([2.0])], 
                [torch.tensor([3.0]), torch.tensor([4.0])]]
        io_desc, param_fmt = describe_io("nested", "nested", data)
        
        # Verify unpacking expression
        expected = ("nested_0_0 = nested[0][0]\n"
                    "nested_0_1 = nested[0][1]\n"
                    "nested_1_0 = nested[1][0]\n"
                    "nested_1_1 = nested[1][1]")
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)

    def test_dict_containing_list(self):
        """TEST 7: Dict containing list"""
        data = {"positions": [torch.tensor([1.0]), torch.tensor([2.0])],
                "velocities": [torch.tensor([3.0]), torch.tensor([4.0])]}
        io_desc, param_fmt = describe_io("physics", "physics", data)
        
        # Verify unpacking expression
        expected = ('physics_positions_0 = physics["positions"][0]\n'
                    'physics_positions_1 = physics["positions"][1]\n'
                    'physics_velocities_0 = physics["velocities"][0]\n'
                    'physics_velocities_1 = physics["velocities"][1]')
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)

    def test_list_of_dicts(self):
        """TEST 8: List of dicts"""
        data = [{"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
                {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}]
        io_desc, param_fmt = describe_io("points", "points", data)
        
        # Verify unpacking expression
        expected = ('points_0_x = points[0]["x"]\n'
                    'points_0_y = points[0]["y"]\n'
                    'points_1_x = points[1]["x"]\n'
                    'points_1_y = points[1]["y"]')
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)

    def test_multiple_parameters_with_mixed_transformations(self):
        """TEST 9: Multiple parameters (with and without dots in raw names)"""
        data1 = [torch.tensor([1.0]), torch.tensor([2.0])]
        data2 = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
        io_desc1, param_fmt1 = describe_io("self_data", "self.data", data1)
        io_desc2, param_fmt2 = describe_io("config", "config", data2)
        
        # Verify unpacking expressions for each parameter (uses name_str)
        expected1 = "self_data_0 = self_data[0]\nself_data_1 = self_data[1]"
        expected2 = 'config_a = config["a"]\nconfig_b = config["b"]'
        self.assertEqual(param_fmt1.unpacked_tensor_expr, expected1)
        self.assertEqual(param_fmt2.unpacked_tensor_expr, expected2)

    def test_deeply_nested_structure(self):
        """TEST 10: Deeply nested structure"""
        data = {"level1": {"level2": [torch.tensor([1.0]), torch.tensor([2.0])]}}
        io_desc, param_fmt = describe_io("deep", "deep", data)
        
        # Verify unpacking expression
        expected = ('deep_level1_level2_0 = deep["level1"]["level2"][0]\n'
                    'deep_level1_level2_1 = deep["level1"]["level2"][1]')
        self.assertEqual(param_fmt.unpacked_tensor_expr, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)

