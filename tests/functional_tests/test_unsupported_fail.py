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
import torch
from leapp import annotate
from .base import LEAPPFunctionalTestBase


class TestUnsupportedFail(LEAPPFunctionalTestBase):
    """Unit tests to see if unsupported io is properly handled"""

    def test_same_variable_used_twice(self):

        @annotate.method()
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method()
        def funcB(inputB: torch.Tensor, inputC: torch.Tensor):
            return inputB+inputC

        annotate.start(name=self.TEST_GRAPH_NAME)
        retvalA = funcA(torch.tensor([1, 2, 3]))
        funcB(retvalA, retvalA)
        annotate.stop()

        try:
            annotate.compile_graph()
        except Exception as e:
            self.assertEqual(
                str(e), "Error: unsupported use of sending the same tensor multiple times to the same node")
            return
        self.fail("Expected an exception")

    def test_function_name_overlap(self):
        node_name = "funcA"

        @annotate.method(node_name=node_name)
        def func(inputA: torch.Tensor):
            return inputA

        @annotate.method(node_name=node_name)
        def func_copy(inputA: torch.Tensor):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)

        try:
            return_value = func(torch.tensor([1, 2, 3]))
            func_copy(return_value)
        except Exception as e:
            annotate.stop()
            first_line = str(e).split('\n')[0]
            expected = "Error: funcA seen twice but detected lines do not match"
            self.assertEqual(first_line, expected)
            return

        annotate.stop()
        self.fail("Expected an exception")

    def test_io_reconciliation_name_overlap(self):
        @annotate.method()
        def funcA(input: torch.Tensor):
            detections = torch.zeros(input.shape)
            # some processing
            return detections

        @annotate.method()
        def funcB(detections):
            retval = torch.tensor([])
            # some processing
            return retval

        @annotate.method()
        def funcC(input, detections):
            retval = torch.tensor([])
            return retval

        annotate.start(name=self.TEST_GRAPH_NAME)
        detections = funcA(torch.tensor([1.0, 2.0, 3.0]))
        funcB(detections)
        funcC(detections, torch.tensor([1.0, 2.0, 3.0]))
        annotate.stop()
        try:
            annotate.compile_graph()
        except Exception as e:
            self.assertEqual(str(e),
                             "Error requesting input name change for funcC/input: detections is already in use")

    def test_mirror_leapp_tags_data_mismatch(self):
        """Test mirror_leapp_tags with various data mismatch scenarios"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        @annotate.method(export_with="torch")
        def funcB(inputA: torch.Tensor):
            return inputA, inputA + 1.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        
        # Test 1: Tensors with at least one element that is not equal
        # This should log an error but not raise an exception
        out_funcA = funcA(torch.tensor([1.0, 2.0, 3.0]))
        wrong_buffer = torch.tensor([1.0, 2.0, 4.0])  # Last element is wrong
        try:
            annotate.mirror_leapp_tags(out_funcA, wrong_buffer)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        # Test 2: List of tensors with one element wrong
        # This should log an error but not raise an exception
        out1, out2 = funcB(torch.tensor([1.0, 2.0, 3.0]))
        source_list = [out1, out2]
        target_list = [out1.clone(), torch.tensor([1.0, 2.0, 99.0])]  # Second tensor is wrong
        try:
            annotate.mirror_leapp_tags(source_list, target_list)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        # Test 3: Lists with different number of elements
        # This will cause an exception because mirror_all_tensor_tags will crash
        source_list2 = [out1, out2]
        target_list2 = [out1.clone()]  # Missing one element
        try:
            annotate.mirror_leapp_tags(source_list2, target_list2)
            self.fail("Expected an exception")
        except Exception as e:
            # Should get "unexpected error" due to index out of bounds
            self.assertIn("unexpected error", str(e))
        
        # Test 4: Dicts with different keys
        # This should log an error but not raise an exception
        source_dict = {'a': out_funcA, 'b': out_funcA + 1.0}
        target_dict = {'a': out_funcA.clone(), 'c': (out_funcA + 1.0).clone()}  # 'b' vs 'c'
        try:
            annotate.mirror_leapp_tags(source_dict, target_dict)  # Logs error, doesn't raise
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("source and target do not match", str(e))
        
        annotate.stop()


if __name__ == '__main__':
    unittest.main(verbosity=2)
