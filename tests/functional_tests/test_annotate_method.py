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


class TestAnnotateMethod(LEAPPFunctionalTestBase):
    def test_annotate_method(self):
        """tests the basic situation of using the annotate.method decorator"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="torch")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            outputC = funcC(outputA)
        annotate.stop()

        self.assertEqual(len(annotate.nodes), 2)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].outputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].outputs), 1)
        self.assertEqual(
            annotate.nodes[funcA.__name__].inputs[0].name, "inputA")
        self.assertEqual(
            annotate.nodes[funcC.__name__].inputs[0].name, "inputB")

    def test_annotate_method_with_kwargs_and_default_value(self):
        """tests the situation where the function has and default values"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA()
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 0)

        expected_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        # check if using default
        self.verify_single_torchscript_model_expected_value(
            [], [expected_output], funcA.__name__)
        # Or get structured data
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 1)
        self.assertEqual(len(model_info['outputs']), 1)

    def test_annotate_method_ignoring_default_values(self):
        """tests the situation where we pass in a value overriding the default"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 2)
        self.assertEqual(len(model_info['outputs']), 1)

    def test_annotate_method_ignoring_middle_kwargs(self):
        """tests the situation where the user provides kwargs out of order"""
        default_tensor = torch.tensor([0])

        @annotate.method(export_with="torch")
        def funcA(input1=default_tensor, input2=default_tensor, input3=default_tensor,
                  input4=default_tensor, input5=default_tensor):
            output = torch.cat([input1, input2, input3, input4, input5], dim=0)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(input1=torch.tensor([1]), input4=torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 2)
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 3)
        self.assertEqual(len(model_info['outputs']), 1)

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1]), torch.tensor([1])], [outputA], funcA.__name__)

    def test_annotate_method_kwargs_out_of_order(self):
        """tests the situation where the user provides kwargs out of order"""
        default_tensor = torch.tensor([0])

        @annotate.method(export_with="torch")
        def funcA(input1=default_tensor, input2=default_tensor, input3=default_tensor,
                  input4=default_tensor, input5=default_tensor):
            output = torch.cat([input1, input2, input3, input4, input5], dim=0)
            return output
        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(input4=torch.tensor(
            [2]), input1=torch.tensor([1]), input5=torch.tensor([3]))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 3)
        input_format = annotate.detected_nodes[funcA.__name__]['formatting']['input_format']
        self.assertEqual(input_format, ['input1', 'input4', 'input5'])
        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 4)
        self.assertEqual(len(model_info['outputs']), 1)
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])], [outputA], funcA.__name__)

    def test_annotate_method_with_multiple_unnamed_returns(self):
        """tests the situation where the function has multiple unnamed returns"""
        @annotate.method(export_with="torch")
        def funcA(input1: torch.Tensor):
            return input1+1, input1+2, input1+3

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA, outputB, outputC = funcA(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].outputs), 3)

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1])], [outputA, outputB, outputC], funcA.__name__)

    def test_annotate_method_with_custom_returns(self):
        """tests the situation where the function has custom returns"""

        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self):
                self.counter += 1

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        mock_module.count()
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 0)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 1)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [], [torch.tensor([1])], 'counter')

    def test_annotate_method_with_mixed_returns(self):
        """tests the situation where the function has both default and custom returns"""

        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self, input: torch.Tensor):
                self.counter += 1
                retval = input*self.counter
                return retval

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        output = mock_module.count(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 1)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 2)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1])], [output, torch.tensor([1])], 'counter')

    def test_annotate_method_with_mixed_returns_in_multiple_locations(self):
        """tests the situation where the function has both default and custom returns in multiple locations"""
        class MockModule:
            def __init__(self):
                self.counter = torch.tensor([0])

            @annotate.method(node_name="counter", export_with="torch", outputs=["self.counter"], register_buffers=["self.counter"])
            def count(self, input: torch.Tensor):
                self.counter += 1
                if input.sum() > 0:
                    return input*self.counter
                else:
                    retval = input+self.counter
                    return retval

        mock_module = MockModule()

        annotate.start(name=self.TEST_GRAPH_NAME)
        output = mock_module.count(torch.tensor([1]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes['counter'].inputs), 1)
        self.assertEqual(len(annotate.nodes['counter'].outputs), 2)
        self.assertEqual(
            annotate.detected_nodes['counter']['outputs'][0]['type'], 'tensor')
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([2])], [torch.tensor([2]), torch.tensor([1])], 'counter')

        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([0])], [torch.tensor([1]), torch.tensor([1])], 'counter')


if __name__ == '__main__':
    unittest.main(verbosity=2)
