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
from .base import LEAPPFunctionalTestBase
import torch
from leapp import annotate


class TestExportSituation(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_export_nnModule_function(self):
        linear = torch.nn.Linear(3, 3)

        @annotate.method(export_with="torch", environment_constants=['linear'])
        def funcA(inputA: torch.Tensor):
            output = linear(inputA)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)

    def test_export_nnModule(self):
        class moduleA(torch.nn.Module):
            def __init__(self):
                super(moduleA, self).__init__()
                self.linear = torch.nn.Linear(3, 3)

            @annotate.method(export_with="torch")
            def forward(self, inputA: torch.Tensor):
                return self.linear(inputA)

        annotate.start(name=self.TEST_GRAPH_NAME)
        moduleA()(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)

    def test_export_nnModule_with_dict_list_io(self):

        @annotate.method(export_with="torch")
        def funcA(inputA: dict):
            list_conversion = list(inputA.values())
            return list_conversion

        input = {'a': torch.tensor([1]), 'b': torch.tensor(
            [2]), 'c': torch.tensor([3]), 'd': torch.tensor([4])}
        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = funcA(input)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=4, outputs=4,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            [input], [expected_output], funcA.__name__)

        model_info = self.inspect_torchscript_model(funcA.__name__)
        self.assertEqual(len(model_info['inputs']), 2)  # self and inputA
        self.assertEqual(len(model_info['outputs']), 1)  # list_conversion

    def test_export_nnModule_with_large_nested_dict_io(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: dict, inputB):
            underlying_values = inputA[0]['nested']
            list_conversion = list(underlying_values.values())
            return_values = list_conversion + [inputB]
            return {'return_values': [return_values]}

        input = [[{'nested': {'a': torch.tensor([1]), 'b': torch.tensor(
            [2]), 'c': torch.tensor([3]), 'd': torch.tensor([4])}}], torch.tensor([5])]
        annotate.start(name=self.TEST_GRAPH_NAME)
        expected_output = funcA(*input)
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=5, outputs=5,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            input, [expected_output], funcA.__name__)

        model_info = self.inspect_torchscript_model(funcA.__name__)
        # self, inputA, and inputB
        self.assertEqual(len(model_info['inputs']), 3)
        self.assertEqual(len(model_info['outputs']), 1)  # return_values

    def test_export_class_method_that_relies_on_dynamic_variable(self):
        class moduleA():
            def __init__(self):
                self.idx = 0
                self.stride = 3

            @annotate.method(export_with="torch", environment_constants=['self.idx', 'self.stride'])
            def get_subset(self, inputA: torch.Tensor):
                retval = inputA[self.idx:self.idx+self.stride]
                self.idx += self.stride
                return retval
        moduleA = moduleA()
        annotate.start(name=self.TEST_GRAPH_NAME)
        subset = moduleA.get_subset(torch.tensor([1, 2, 3]))
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1,
                                    internal_connections=0)
        self.verify_single_torchscript_model_expected_value(
            [torch.tensor([1, 2, 3])], [subset], moduleA.get_subset.__name__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
