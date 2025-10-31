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
from leapp import MergeCfgEnum


class TestNodeMerging(LEAPPFunctionalTestBase):

    def test_combine_two_nodes_automatically(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor, inputA2: torch.Tensor):
            return inputA + inputA2

        @annotate.method(export_with="torch")
        def funcB(inputB: torch.Tensor):
            output1 = inputB * 2.0
            output2 = inputB * 3.0
            return output1, output2

        @annotate.method(export_with="torch")
        def funcC(inputC: torch.Tensor):
            a = inputC - 1.0
            b = inputC
            return a, b

        @annotate.method(export_with="torch")
        def funcD(inputD: torch.Tensor):
            a = inputD - 1.0
            return a

        @annotate.method(export_with="torch")
        def funcE(inputE: torch.Tensor):
            a = inputE + 1.0
            return a

        annotate.start(name="test_graph")

        out = funcA(torch.tensor([1.0, 1.0, 1.0]),
                    torch.tensor([2.0, 2.0, 2.0]))
        outb1, outb2 = funcB(out)
        outc1, outc2 = funcC(outb1)
        outd = funcD(outb2)
        oute = funcE(outd)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=3, inputs=2, outputs=3,
                                    internal_connections=2)
        expected_feedback_node_name1 = '-'.join(['funcA', 'funcB'])
        expected_feedback_node_name2 = '-'.join(['funcD', 'funcE'])
        expected_node_names = [
            funcC.__name__, expected_feedback_node_name1, expected_feedback_node_name2]
        for node_name in expected_node_names:
            self.assertIn(node_name, list(annotate.nodes.keys()))
        input_tensor = [torch.tensor(
            [1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])]
        expected_output = funcB(funcA(*input_tensor))
        self.verify_single_torchscript_model_expected_value(
            input_tensor, expected_output, expected_feedback_node_name1)

    def test_combine_four_nodes_automatically(self):
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA + 1.0

        @annotate.method(export_with="torch")
        def funcB(inputB: torch.Tensor):
            return inputB*2.0

        @annotate.method(export_with="torch")
        def funcC(inputC: torch.Tensor):
            return inputC*3.0

        @annotate.method(export_with="torch")
        def funcD(inputD: torch.Tensor):
            return inputD*4.0

        input_tensor = torch.tensor([1.0, 1.0, 1.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        out = funcA(input_tensor.clone().detach())
        outb1 = funcB(out)
        outc1 = funcC(outb1)
        outd = funcD(outc1)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        expected_output = outd
        expected_name = "funcA-funcB-funcC-funcD"

        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1,
                                    internal_connections=0)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(list(annotate.nodes.keys())[0],
                         expected_name)
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], expected_name)

    def test_combine_graph_with_feedback_automatically(self):

        @annotate.method(export_with='torch')
        def funcA(inputA: torch.Tensor, loop_back: torch.Tensor):
            return inputA + loop_back

        @annotate.method(export_with='torch')
        def funcB(inputB: torch.Tensor):
            return inputB * 2.0

        @annotate.method(export_with='torch')
        def funcC(inputC: torch.Tensor):
            return inputC + 1.0

        @annotate.method(export_with='torch')
        def funcD(inputD: torch.Tensor):
            return inputD/2.0
        input_tensor = torch.tensor([1.0, 1.0, 1.0])
        annotate.start(name=self.TEST_GRAPH_NAME)
        loop_back_input = torch.tensor([0.0, 0.0, 0.0])
        for i in range(3):
            out = funcA(input_tensor.clone().detach(),
                        loop_back_input)
            outb1 = funcB(out)
            loop_back_input = outb1.clone()
            outc1 = funcC(outb1)
            outd = funcD(outc1)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=3, inputs=1, outputs=1,
                                    internal_connections=2, feedback_connections=1)

        expected_feedback_node_name = '-'.join(['funcC', 'funcD'])
        expected_node_names = [funcA.__name__,
                               funcB.__name__, expected_feedback_node_name]
        for node_name in expected_node_names:
            self.assertIn(node_name, list(annotate.nodes.keys()))
        expected_output = funcD(funcC(input_tensor.clone().detach()))
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], expected_feedback_node_name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
