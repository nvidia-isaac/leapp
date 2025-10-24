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
        outd = funcE(outd)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=3, inputs=2, outputs=3,
                                    internal_connections=2)

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        out = funcA(torch.tensor([1.0, 1.0, 1.0]))
        outb1 = funcB(out)
        outc1 = funcC(outb1)
        outd = funcD(outc1)
        annotate.stop()
        annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)

        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1,
                                    internal_connections=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
