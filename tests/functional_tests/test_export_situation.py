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


if __name__ == '__main__':
    unittest.main(verbosity=2)
