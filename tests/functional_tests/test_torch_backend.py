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
import os


class TestTorchBackend(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_torch_trace_backend(self):
        @annotate.method(export_with="torch", use_trace=True)
        def funcA(inputA: torch.Tensor):
            return inputA*2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        traced_model = torch.jit.load(os.path.join(
            self.TEST_GRAPH_NAME, funcA.__name__+".pt"))
        self.assertTrue(torch.allclose(
            traced_model(input_tensor), expected_output))

    def test_torch_script_backend(self):
        @annotate.method(export_with="torch", use_trace=False)
        def funcA(inputA: torch.Tensor):
            return inputA*2

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        scripted_model = torch.jit.load(os.path.join(
            self.TEST_GRAPH_NAME, funcA.__name__+".pt"))
        self.assertTrue(torch.allclose(
            scripted_model(input_tensor), expected_output))


if __name__ == '__main__':
    unittest.main(verbosity=2)
