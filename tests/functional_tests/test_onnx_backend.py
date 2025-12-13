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
import pytest



class TestOnnxBackend(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_onnx_backend(self):
        @annotate.method(export_with="onnx")
        def funcA(inputA: torch.Tensor):
            return inputA*2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_backend_script(self):
        @annotate.method(export_with="onnx", backend_params={"dynamo": False})
        def funcA(inputA: torch.Tensor):
            return inputA*2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)
    
    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_backend_prescript_with_traced_model(self):
        """Test ONNX export with a torch.jit.trace model, prescript=True, and conditional flow."""

        # Create a simple model and trace it
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        simple_model = SimpleModel()
        example_input = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        traced_model = torch.jit.trace(simple_model, example_input)

        # Use the traced model in an annotated method with environment_constants
        # and conditional flow (using torch.where for TorchScript compatibility)
        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_model"]
        )
        def funcA(inputA: torch.Tensor, threshold: torch.Tensor):
            # Apply the traced model
            output = traced_model(inputA)
            # Conditional flow: threshold with torch.where
            result = torch.where(output > threshold, output, torch.zeros_like(output))
            return result

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        threshold = torch.tensor(0.5, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor, threshold)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        


if __name__ == '__main__':
    unittest.main(verbosity=2)
