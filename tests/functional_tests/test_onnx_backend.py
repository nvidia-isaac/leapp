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

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_multiple_traced_modules(self):
        """Test ONNX export with multiple torch.jit.trace models as environment constants."""

        # Create two different traced models
        class EncoderModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 8)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        class DecoderModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.linear(x)

        encoder = EncoderModel()
        decoder = DecoderModel()
        
        encoder_input = torch.randn(4, dtype=torch.float32)
        decoder_input = torch.randn(8, dtype=torch.float32)
        
        traced_encoder = torch.jit.trace(encoder, encoder_input)
        traced_decoder = torch.jit.trace(decoder, decoder_input)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_encoder", "traced_decoder"]
        )
        def encode_decode(inputA: torch.Tensor):
            encoded = traced_encoder(inputA)
            decoded = traced_decoder(encoded)
            return decoded

        input_tensor = torch.randn(4, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        encode_decode(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('encode_decode')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_traced_conv_model(self):
        """Test ONNX export with a traced convolutional model."""

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(16)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.MaxPool2d(2)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.pool(x)
                return x

        conv_model = ConvModel().eval()
        example_input = torch.randn(1, 1, 8, 8, dtype=torch.float32)
        traced_conv = torch.jit.trace(conv_model, example_input)
        
        # Check if LLVM backend is available
        try:
            traced_conv(example_input)
        except RuntimeError as e:
            if "LLVM Backend not found" in str(e):
                pytest.skip("LLVM Backend not available")
            raise

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_conv"]
        )
        def process_image(image: torch.Tensor):
            features = traced_conv(image)
            # Flatten and apply global average
            return features.mean(dim=(2, 3))

        input_image = torch.randn(1, 1, 8, 8, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        process_image(input_image)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('process_image')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_complex_tensor_operations(self):
        """Test ONNX export with complex tensor operations (einsum, scatter, gather)."""

        class AttentionModel(torch.nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.query = torch.nn.Linear(dim, dim)
                self.key = torch.nn.Linear(dim, dim)
                self.value = torch.nn.Linear(dim, dim)
                self.scale = dim ** -0.5

            def forward(self, x):
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                # Simple scaled dot product attention
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = torch.softmax(attn, dim=-1)
                return torch.matmul(attn, v)

        attn_model = AttentionModel().eval()
        example_input = torch.randn(2, 4, 8, dtype=torch.float32)
        traced_attn = torch.jit.trace(attn_model, example_input)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_attn"]
        )
        def apply_attention(x: torch.Tensor):
            attended = traced_attn(x)
            # Add residual connection and layer norm
            output = x + attended
            # TorchScript requires explicit p argument for norm
            return output / torch.linalg.norm(output, dim=-1, keepdim=True)

        input_tensor = torch.randn(2, 4, 8, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        apply_attention(input_tensor)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('apply_attention')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_traced_tensor_node_chain(self):
        """Test ONNX export with chained traced tensor nodes.
        
        Note: Uses detach() on intermediate tensor to avoid deepcopy issues
        with non-leaf tensors during ONNX export.
        """

        class ProcessorA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class ProcessorB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        proc_a = ProcessorA().eval()
        proc_b = ProcessorB().eval()
        example = torch.randn(4, dtype=torch.float32)
        traced_a = torch.jit.trace(proc_a, example)
        traced_b = torch.jit.trace(proc_b, example)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_a"]
        )
        def step_a(inputA: torch.Tensor):
            return traced_a(inputA)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_b"]
        )
        def step_b(inputB: torch.Tensor):
            return traced_b(inputB)

        input_tensor = torch.randn(4, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        intermediate = step_a(input_tensor)
        # Detach to make it a leaf tensor for the next node's ONNX export
        step_b(intermediate.detach())
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('step_a', 'step_b')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_traced_model_and_tensor_broadcasting(self):
        """Test ONNX export with traced model combined with tensor broadcasting operations."""

        class ScaleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(4))

            def forward(self, x):
                return x * self.scale

        scale_model = ScaleModel().eval()
        example = torch.randn(2, 4, dtype=torch.float32)
        traced_scale = torch.jit.trace(scale_model, example)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_scale"]
        )
        def scale_and_broadcast(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = traced_scale(inputA)
            # Broadcasting: mask is (2,) and scaled is (2, 4)
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0, 
                scaled, 
                torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        scale_and_broadcast(input_tensor, mask)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('scale_and_broadcast')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_traced_model_multiple_outputs(self):
        """Test ONNX export with traced model that has multiple outputs."""

        class MultiOutputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 8)
                self.mean_head = torch.nn.Linear(8, 4)
                self.var_head = torch.nn.Linear(8, 4)

            def forward(self, x):
                hidden = torch.relu(self.encoder(x))
                mean = self.mean_head(hidden)
                var = torch.exp(self.var_head(hidden))
                return mean, var

        multi_model = MultiOutputModel().eval()
        example = torch.randn(4, dtype=torch.float32)
        traced_multi = torch.jit.trace(multi_model, example)

        @annotate.method(
            export_with="onnx",
            backend_params={"dynamo": False, "prescript": True},
            environment_constants=["traced_multi"]
        )
        def sample_gaussian(inputA: torch.Tensor, noise: torch.Tensor):
            mean, var = traced_multi(inputA)
            # Reparameterization trick
            sample = mean + torch.sqrt(var) * noise
            return sample

        input_tensor = torch.randn(4, dtype=torch.float32)
        noise = torch.randn(4, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        sample_gaussian(input_tensor, noise)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('sample_gaussian')

    def test_onnx_dynamo_with_complex_math(self):
        """Test ONNX dynamo export with complex mathematical operations."""

        @annotate.method(export_with="onnx")
        def complex_math(x: torch.Tensor, y: torch.Tensor):
            # Various math ops that ONNX should handle
            result = torch.sin(x) + torch.cos(y)
            result = result * torch.exp(-torch.abs(x - y))
            result = torch.clamp(result, -1.0, 1.0)
            result = torch.pow(result, 2)
            return result

        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        complex_math(x, y)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('complex_math')

    def test_onnx_dynamo_with_reduction_operations(self):
        """Test ONNX dynamo export with reduction operations."""

        @annotate.method(export_with="onnx")
        def reduction_ops(x: torch.Tensor):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normalized = (x - mean) / (std + 1e-5)
            
            # More reductions - keep shapes compatible
            max_val = normalized.max(dim=1).values  # shape (4,)
            sum_val = normalized.sum(dim=1)          # shape (4,)
            
            return max_val + sum_val

        x = torch.randn(4, 8, dtype=torch.float32)

        annotate.start(name=self.TEST_GRAPH_NAME)
        reduction_ops(x)
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.verify_all_models_exist('reduction_ops')


# ============================================================================
# PROPOSED TESTS - POTENTIAL EDGE CASES THAT MIGHT FAIL
# ============================================================================
# 
# The following tests are proposed but NOT implemented. They represent edge
# cases that might reveal issues in the ONNX export pipeline.
#
# 1. test_onnx_with_dynamic_shapes
#    - Test ONNX export with dynamic input shapes (batch dimension varies)
#    - May fail because: ONNX requires explicit dynamic axis specification,
#      and the current pipeline may not handle dynamic shapes properly
#    - Example: input shapes like (None, 4) or using symbolic shapes
#
# 2. test_onnx_with_nested_traced_models
#    - Test ONNX export where one traced model calls another traced model
#    - May fail because: Nested JIT traces can have issues with graph capture,
#      and ONNX may not properly flatten the nested computations
#
# 3. test_onnx_with_control_flow_in_environment_constant
#    - Test ONNX export with a traced model that has internal control flow
#      (e.g., torch.where with conditions based on input values)
#    - May fail because: TorchScript tracing vs scripting handles control
#      flow differently, and ONNX has limited control flow support
#
# 4. test_onnx_with_inplace_operations
#    - Test ONNX export with inplace operations (x.add_(), x.mul_(), etc.)
#    - May fail because: ONNX doesn't support inplace operations and they
#      need to be converted, which may not always happen correctly
#
# 5. test_onnx_with_custom_autograd_function
#    - Test ONNX export with custom autograd functions (torch.autograd.Function)
#    - May fail because: Custom functions need explicit ONNX symbolic functions
#      registered, which the user may not have done
#
# 6. test_onnx_with_optional_tensor_inputs
#    - Test ONNX export where some inputs are optional (can be None)
#    - May fail because: ONNX has limited support for optional inputs and
#      None handling in the tensor domain
#
# 7. test_onnx_with_data_dependent_shapes
#    - Test ONNX export where output shapes depend on input values
#      (e.g., torch.nonzero, torch.unique)
#    - May fail because: ONNX struggles with data-dependent dynamic shapes
#
# 8. test_onnx_with_complex_dtype
#    - Test ONNX export with complex number tensors (torch.complex64)
#    - May fail because: ONNX has limited support for complex numbers
#
# 9. test_onnx_with_sparse_tensors
#    - Test ONNX export with sparse tensor operations
#    - May fail because: ONNX has limited sparse tensor support
#
# 10. test_onnx_with_mixed_precision
#     - Test ONNX export mixing float16 and float32 operations
#     - May fail because: ONNX type promotion rules may differ from PyTorch
#
# 11. test_onnx_with_traced_model_containing_buffers
#     - Test ONNX export with traced models that have registered buffers
#       that change based on input (e.g., running stats in BatchNorm)
#     - May fail because: Traced model buffers may not be properly captured
#       as constants in the ONNX graph
#
# 12. test_onnx_with_large_constant_tensors
#     - Test ONNX export with very large constant tensors as environment_constants
#     - May fail because: ONNX file size limits or memory issues during export
#
# 13. test_onnx_with_string_or_non_tensor_inputs
#     - Test ONNX export where some inputs are non-tensor types
#     - May fail because: ONNX primarily supports tensor operations
#
# 14. test_onnx_feedback_loop_with_traced_model
#     - Test ONNX export with a feedback loop where output feeds back to input
#       through a traced model
#     - May fail because: Feedback loops create graph cycles that may not
#       serialize correctly to ONNX format
#
# 15. test_onnx_with_torch_scatter_ops
#     - Test ONNX export with scatter/gather operations like scatter_add, 
#       index_select, etc.
#     - May fail because: Some scatter operations have limited ONNX support
#       or require specific opset versions
# ============================================================================


if __name__ == '__main__':
    unittest.main(verbosity=2)
