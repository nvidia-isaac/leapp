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
import os
import unittest
from unittest import mock
import torch
import yaml
import onnx
import leapp
from leapp.leapp import _MANAGER as annotate
from leapp import TensorSemantics, InputKindEnum, OutputKindEnum
from leapp.backends.export_backend import SimplifiedONNXProgram
from .base import LEAPPFunctionalTestBase
import pytest



class TestOnnxProviderSelection(unittest.TestCase):
    def test_cuda_provider_is_always_preferred(self):
        program = SimplifiedONNXProgram("fake.onnx")

        with mock.patch(
            "leapp.backends.export_backend.ort.get_available_providers",
            return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
        ):
            self.assertEqual(
                program._get_providers(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.assertEqual(
                program._get_providers(),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def test_warns_when_cuda_provider_is_unavailable(self):
        program = SimplifiedONNXProgram("fake.onnx")

        with mock.patch(
            "leapp.backends.export_backend.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ), mock.patch("leapp.backends.export_backend._get_logger") as get_logger:
            providers = program._get_providers()

        self.assertEqual(providers, ["CPUExecutionProvider"])
        get_logger.return_value.warning.assert_called_once_with(
            "CUDA execution provider not available. Falling back to CPU."
        )

    def test_cuda_iobinding_helper_binds_real_session_calls(self):
        program = SimplifiedONNXProgram("fake.onnx")
        binding = mock.Mock()
        fake_session = mock.Mock()
        fake_session.io_binding.return_value = binding
        program._session = fake_session
        program._input_names = ["input"]

        output_meta = mock.Mock()
        output_meta.name = "output"
        output_meta.shape = [2, 3]
        output_meta.type = "tensor(float)"
        program._output_metas = [output_meta]

        input_tensor = torch.randn(3, 2, dtype=torch.float32).transpose(0, 1)
        outputs = program._run_with_cuda_iobinding((input_tensor,))

        fake_session.io_binding.assert_called_once_with()
        binding.bind_input.assert_called_once()
        binding.bind_output.assert_called_once()
        fake_session.run_with_iobinding.assert_called_once_with(binding)

        input_kwargs = binding.bind_input.call_args.kwargs
        self.assertEqual(input_kwargs["name"], "input")
        self.assertEqual(input_kwargs["device_type"], "cuda")
        self.assertEqual(input_kwargs["device_id"], 0)
        self.assertEqual(input_kwargs["element_type"], torch.tensor([], dtype=torch.float32).numpy().dtype.type)
        self.assertEqual(input_kwargs["shape"], (2, 3))
        self.assertIsInstance(input_kwargs["buffer_ptr"], int)
        self.assertGreater(input_kwargs["buffer_ptr"], 0)

        output_kwargs = binding.bind_output.call_args.kwargs
        self.assertEqual(output_kwargs["name"], "output")
        self.assertEqual(output_kwargs["device_type"], "cuda")
        self.assertEqual(output_kwargs["device_id"], 0)
        self.assertEqual(output_kwargs["element_type"], torch.tensor([], dtype=torch.float32).numpy().dtype.type)
        self.assertEqual(output_kwargs["shape"], (2, 3))
        self.assertIsInstance(output_kwargs["buffer_ptr"], int)
        self.assertGreater(output_kwargs["buffer_ptr"], 0)

        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], torch.Tensor)
        self.assertEqual(tuple(outputs[0].shape), (2, 3))
        self.assertEqual(outputs[0].dtype, torch.float32)

    def test_onnx_program_uses_iobinding_when_cuda_provider_is_active(self):
        program = SimplifiedONNXProgram("fake.onnx")
        fake_session = mock.Mock()
        fake_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        program._session = fake_session
        program._input_names = ["input"]
        program._output_metas = []
        program._active_provider = "CUDAExecutionProvider"

        with mock.patch.object(
            program, "_can_use_cuda_iobinding", return_value=True
        ) as can_use, mock.patch.object(
            program, "_run_with_cuda_iobinding", return_value=("fast-path",)
        ) as run_cuda, mock.patch.object(
            program, "_run_with_standard_inference"
        ) as run_standard:
            result = program(torch.tensor([1.0], dtype=torch.float32))

        can_use.assert_called_once()
        run_cuda.assert_called_once()
        run_standard.assert_not_called()
        self.assertEqual(result, ("fast-path",))

    def test_onnx_program_falls_back_without_cuda_iobinding(self):
        program = SimplifiedONNXProgram("fake.onnx")
        fake_session = mock.Mock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        program._session = fake_session
        program._input_names = ["input"]
        program._output_metas = []
        program._active_provider = "CPUExecutionProvider"

        with mock.patch.object(
            program, "_can_use_cuda_iobinding", return_value=False
        ) as can_use, mock.patch.object(
            program, "_run_with_cuda_iobinding"
        ) as run_cuda, mock.patch.object(
            program, "_run_with_standard_inference", return_value=("cpu-path",)
        ) as run_standard:
            result = program(torch.tensor([1.0], dtype=torch.float32))

        can_use.assert_called_once()
        run_cuda.assert_not_called()
        run_standard.assert_called_once()
        self.assertEqual(result, ("cpu-path",))


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

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

    def test_onnx_backend_reverse_scalar_mul(self):
        @annotate.method(export_with="onnx")
        def funcA(inputA: torch.Tensor):
            return 2.0 * inputA

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

    def test_onnx_backend_reverse_scalar_add(self):
        @annotate.method(export_with="onnx")
        def funcA(inputA: torch.Tensor):
            return 2.0 + inputA

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_backend_script(self):
        @annotate.method(export_with="onnx-torchscript")
        def funcA(inputA: torch.Tensor):
            return inputA*2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_backend_name_overlap_is_renamed(self):
        """Overlapping input/output names should be disambiguated for ONNX export."""
        cases = [
            ("onnx",),
            ("onnx-torchscript",),
        ]

        for (export_with,) in cases:
            with self.subTest(export_with=export_with):
                leapp.start(name=self.TEST_GRAPH_NAME)
                traced = annotate.input_tensors(
                    "func_overlap",
                    TensorSemantics(
                        name="joint_pos",
                        ref=torch.tensor([1.0, 2.0, 3.0]),
                        kind=InputKindEnum.JOINT_POSITION,
                    ),
                )
                annotate.output_tensors(
                    "func_overlap",
                    TensorSemantics(
                        name="joint_pos",
                        ref=traced + 1.0,
                        kind=OutputKindEnum.JOINT_POSITION,
                    ),
                    export_with=export_with,
                )
                leapp.stop()
                leapp.compile_graph(visualize=False)

                yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
                with open(yaml_path) as f:
                    exported = yaml.safe_load(f)

                model_desc = exported["models"]["func_overlap"]
                input_names = [desc["name"] for desc in model_desc["inputs"]]
                output_names = [desc["name"] for desc in model_desc["outputs"]]
                self.assertEqual(input_names, ["joint_pos_in"])
                self.assertEqual(output_names, ["joint_pos_out"])
                self.assertEqual(
                    [desc["kind"] for desc in model_desc["inputs"]],
                    [InputKindEnum.JOINT_POSITION.value],
                )
                self.assertEqual(
                    [desc["kind"] for desc in model_desc["outputs"]],
                    [OutputKindEnum.JOINT_POSITION.value],
                )

                model_path = os.path.join(self.TEST_GRAPH_NAME, "func_overlap.onnx")
                model = onnx.load(model_path)
                initializer_names = {init.name for init in model.graph.initializer}
                onnx_input_names = [
                    value.name for value in model.graph.input
                    if value.name not in initializer_names
                ]
                onnx_output_names = [value.name for value in model.graph.output]
                self.assertEqual(onnx_input_names, ["joint_pos_in"])
                self.assertEqual(onnx_output_names, ["joint_pos_out"])

    def test_onnx_connected_node_names_match_artifact_names(self):
        """Connected nodes should keep each node's own ONNX I/O names."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        source_input = annotate.input_tensors(
            "producer",
            {
                "source_input": torch.tensor(
                    [1.0, 2.0, 3.0], dtype=torch.float32)
            },
        )
        producer_output = source_input * 2.0
        annotate.output_tensors(
            "producer",
            {"producer_original_output": producer_output},
            export_with="onnx",
        )

        consumer_input = annotate.input_tensors(
            "consumer",
            {"consumer_input_name": producer_output},
        )
        annotate.output_tensors(
            "consumer",
            {"final_output": consumer_input + 1.0},
            export_with="onnx",
        )

        leapp.stop()
        leapp.compile_graph(visualize=False, validate=False)

        yaml_path = os.path.join(
            self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(yaml_path) as f:
            exported = yaml.safe_load(f)

        producer_desc = exported["models"]["producer"]
        producer_yaml_outputs = [
            desc["name"] for desc in producer_desc["outputs"]
        ]
        self.assertEqual(
            producer_yaml_outputs,
            ["producer_original_output"],
        )
        self.assertEqual(
            exported["pipeline"]["data_flow"],
            {
                "producer/producer_original_output": [
                    "consumer/consumer_input_name"
                ]
            },
        )

        model_path = os.path.join(self.TEST_GRAPH_NAME, "producer.onnx")
        model = onnx.load(model_path)
        onnx_output_names = [value.name for value in model.graph.output]
        self.assertEqual(onnx_output_names, producer_yaml_outputs)

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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
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

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor, threshold)
        leapp.stop()
        leapp.compile_graph(visualize=False)

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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_encoder", "traced_decoder"]
        )
        def encode_decode(inputA: torch.Tensor):
            encoded = traced_encoder(inputA)
            decoded = traced_decoder(encoded)
            return decoded

        input_tensor = torch.randn(4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        encode_decode(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_conv"]
        )
        def process_image(image: torch.Tensor):
            features = traced_conv(image)
            # Flatten and apply global average
            return features.mean(dim=(2, 3))

        input_image = torch.randn(1, 1, 8, 8, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_image(input_image)
        leapp.stop()
        leapp.compile_graph(visualize=False)
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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_attn"]
        )
        def apply_attention(x: torch.Tensor):
            attended = traced_attn(x)
            # Add residual connection and layer norm
            output = x + attended
            # TorchScript requires explicit p argument for norm
            return output / torch.linalg.norm(output, dim=-1, keepdim=True)

        input_tensor = torch.randn(2, 4, 8, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        apply_attention(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False, atol=1e-3)
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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_a"]
        )
        def step_a(inputA: torch.Tensor):
            return traced_a(inputA)

        @annotate.method(
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_b"]
        )
        def step_b(inputB: torch.Tensor):
            return traced_b(inputB)

        input_tensor = torch.randn(4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        intermediate = step_a(input_tensor)
        # Detach to make it a leaf tensor for the next node's ONNX export
        step_b(intermediate.detach())
        leapp.stop()
        leapp.compile_graph(visualize=False)
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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
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

        leapp.start(name=self.TEST_GRAPH_NAME)
        scale_and_broadcast(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
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
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_multi"]
        )
        def sample_gaussian(inputA: torch.Tensor, noise: torch.Tensor):
            mean, var = traced_multi(inputA)
            # Reparameterization trick
            sample = mean + torch.sqrt(var) * noise
            return sample

        input_tensor = torch.randn(4, dtype=torch.float32)
        noise = torch.randn(4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        sample_gaussian(input_tensor, noise)
        leapp.stop()
        leapp.compile_graph(visualize=False)
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

        leapp.start(name=self.TEST_GRAPH_NAME)
        complex_math(x, y)
        leapp.stop()
        leapp.compile_graph(visualize=False)
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

        leapp.start(name=self.TEST_GRAPH_NAME)
        reduction_ops(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('reduction_ops')

    def test_onnx_dict_inputs_to_list_outputs(self):
        """Test ONNX export with dict-like inputs and list-like outputs.
        
        This tests that the ONNX exporter correctly handles:
        - Flattening dict inputs into individual tensor inputs
        - Packing multiple outputs into a list
        """

        @annotate.method(export_with="onnx")
        def dict_to_list(inputs: dict):
            a = inputs['a']
            b = inputs['b']
            # Return as a list of outputs
            sum_result = a + b
            diff_result = a - b
            return [sum_result, diff_result]

        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        dict_to_list(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('dict_to_list')

    def test_onnx_list_inputs_to_dict_outputs(self):
        """Test ONNX export with list-like inputs and dict-like outputs.
        
        This tests that the ONNX exporter correctly handles:
        - Flattening list inputs into individual tensor inputs
        - Unpacking dict outputs into individual tensors
        """

        @annotate.method(export_with="onnx")
        def list_to_dict(inputs: list):
            a = inputs[0]
            b = inputs[1]
            # Return as a dict of outputs
            return {
                'sum': a + b,
                'product': a * b,
            }

        input_list = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        list_to_dict(input_list)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('list_to_dict')

    def test_onnx_nested_dict_input(self):
        """Test ONNX export with deeply nested dict input structure."""

        @annotate.method(export_with="onnx")
        def process_nested_dict(nested: dict):
            # Access deeply nested values
            a = nested['level1']['a']
            b = nested['level1']['b']
            c = nested['level2']['c']
            return a + b + c

        nested_input = {
            'level1': {
                'a': torch.randn(3, 4, dtype=torch.float32),
                'b': torch.randn(3, 4, dtype=torch.float32),
            },
            'level2': {
                'c': torch.randn(3, 4, dtype=torch.float32),
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_dict(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_nested_dict')

    def test_onnx_nested_list_input(self):
        """Test ONNX export with nested list input structure."""

        @annotate.method(export_with="onnx")
        def process_nested_list(nested: list):
            # Access nested list values
            a = nested[0][0]
            b = nested[0][1]
            c = nested[1][0]
            return a * b + c

        nested_input = [
            [torch.randn(3, 4, dtype=torch.float32), torch.randn(3, 4, dtype=torch.float32)],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_list(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_nested_list')

    def test_onnx_mixed_dict_list_input(self):
        """Test ONNX export with mixed dict and list inputs."""

        @annotate.method(export_with="onnx")
        def process_mixed_inputs(dict_input: dict, list_input: list):
            a = dict_input['a']
            b = dict_input['b']
            c = list_input[0]
            d = list_input[1]
            return a + b, c * d

        dict_input = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_mixed_inputs(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_mixed_inputs')

    def test_onnx_nested_list_output(self):
        """Test ONNX export with nested list output structure."""

        @annotate.method(export_with="onnx")
        def create_nested_list_output(x: torch.Tensor):
            # Create nested list output
            return [[x[0:1], x[1:2]], [x[2:3], x[3:4]]]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_nested_list_output(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_nested_list_output')

    def test_onnx_dict_of_lists_output(self):
        """Test ONNX export with dict containing list values as output."""

        @annotate.method(export_with="onnx")
        def create_dict_of_lists(x: torch.Tensor):
            return {
                'group_a': [x[0:1], x[1:2]],
                'group_b': [x[2:3], x[3:4]],
            }

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_dict_of_lists(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_dict_of_lists')

    def test_onnx_list_of_dicts_output(self):
        """Test ONNX export with list containing dict values as output."""

        @annotate.method(export_with="onnx")
        def create_list_of_dicts(x: torch.Tensor):
            return [
                {'a': x[0:1], 'b': x[1:2]},
                {'a': x[2:3], 'b': x[3:4]},
            ]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_list_of_dicts(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_list_of_dicts')

    def test_onnx_mixed_return_types(self):
        """Test ONNX export with mixed return types: tensor, list, and dict."""

        @annotate.method(export_with="onnx")
        def mixed_outputs(x: torch.Tensor):
            single = x[0:1]
            list_out = [x[1:2], x[2:3]]
            dict_out = {'a': x[3:4], 'b': x[4:5]}
            return single, list_out, dict_out

        input_tensor = torch.randn(5, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        mixed_outputs(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('mixed_outputs')

    def test_onnx_large_nested_structure(self):
        """Test ONNX export with large nested dict/list structure (stress test)."""

        @annotate.method(export_with="onnx")
        def process_large_structure(data: dict):
            # Process a large nested structure
            results = []
            for i in range(4):
                group = data[f'group_{i}']
                for j in range(2):
                    results.append(group[j] * 2.0)
            # Return sum of all processed tensors
            return torch.stack(results).sum(dim=0)

        # Create large nested structure: 4 groups, each with 2 tensors
        large_input = {
            f'group_{i}': [
                torch.randn(3, 4, dtype=torch.float32) for _ in range(2)
            ] for i in range(4)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_large_structure(large_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_large_structure')

    def test_onnx_deeply_nested_dict_input(self):
        """Test ONNX export with deeply nested dict input (3+ levels)."""

        @annotate.method(export_with="onnx")
        def process_deep_dict(nested: dict):
            # Access 3-level deep nested values
            a = nested['l1']['l2']['l3']['a']
            b = nested['l1']['l2']['l3']['b']
            return a + b

        deep_input = {
            'l1': {
                'l2': {
                    'l3': {
                        'a': torch.randn(3, 4, dtype=torch.float32),
                        'b': torch.randn(3, 4, dtype=torch.float32),
                    }
                }
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_deep_dict(deep_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_deep_dict')

    def test_onnx_many_tensor_inputs(self):
        """Test ONNX export with many individual tensor inputs (stress test)."""

        @annotate.method(export_with="onnx")
        def sum_many_tensors(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, 
                             t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor,
                             t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, 
                             t9: torch.Tensor):
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

        tensors = [torch.randn(3, 4, dtype=torch.float32) for _ in range(10)]

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_many_tensors(*tensors)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('sum_many_tensors')

    def test_onnx_many_dict_tensor_inputs(self):
        """Test ONNX export with dict containing many tensors."""

        @annotate.method(export_with="onnx")
        def sum_dict_tensors(data: dict):
            total = data['t0']
            for i in range(1, 10):
                total = total + data[f't{i}']
            return total

        input_dict = {f't{i}': torch.randn(3, 4, dtype=torch.float32) for i in range(10)}

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_dict_tensors(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('sum_dict_tensors')

    def test_onnx_split_tensor_to_many_outputs(self):
        """Test ONNX export that splits one tensor into many outputs."""

        @annotate.method(export_with="onnx")
        def split_to_many(x: torch.Tensor):
            # Split tensor into 8 parts
            return [x[i:i+1] for i in range(8)]

        input_tensor = torch.randn(8, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        split_to_many(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('split_to_many')

    def test_onnx_bidirectional_dict_list_io(self):
        """Test ONNX export with dict and list inputs producing list and dict outputs.
        
        Mirrors test_export_dict_and_list_bidirectional_io from test_export_situation.py
        """

        @annotate.method(export_with="onnx")
        def bidirectional_io(dict_input: dict, list_input: list):
            # Dict input -> list output
            list_out = [v+1 for v in dict_input.values()]
            # List input -> dict output
            dict_out = {f'item_{i}': v+1 for i, v in enumerate(list_input)}
            return list_out, dict_out

        dict_input = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
            'c': torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        bidirectional_io(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('bidirectional_io')

    def test_onnx_complex_nested_dict_with_extra_input(self):
        """Test ONNX export with nested dict input plus additional tensor input.
        
        Similar to test_export_nnModule_with_large_nested_dict_io from test_export_situation.py
        """

        @annotate.method(export_with="onnx")
        def nested_with_extra(nested_input: dict, extra_tensor: torch.Tensor):
            # Access nested values
            nested = nested_input[0]['nested']
            a = nested['a']
            b = nested['b']
            # Combine with extra tensor
            return [a + extra_tensor, b * extra_tensor]

        nested = [
            {
                'nested': {
                    'a': torch.randn(3, 4, dtype=torch.float32),
                    'b': torch.randn(3, 4, dtype=torch.float32),
                }
            }
        ]
        extra = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        nested_with_extra(nested, extra)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('nested_with_extra')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_torchscript_dict_list_io(self):
        """Test ONNX TorchScript export with dict/list I/O (non-dynamo)."""

        @annotate.method(export_with="onnx-torchscript")
        def torchscript_io(inputs: dict):
            a = inputs['a']
            b = inputs['b']
            return [a + b, a - b]

        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        torchscript_io(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('torchscript_io')

    @pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export")
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_torchscript_nested_inputs(self):
        """Test ONNX TorchScript export with nested list/dict inputs (non-dynamo)."""

        @annotate.method(export_with="onnx-torchscript")
        def torchscript_nested(data: list):
            # Nested list: [[tensor, tensor], [tensor]]
            a = data[0][0]
            b = data[0][1]
            c = data[1][0]
            return a + b + c

        nested_input = [
            [torch.randn(3, 4, dtype=torch.float32), torch.randn(3, 4, dtype=torch.float32)],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        torchscript_nested(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('torchscript_nested')

    def test_onnx_with_nn_module_and_dict_input(self):
        """Test ONNX export with nn.Module method that takes dict input."""

        class ProcessingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            @annotate.method(export_with="onnx")
            def process(self, inputs: dict):
                a = inputs['a']
                b = inputs['b']
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False, atol=1e-3)
        self.verify_all_models_exist('process')

    def test_onnx_with_nn_module_and_list_output(self):
        """Test ONNX export with nn.Module method that returns list output."""

        class SplitModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            @annotate.method(export_with="onnx")
            def split_process(self, x: torch.Tensor):
                return [self.linear1(x), self.linear2(x)]

        module = SplitModule()
        input_tensor = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.split_process(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False, atol=1e-3)
        self.verify_all_models_exist('split_process')
    
class TestTorchBackend(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_torch_trace_backend(self):
        @annotate.method(export_with="jit-trace")
        def funcA(inputA: torch.Tensor):
            return inputA*2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__)

    def test_torch_script_backend(self):
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA*2

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor*2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__)

    def test_exported_program_backend(self):
        @annotate.method(export_with="exported-program")
        def funcA(inputA: torch.Tensor):
            return inputA * 2

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)
        self.verify_single_exported_program_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__)

    def test_pt2_alias_backend(self):
        @annotate.method(export_with="pt2")
        def funcA(inputA: torch.Tensor):
            return inputA * 2

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)
        self.verify_single_exported_program_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__)

        with open(os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")) as f:
            yaml_data = yaml.safe_load(f)
        self.assertEqual(
            yaml_data["models"][funcA.__name__]["parameters"]["backend"], "pt2")

    # -----------------------------------------------------------------
    # Complex math operations
    # -----------------------------------------------------------------

    def test_torch_trace_complex_math(self):
        """Test jit-trace export with complex mathematical operations."""

        @annotate.method(export_with="jit-trace")
        def complex_math(x: torch.Tensor, y: torch.Tensor):
            result = torch.sin(x) + torch.cos(y)
            result = result * torch.exp(-torch.abs(x - y))
            result = torch.clamp(result, -1.0, 1.0)
            result = torch.pow(result, 2)
            return result

        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        complex_math(x, y)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('complex_math')

    def test_torch_script_complex_math(self):
        """Test jit-script export with complex mathematical operations."""

        @annotate.method(export_with="jit")
        def complex_math(x: torch.Tensor, y: torch.Tensor):
            result = torch.sin(x) + torch.cos(y)
            result = result * torch.exp(-torch.abs(x - y))
            result = torch.clamp(result, -1.0, 1.0)
            result = torch.pow(result, 2)
            return result

        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        complex_math(x, y)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('complex_math')

    # -----------------------------------------------------------------
    # Reduction operations
    # -----------------------------------------------------------------

    def test_torch_trace_reduction_operations(self):
        """Test jit-trace export with reduction operations."""

        @annotate.method(export_with="jit-trace")
        def reduction_ops(x: torch.Tensor):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normalized = (x - mean) / (std + 1e-5)
            max_val = normalized.max(dim=1).values
            sum_val = normalized.sum(dim=1)
            return max_val + sum_val

        x = torch.randn(4, 8, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        reduction_ops(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('reduction_ops')

    def test_torch_trace_tensor_method_norm(self):
        """Test jit-trace export with tensor.norm() method calls."""

        @annotate.method(export_with="jit-trace")
        def tensor_method_norm(x: torch.Tensor):
            return x.norm(p=2, dim=-1)

        x = torch.randn(4, 8, dtype=torch.float32)
        expected = x.norm(p=2, dim=-1)

        leapp.start(name=self.TEST_GRAPH_NAME)
        tensor_method_norm(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [x], [expected], 'tensor_method_norm')

    def test_torch_script_reduction_operations(self):
        """Test jit-script export with reduction operations."""

        @annotate.method(export_with="jit")
        def reduction_ops(x: torch.Tensor):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normalized = (x - mean) / (std + 1e-5)
            max_val = normalized.max(dim=1).values
            sum_val = normalized.sum(dim=1)
            return max_val + sum_val

        x = torch.randn(4, 8, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        reduction_ops(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('reduction_ops')

    # -----------------------------------------------------------------
    # Dict / list I/O
    # -----------------------------------------------------------------

    def test_torch_trace_dict_inputs_to_list_outputs(self):
        """Test jit-trace export with dict inputs and list outputs."""

        @annotate.method(export_with="jit-trace")
        def dict_to_list(inputs: dict):
            a = inputs['a']
            b = inputs['b']
            sum_result = a + b
            diff_result = a - b
            return [sum_result, diff_result]

        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        dict_to_list(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('dict_to_list')

    def test_torch_trace_list_inputs_to_dict_outputs(self):
        """Test jit-trace export with list inputs and dict outputs."""

        @annotate.method(export_with="jit-trace")
        def list_to_dict(inputs: list):
            a = inputs[0]
            b = inputs[1]
            return {
                'sum': a + b,
                'product': a * b,
            }

        input_list = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        list_to_dict(input_list)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('list_to_dict')

    def test_torch_trace_nested_dict_input(self):
        """Test jit-trace export with nested dict input."""

        @annotate.method(export_with="jit-trace")
        def process_nested_dict(nested: dict):
            a = nested['level1']['a']
            b = nested['level1']['b']
            c = nested['level2']['c']
            return a + b + c

        nested_input = {
            'level1': {
                'a': torch.randn(3, 4, dtype=torch.float32),
                'b': torch.randn(3, 4, dtype=torch.float32),
            },
            'level2': {
                'c': torch.randn(3, 4, dtype=torch.float32),
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_dict(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_nested_dict')

    def test_torch_trace_nested_list_input(self):
        """Test jit-trace export with nested list input."""

        @annotate.method(export_with="jit-trace")
        def process_nested_list(nested: list):
            a = nested[0][0]
            b = nested[0][1]
            c = nested[1][0]
            return a * b + c

        nested_input = [
            [torch.randn(3, 4, dtype=torch.float32), torch.randn(3, 4, dtype=torch.float32)],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_list(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_nested_list')

    def test_torch_trace_mixed_dict_list_input(self):
        """Test jit-trace export with mixed dict and list inputs."""

        @annotate.method(export_with="jit-trace")
        def process_mixed_inputs(dict_input: dict, list_input: list):
            a = dict_input['a']
            b = dict_input['b']
            c = list_input[0]
            d = list_input[1]
            return a + b, c * d

        dict_input = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_mixed_inputs(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_mixed_inputs')

    def test_torch_trace_nested_list_output(self):
        """Test jit-trace export with nested list output structure."""

        @annotate.method(export_with="jit-trace")
        def create_nested_list_output(x: torch.Tensor):
            return [[x[0:1], x[1:2]], [x[2:3], x[3:4]]]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_nested_list_output(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_nested_list_output')

    def test_torch_trace_dict_of_lists_output(self):
        """Test jit-trace export with dict containing list values as output."""

        @annotate.method(export_with="jit-trace")
        def create_dict_of_lists(x: torch.Tensor):
            return {
                'group_a': [x[0:1], x[1:2]],
                'group_b': [x[2:3], x[3:4]],
            }

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_dict_of_lists(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_dict_of_lists')

    def test_torch_trace_list_of_dicts_output(self):
        """Test jit-trace export with list containing dict values as output."""

        @annotate.method(export_with="jit-trace")
        def create_list_of_dicts(x: torch.Tensor):
            return [
                {'a': x[0:1], 'b': x[1:2]},
                {'a': x[2:3], 'b': x[3:4]},
            ]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_list_of_dicts(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('create_list_of_dicts')

    def test_torch_trace_mixed_return_types(self):
        """Test jit-trace export with mixed return types: tensor, list, and dict."""

        @annotate.method(export_with="jit-trace")
        def mixed_outputs(x: torch.Tensor):
            single = x[0:1]
            list_out = [x[1:2], x[2:3]]
            dict_out = {'a': x[3:4], 'b': x[4:5]}
            return single, list_out, dict_out

        input_tensor = torch.randn(5, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        mixed_outputs(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('mixed_outputs')

    def test_torch_trace_bidirectional_dict_list_io(self):
        """Test jit-trace export with dict and list inputs producing list and dict outputs."""

        @annotate.method(export_with="jit-trace")
        def bidirectional_io(dict_input: dict, list_input: list):
            list_out = [v + 1 for v in dict_input.values()]
            dict_out = {f'item_{i}': v + 1 for i, v in enumerate(list_input)}
            return list_out, dict_out

        dict_input = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
            'c': torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        bidirectional_io(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('bidirectional_io')

    # -----------------------------------------------------------------
    # Deeply nested and stress tests
    # -----------------------------------------------------------------

    def test_torch_trace_deeply_nested_dict_input(self):
        """Test jit-trace export with deeply nested dict input (3+ levels)."""

        @annotate.method(export_with="jit-trace")
        def process_deep_dict(nested: dict):
            a = nested['l1']['l2']['l3']['a']
            b = nested['l1']['l2']['l3']['b']
            return a + b

        deep_input = {
            'l1': {
                'l2': {
                    'l3': {
                        'a': torch.randn(3, 4, dtype=torch.float32),
                        'b': torch.randn(3, 4, dtype=torch.float32),
                    }
                }
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_deep_dict(deep_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_deep_dict')

    def test_torch_trace_many_tensor_inputs(self):
        """Test jit-trace export with many individual tensor inputs."""

        @annotate.method(export_with="jit-trace")
        def sum_many_tensors(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor,
                             t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor,
                             t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor,
                             t9: torch.Tensor):
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

        tensors = [torch.randn(3, 4, dtype=torch.float32) for _ in range(10)]

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_many_tensors(*tensors)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('sum_many_tensors')

    def test_torch_trace_many_dict_tensor_inputs(self):
        """Test jit-trace export with dict containing many tensors."""

        @annotate.method(export_with="jit-trace")
        def sum_dict_tensors(data: dict):
            total = data['t0']
            for i in range(1, 10):
                total = total + data[f't{i}']
            return total

        input_dict = {f't{i}': torch.randn(3, 4, dtype=torch.float32) for i in range(10)}

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_dict_tensors(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('sum_dict_tensors')

    def test_torch_trace_split_tensor_to_many_outputs(self):
        """Test jit-trace export that splits one tensor into many outputs."""

        @annotate.method(export_with="jit-trace")
        def split_to_many(x: torch.Tensor):
            return [x[i:i+1] for i in range(8)]

        input_tensor = torch.randn(8, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        split_to_many(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('split_to_many')

    def test_torch_trace_large_nested_structure(self):
        """Test jit-trace export with large nested dict/list structure."""

        @annotate.method(export_with="jit-trace")
        def process_large_structure(data: dict):
            results = []
            for i in range(4):
                group = data[f'group_{i}']
                for j in range(2):
                    results.append(group[j] * 2.0)
            return torch.stack(results).sum(dim=0)

        large_input = {
            f'group_{i}': [
                torch.randn(3, 4, dtype=torch.float32) for _ in range(2)
            ] for i in range(4)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_large_structure(large_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process_large_structure')

    def test_torch_trace_complex_nested_dict_with_extra_input(self):
        """Test jit-trace export with nested dict input plus additional tensor input."""

        @annotate.method(export_with="jit-trace")
        def nested_with_extra(nested_input: dict, extra_tensor: torch.Tensor):
            nested = nested_input[0]['nested']
            a = nested['a']
            b = nested['b']
            return [a + extra_tensor, b * extra_tensor]

        nested = [
            {
                'nested': {
                    'a': torch.randn(3, 4, dtype=torch.float32),
                    'b': torch.randn(3, 4, dtype=torch.float32),
                }
            }
        ]
        extra = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        nested_with_extra(nested, extra)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('nested_with_extra')

    # -----------------------------------------------------------------
    # Chained nodes
    # -----------------------------------------------------------------

    def test_torch_trace_chained_nodes(self):
        """Test jit-trace export with two chained traced nodes."""

        @annotate.method(export_with="jit-trace")
        def step_a(inputA: torch.Tensor):
            return torch.relu(inputA * 2.0)

        @annotate.method(export_with="jit-trace")
        def step_b(inputB: torch.Tensor):
            return torch.sigmoid(inputB + 1.0)

        input_tensor = torch.randn(4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        intermediate = step_a(input_tensor)
        step_b(intermediate.detach())
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('step_a', 'step_b')

    def test_torch_script_chained_nodes(self):
        """Test jit-script export with two chained scripted nodes."""

        @annotate.method(export_with="jit")
        def step_a(inputA: torch.Tensor):
            return torch.relu(inputA * 2.0)

        @annotate.method(export_with="jit")
        def step_b(inputB: torch.Tensor):
            return torch.sigmoid(inputB + 1.0)

        input_tensor = torch.randn(4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        intermediate = step_a(input_tensor)
        step_b(intermediate.detach())
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('step_a', 'step_b')

    # -----------------------------------------------------------------
    # nn.Module tests
    # -----------------------------------------------------------------

    def test_torch_trace_nn_module_with_dict_input(self):
        """Test jit-trace export with nn.Module method that takes dict input."""

        class ProcessingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            @annotate.method(export_with="jit-trace")
            def process(self, inputs: dict):
                a = inputs['a']
                b = inputs['b']
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process')

    def test_torch_trace_nn_module_with_list_output(self):
        """Test jit-trace export with nn.Module method that returns list output."""

        class SplitModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            @annotate.method(export_with="jit-trace")
            def split_process(self, x: torch.Tensor):
                return [self.linear1(x), self.linear2(x)]

        module = SplitModule()
        input_tensor = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.split_process(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('split_process')

    def test_torch_script_nn_module_with_dict_input(self):
        """Test jit-script export with nn.Module method that takes dict input."""

        class ProcessingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            @annotate.method(export_with="jit")
            def process(self, inputs: dict):
                a = inputs['a']
                b = inputs['b']
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            'a': torch.randn(3, 4, dtype=torch.float32),
            'b': torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('process')

    def test_torch_script_nn_module_with_list_output(self):
        """Test jit-script export with nn.Module method that returns list output."""

        class SplitModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            @annotate.method(export_with="jit")
            def split_process(self, x: torch.Tensor):
                return [self.linear1(x), self.linear2(x)]

        module = SplitModule()
        input_tensor = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.split_process(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('split_process')

    # -----------------------------------------------------------------
    # Multiple outputs / broadcasting
    # -----------------------------------------------------------------

    def test_torch_trace_multiple_outputs(self):
        """Test jit-trace export with multiple tensor outputs (tuple return)."""

        @annotate.method(export_with="jit-trace")
        def multi_output(x: torch.Tensor):
            return torch.relu(x), torch.sigmoid(x), torch.tanh(x)

        input_tensor = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        multi_output(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('multi_output')

    def test_torch_script_multiple_outputs(self):
        """Test jit-script export with multiple tensor outputs (tuple return)."""

        @annotate.method(export_with="jit")
        def multi_output(x: torch.Tensor):
            return torch.relu(x), torch.sigmoid(x), torch.tanh(x)

        input_tensor = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        multi_output(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('multi_output')

    def test_torch_trace_tensor_broadcasting(self):
        """Test jit-trace export with tensor broadcasting operations."""

        @annotate.method(export_with="jit-trace")
        def broadcast_ops(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = inputA * 2.0
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0,
                scaled,
                torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        broadcast_ops(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('broadcast_ops')

    def test_torch_script_tensor_broadcasting(self):
        """Test jit-script export with tensor broadcasting operations."""

        @annotate.method(export_with="jit")
        def broadcast_ops(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = inputA * 2.0
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0,
                scaled,
                torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        broadcast_ops(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist('broadcast_ops')

    # -----------------------------------------------------------------
    # Value verification tests
    # -----------------------------------------------------------------

    def test_torch_trace_value_verification_complex(self):
        """Test jit-trace export and verify output values for a non-trivial function."""

        @annotate.method(export_with="jit-trace")
        def compute(x: torch.Tensor, y: torch.Tensor):
            return torch.relu(x) + torch.sigmoid(y)

        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32)
        y = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], dtype=torch.float32)
        expected = torch.relu(x) + torch.sigmoid(y)

        leapp.start(name=self.TEST_GRAPH_NAME)
        compute(x, y)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [x, y], [expected], 'compute')

    def test_torch_script_value_verification_complex(self):
        """Test jit-script export and verify output values for a non-trivial function."""

        @annotate.method(export_with="jit")
        def compute(x: torch.Tensor, y: torch.Tensor):
            return torch.relu(x) + torch.sigmoid(y)

        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32)
        y = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], dtype=torch.float32)
        expected = torch.relu(x) + torch.sigmoid(y)

        leapp.start(name=self.TEST_GRAPH_NAME)
        compute(x, y)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [x, y], [expected], 'compute')

    def test_torch_trace_value_verification_multi_output(self):
        """Test jit-trace export and verify values for a multi-output function."""

        @annotate.method(export_with="jit-trace")
        def split_and_scale(x: torch.Tensor):
            half1 = x[:2] * 2.0
            half2 = x[2:] * 3.0
            return half1, half2

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        expected_half1 = x[:2] * 2.0
        expected_half2 = x[2:] * 3.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        split_and_scale(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [x], [expected_half1, expected_half2], 'split_and_scale')


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
