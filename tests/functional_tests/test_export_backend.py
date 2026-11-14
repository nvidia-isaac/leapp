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
import errno
import functools
import importlib.abc
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import types
import unittest
import tempfile
from unittest import mock
import torch
import yaml
import onnx
import leapp
from leapp.leapp import _MANAGER as annotate
from leapp import TensorSemantics, InputKindEnum, OutputKindEnum
from leapp.backends.executorch_backend import ExecuTorchExportBackend, ExecuTorchProgram
from leapp.backends.export_backend import SimplifiedONNXProgram
from .base import LEAPPFunctionalTestBase
import pytest

HAS_EXECUTORCH = importlib.util.find_spec("executorch") is not None
NO_TENSORRT = "Torch-TensorRT 2.15.0 or its runtime is not installed"


@functools.cache
def _has_tensorrt():
    """Whether Torch-TensorRT, which builds the engines, and its runtime are
    installed. The runtime wheel does not require Torch-TensorRT."""
    if (
        importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None
        or importlib.util.find_spec("torch_tensorrt") is None
    ):
        return False
    import torch_tensorrt

    # The backend refuses the early 2.15.0 release candidates, which lack it.
    return hasattr(torch_tensorrt, "compile_and_save")


def _execution_plan(program_path):
    """Return the execution plan of an ExecuTorch ``.pte`` file."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(program_path, "rb") as f:
        (plan,) = deserialize_pte_binary(f.read()).program.execution_plan
    return plan


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

        with (
            mock.patch(
                "leapp.backends.export_backend.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            mock.patch("leapp.backends.export_backend._get_logger") as get_logger,
        ):
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
        self.assertEqual(
            input_kwargs["element_type"],
            torch.tensor([], dtype=torch.float32).numpy().dtype.type,
        )
        self.assertEqual(input_kwargs["shape"], (2, 3))
        self.assertIsInstance(input_kwargs["buffer_ptr"], int)
        self.assertGreater(input_kwargs["buffer_ptr"], 0)

        output_kwargs = binding.bind_output.call_args.kwargs
        self.assertEqual(output_kwargs["name"], "output")
        self.assertEqual(output_kwargs["device_type"], "cuda")
        self.assertEqual(output_kwargs["device_id"], 0)
        self.assertEqual(
            output_kwargs["element_type"],
            torch.tensor([], dtype=torch.float32).numpy().dtype.type,
        )
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
        fake_session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        program._session = fake_session
        program._input_names = ["input"]
        program._output_metas = []
        program._active_provider = "CUDAExecutionProvider"

        with (
            mock.patch.object(
                program, "_can_use_cuda_iobinding", return_value=True
            ) as can_use,
            mock.patch.object(
                program, "_run_with_cuda_iobinding", return_value=("fast-path",)
            ) as run_cuda,
            mock.patch.object(program, "_run_with_standard_inference") as run_standard,
        ):
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

        with (
            mock.patch.object(
                program, "_can_use_cuda_iobinding", return_value=False
            ) as can_use,
            mock.patch.object(program, "_run_with_cuda_iobinding") as run_cuda,
            mock.patch.object(
                program, "_run_with_standard_inference", return_value=("cpu-path",)
            ) as run_standard,
        ):
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

    def test_reused_parameter_export(self):
        # Exceed onnxscript's small-initializer deduplication threshold.
        dim = 64
        linear = torch.nn.Linear(dim, dim).eval()

        def forward(x):
            for _ in range(4):
                x = x + torch.tanh(linear(x)) / 4
            return x

        cases = [
            ("onnx", {}),
            ("onnx", {"optimize": False}),
            ("onnx-torchscript", {}),
            ("exported-program", {}),
            ("jit-script", {}),
            ("jit-trace", {}),
        ]
        if HAS_EXECUTORCH:
            cases.append(("executorch", {}))
        for backend, params in cases:
            for global_patching in (False, True):
                with (
                    self.subTest(
                        backend=backend, params=params, global_patching=global_patching
                    ),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    leapp.start("graph", save_path=tmp, global_patching=global_patching)
                    try:
                        with torch.no_grad():
                            for _ in range(2):
                                x = annotate.input_tensors(
                                    "policy", {"x": torch.randn(1, dim)}
                                )
                                annotate.output_tensors(
                                    "policy",
                                    {"y": forward(x)},
                                    export_with=backend,
                                    backend_params=params,
                                )
                    finally:
                        leapp.stop()
                    self.assertEqual(
                        leapp.compile_graph(
                            visualize=False, validate=True, strict=True
                        ),
                        {"policy": True},
                    )

                    if backend.startswith("onnx"):
                        initializers = onnx.load(
                            os.path.join(tmp, "graph", "policy.onnx")
                        ).graph.initializer
                        self.assertEqual(
                            sum(tuple(t.dims) == (dim, dim) for t in initializers), 1
                        )
                        self.assertLess(
                            sum(
                                onnx.numpy_helper.to_array(t).nbytes
                                for t in initializers
                            ),
                            2 * dim * dim * 4,
                        )

                    manager = leapp.InferenceManager(
                        os.path.join(tmp, "graph", "graph.yaml")
                    )
                    with torch.no_grad():
                        for scale in (0.2, 1.0, 3.0):
                            fresh = torch.randn(1, dim) * scale
                            actual = manager.run_policy(
                                {"policy/x": fresh.to(manager.nodes["policy"].device)}
                            )["policy/y"]
                            torch.testing.assert_close(
                                actual.cpu(), forward(fresh), rtol=1e-4, atol=1e-5
                            )

    def test_onnx_backend(self):
        @annotate.method(export_with="onnx")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

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

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_backend_script(self):
        @annotate.method(export_with="onnx-torchscript")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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

                yaml_path = os.path.join(
                    self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml"
                )
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
                    value.name
                    for value in model.graph.input
                    if value.name not in initializer_names
                ]
                onnx_output_names = [value.name for value in model.graph.output]
                self.assertEqual(onnx_input_names, ["joint_pos_in"])
                self.assertEqual(onnx_output_names, ["joint_pos_out"])

    def test_onnx_aliased_outputs_keep_declared_names(self):
        names = ["result", "copy", "identity", "identity_copy"]
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                leapp.start(self.TEST_GRAPH_NAME)
                x = annotate.input_tensors("node", {"x": torch.ones(2)})
                y = x * 2
                annotate.output_tensors(
                    "node",
                    dict(zip(names, (y, y, x, x))),
                    export_with="onnx",
                    backend_params={"optimize": optimize},
                )
                leapp.stop()
                self.assertTrue(leapp.compile_graph(visualize=False)["node"])
                with open(annotate.config_path) as stream:
                    description = yaml.safe_load(stream)["models"]["node"]
                model = onnx.load(os.path.join(self.TEST_GRAPH_NAME, "node.onnx"))
                self.assertEqual([port.name for port in model.graph.input], ["x"])
                self.assertEqual([port.name for port in model.graph.output], names)
                self.assertEqual(
                    [port["name"] for port in description["outputs"]], names
                )
                runtime = leapp.InferenceManager(annotate.config_path)
                value = torch.tensor([3.0, 4.0], device=runtime.nodes["node"].device)
                actual = runtime({"node/x": value})
                for name, expected in zip(names, (value * 2, value * 2, value, value)):
                    torch.testing.assert_close(
                        actual["node/" + name].cpu(), expected.cpu()
                    )

    def test_onnx_connected_node_names_match_artifact_names(self):
        """Connected nodes should keep each node's own ONNX I/O names."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        source_input = annotate.input_tensors(
            "producer",
            {"source_input": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)},
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

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(yaml_path) as f:
            exported = yaml.safe_load(f)

        producer_desc = exported["models"]["producer"]
        producer_yaml_outputs = [desc["name"] for desc in producer_desc["outputs"]]
        self.assertEqual(
            producer_yaml_outputs,
            ["producer_original_output"],
        )
        self.assertEqual(
            exported["pipeline"]["data_flow"],
            {"producer/producer_original_output": ["consumer/consumer_input_name"]},
        )

        model_path = os.path.join(self.TEST_GRAPH_NAME, "producer.onnx")
        model = onnx.load(model_path)
        onnx_output_names = [value.name for value in model.graph.output]
        self.assertEqual(onnx_output_names, producer_yaml_outputs)

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_model"],
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

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_encoder", "traced_decoder"],
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
        self.verify_all_models_exist("encode_decode")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_conv"],
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
        self.verify_all_models_exist("process_image")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_with_complex_tensor_operations(self):
        """Test ONNX export with complex tensor operations (einsum, scatter, gather)."""

        class AttentionModel(torch.nn.Module):
            def __init__(self, dim=8):
                super().__init__()
                self.query = torch.nn.Linear(dim, dim)
                self.key = torch.nn.Linear(dim, dim)
                self.value = torch.nn.Linear(dim, dim)
                self.scale = dim**-0.5

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
            environment_constants=["traced_attn"],
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
        self.verify_all_models_exist("apply_attention")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_a"],
        )
        def step_a(inputA: torch.Tensor):
            return traced_a(inputA)

        @annotate.method(
            export_with="onnx-torchscript",
            backend_params={"prescript": True},
            environment_constants=["traced_b"],
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
        self.verify_all_models_exist("step_a", "step_b")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_scale"],
        )
        def scale_and_broadcast(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = traced_scale(inputA)
            # Broadcasting: mask is (2,) and scaled is (2, 4)
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0, scaled, torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        scale_and_broadcast(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("scale_and_broadcast")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            environment_constants=["traced_multi"],
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
        self.verify_all_models_exist("sample_gaussian")

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
        self.verify_all_models_exist("complex_math")

    def test_onnx_dynamo_with_reduction_operations(self):
        """Test ONNX dynamo export with reduction operations."""

        @annotate.method(export_with="onnx")
        def reduction_ops(x: torch.Tensor):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            normalized = (x - mean) / (std + 1e-5)

            # More reductions - keep shapes compatible
            max_val = normalized.max(dim=1).values  # shape (4,)
            sum_val = normalized.sum(dim=1)  # shape (4,)

            return max_val + sum_val

        x = torch.randn(4, 8, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        reduction_ops(x)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("reduction_ops")

    def test_onnx_dict_inputs_to_list_outputs(self):
        """Test ONNX export with dict-like inputs and list-like outputs.

        This tests that the ONNX exporter correctly handles:
        - Flattening dict inputs into individual tensor inputs
        - Packing multiple outputs into a list
        """

        @annotate.method(export_with="onnx")
        def dict_to_list(inputs: dict):
            a = inputs["a"]
            b = inputs["b"]
            # Return as a list of outputs
            sum_result = a + b
            diff_result = a - b
            return [sum_result, diff_result]

        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        dict_to_list(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("dict_to_list")

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
                "sum": a + b,
                "product": a * b,
            }

        input_list = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        list_to_dict(input_list)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("list_to_dict")

    def test_onnx_nested_dict_input(self):
        """Test ONNX export with deeply nested dict input structure."""

        @annotate.method(export_with="onnx")
        def process_nested_dict(nested: dict):
            # Access deeply nested values
            a = nested["level1"]["a"]
            b = nested["level1"]["b"]
            c = nested["level2"]["c"]
            return a + b + c

        nested_input = {
            "level1": {
                "a": torch.randn(3, 4, dtype=torch.float32),
                "b": torch.randn(3, 4, dtype=torch.float32),
            },
            "level2": {
                "c": torch.randn(3, 4, dtype=torch.float32),
            },
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_dict(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_nested_dict")

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
            [
                torch.randn(3, 4, dtype=torch.float32),
                torch.randn(3, 4, dtype=torch.float32),
            ],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_list(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_nested_list")

    def test_onnx_mixed_dict_list_input(self):
        """Test ONNX export with mixed dict and list inputs."""

        @annotate.method(export_with="onnx")
        def process_mixed_inputs(dict_input: dict, list_input: list):
            a = dict_input["a"]
            b = dict_input["b"]
            c = list_input[0]
            d = list_input[1]
            return a + b, c * d

        dict_input = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_mixed_inputs(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_mixed_inputs")

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
        self.verify_all_models_exist("create_nested_list_output")

    def test_onnx_dict_of_lists_output(self):
        """Test ONNX export with dict containing list values as output."""

        @annotate.method(export_with="onnx")
        def create_dict_of_lists(x: torch.Tensor):
            return {
                "group_a": [x[0:1], x[1:2]],
                "group_b": [x[2:3], x[3:4]],
            }

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_dict_of_lists(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("create_dict_of_lists")

    def test_onnx_list_of_dicts_output(self):
        """Test ONNX export with list containing dict values as output."""

        @annotate.method(export_with="onnx")
        def create_list_of_dicts(x: torch.Tensor):
            return [
                {"a": x[0:1], "b": x[1:2]},
                {"a": x[2:3], "b": x[3:4]},
            ]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_list_of_dicts(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("create_list_of_dicts")

    def test_onnx_mixed_return_types(self):
        """Test ONNX export with mixed return types: tensor, list, and dict."""

        @annotate.method(export_with="onnx")
        def mixed_outputs(x: torch.Tensor):
            single = x[0:1]
            list_out = [x[1:2], x[2:3]]
            dict_out = {"a": x[3:4], "b": x[4:5]}
            return single, list_out, dict_out

        input_tensor = torch.randn(5, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        mixed_outputs(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("mixed_outputs")

    def test_onnx_large_nested_structure(self):
        """Test ONNX export with large nested dict/list structure (stress test)."""

        @annotate.method(export_with="onnx")
        def process_large_structure(data: dict):
            # Process a large nested structure
            results = []
            for i in range(4):
                group = data[f"group_{i}"]
                for j in range(2):
                    results.append(group[j] * 2.0)
            # Return sum of all processed tensors
            return torch.stack(results).sum(dim=0)

        # Create large nested structure: 4 groups, each with 2 tensors
        large_input = {
            f"group_{i}": [torch.randn(3, 4, dtype=torch.float32) for _ in range(2)]
            for i in range(4)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_large_structure(large_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_large_structure")

    def test_onnx_deeply_nested_dict_input(self):
        """Test ONNX export with deeply nested dict input (3+ levels)."""

        @annotate.method(export_with="onnx")
        def process_deep_dict(nested: dict):
            # Access 3-level deep nested values
            a = nested["l1"]["l2"]["l3"]["a"]
            b = nested["l1"]["l2"]["l3"]["b"]
            return a + b

        deep_input = {
            "l1": {
                "l2": {
                    "l3": {
                        "a": torch.randn(3, 4, dtype=torch.float32),
                        "b": torch.randn(3, 4, dtype=torch.float32),
                    }
                }
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_deep_dict(deep_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_deep_dict")

    def test_onnx_many_tensor_inputs(self):
        """Test ONNX export with many individual tensor inputs (stress test)."""

        @annotate.method(export_with="onnx")
        def sum_many_tensors(
            t0: torch.Tensor,
            t1: torch.Tensor,
            t2: torch.Tensor,
            t3: torch.Tensor,
            t4: torch.Tensor,
            t5: torch.Tensor,
            t6: torch.Tensor,
            t7: torch.Tensor,
            t8: torch.Tensor,
            t9: torch.Tensor,
        ):
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

        tensors = [torch.randn(3, 4, dtype=torch.float32) for _ in range(10)]

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_many_tensors(*tensors)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("sum_many_tensors")

    def test_onnx_many_dict_tensor_inputs(self):
        """Test ONNX export with dict containing many tensors."""

        @annotate.method(export_with="onnx")
        def sum_dict_tensors(data: dict):
            total = data["t0"]
            for i in range(1, 10):
                total = total + data[f"t{i}"]
            return total

        input_dict = {
            f"t{i}": torch.randn(3, 4, dtype=torch.float32) for i in range(10)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_dict_tensors(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("sum_dict_tensors")

    def test_onnx_split_tensor_to_many_outputs(self):
        """Test ONNX export that splits one tensor into many outputs."""

        @annotate.method(export_with="onnx")
        def split_to_many(x: torch.Tensor):
            # Split tensor into 8 parts
            return [x[i : i + 1] for i in range(8)]

        input_tensor = torch.randn(8, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        split_to_many(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("split_to_many")

    def test_onnx_bidirectional_dict_list_io(self):
        """Test ONNX export with dict and list inputs producing list and dict outputs.

        Mirrors test_export_dict_and_list_bidirectional_io from test_export_situation.py
        """

        @annotate.method(export_with="onnx")
        def bidirectional_io(dict_input: dict, list_input: list):
            # Dict input -> list output
            list_out = [v + 1 for v in dict_input.values()]
            # List input -> dict output
            dict_out = {f"item_{i}": v + 1 for i, v in enumerate(list_input)}
            return list_out, dict_out

        dict_input = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
            "c": torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        bidirectional_io(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("bidirectional_io")

    def test_onnx_complex_nested_dict_with_extra_input(self):
        """Test ONNX export with nested dict input plus additional tensor input.

        Similar to test_export_nnModule_with_large_nested_dict_io from test_export_situation.py
        """

        @annotate.method(export_with="onnx")
        def nested_with_extra(nested_input: dict, extra_tensor: torch.Tensor):
            # Access nested values
            nested = nested_input[0]["nested"]
            a = nested["a"]
            b = nested["b"]
            # Combine with extra tensor
            return [a + extra_tensor, b * extra_tensor]

        nested = [
            {
                "nested": {
                    "a": torch.randn(3, 4, dtype=torch.float32),
                    "b": torch.randn(3, 4, dtype=torch.float32),
                }
            }
        ]
        extra = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        nested_with_extra(nested, extra)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("nested_with_extra")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
    @pytest.mark.filterwarnings("ignore:The feature will be removed")
    def test_onnx_torchscript_dict_list_io(self):
        """Test ONNX TorchScript export with dict/list I/O (non-dynamo)."""

        @annotate.method(export_with="onnx-torchscript")
        def torchscript_io(inputs: dict):
            a = inputs["a"]
            b = inputs["b"]
            return [a + b, a - b]

        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        torchscript_io(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("torchscript_io")

    @pytest.mark.filterwarnings(
        "ignore:You are using the legacy TorchScript-based ONNX export"
    )
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
            [
                torch.randn(3, 4, dtype=torch.float32),
                torch.randn(3, 4, dtype=torch.float32),
            ],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        torchscript_nested(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("torchscript_nested")

    def test_onnx_with_nn_module_and_dict_input(self):
        """Test ONNX export with nn.Module method that takes dict input."""

        class ProcessingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            @annotate.method(export_with="onnx")
            def process(self, inputs: dict):
                a = inputs["a"]
                b = inputs["b"]
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False, atol=1e-3)
        self.verify_all_models_exist("process")

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
        self.verify_all_models_exist("split_process")


class TestTorchBackend(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_torch_trace_backend(self):
        @annotate.method(export_with="jit-trace")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__
        )

    def test_torch_script_backend(self):
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA * 2

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        expected_output = input_tensor * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcA(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_single_torchscript_model_expected_value(
            [input_tensor], [expected_output], funcA.__name__
        )

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
            [input_tensor], [expected_output], funcA.__name__
        )

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
            [input_tensor], [expected_output], funcA.__name__
        )

        with open(
            os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        ) as f:
            yaml_data = yaml.safe_load(f)
        self.assertEqual(
            yaml_data["models"][funcA.__name__]["parameters"]["backend"], "pt2"
        )

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend(self):
        """Covers the ExecuTorch delegates, which lower and run the graph outside
        PyTorch, so validation and the InferenceManager run the ``.pte`` file."""
        from executorch.runtime import Runtime

        # Wide enough that TF32 rounding would fail validation.
        generator = torch.Generator().manual_seed(0)
        weight = (torch.rand(512, 512, generator=generator) - 0.5) / 16
        # A loaded node reports the GPU when there is one, as a TorchScript node does.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cases = []
        if torch.cuda.is_available():
            cases += [("cuda", True), ("cuda", False)]
            if _has_tensorrt():
                cases += [("tensorrt", True), ("tensorrt", False)]
        # Last, so the folder still holds the .ptd a CUDA case wrote.
        cases.append(("xnnpack", True))
        for delegate, device_io in cases:
            with self.subTest(delegate=delegate, device_io=device_io):
                backend_params = {"delegate": delegate}
                if not device_io:
                    backend_params["device_io"] = False

                @annotate.method(
                    export_with="executorch", backend_params=backend_params
                )
                def funcA(inputA: torch.Tensor):
                    return inputA @ weight

                leapp.start(name=self.TEST_GRAPH_NAME)
                # Transposed, so the program must not depend on the traced strides.
                funcA(torch.linspace(-1, 1, 64 * 512).reshape(512, 64).t())
                leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {funcA.__name__: True},
                )

                with open(
                    os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
                ) as f:
                    parameters = yaml.safe_load(f)["models"][funcA.__name__][
                        "parameters"
                    ]
                self.assertEqual(parameters["backend"], "pte")
                self.assertEqual(parameters["delegate"], delegate)
                self.assertEqual(
                    "torch_tensorrt_version" in parameters, delegate == "tensorrt"
                )
                # TensorRT takes this whole graph, so only the CUDA backend
                # writes a .ptd, and an earlier case's file must not be recorded.
                self.assertEqual("data_path" in parameters, delegate == "cuda")
                program_path = os.path.join(
                    self.TEST_GRAPH_NAME, f"{funcA.__name__}.pte"
                )
                plan = _execution_plan(program_path)
                backend_id = {
                    "xnnpack": "XnnpackBackend",
                    "cuda": "CudaBackend",
                    "tensorrt": "TensorRTBackend",
                }
                self.assertEqual([d.id for d in plan.delegates], [backend_id[delegate]])
                # The delegate runs the whole graph; only the copies between host
                # and GPU memory run outside it.
                copies = {"et_copy::_h2d_copy", "et_copy::_d2h_copy"}
                self.assertEqual(
                    {op.name for op in plan.operators},
                    copies if delegate != "xnnpack" and not device_io else set(),
                )
                # The program's own output, before the copy to the inputs' device.
                data_path = parameters.get("data_path")
                if data_path is not None:
                    data_path = os.path.join(self.TEST_GRAPH_NAME, data_path)
                method = (
                    Runtime.get()
                    .load_program(program_path, data_path=data_path)
                    .load_method("forward")
                )
                self.assertEqual(
                    torch.device(method.metadata.output_tensor_meta(0).device()).type,
                    "cpu" if delegate == "xnnpack" or not device_io else "cuda",
                )

                manager = leapp.InferenceManager(
                    os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
                )
                self.assertEqual(
                    torch.device(manager.nodes[funcA.__name__].device).type, device
                )
                (output_key,) = manager.outputs
                # Two calls, so a program that returns its first result fails.
                for _ in range(2):
                    # Sliced after the move, which would make it contiguous.
                    fresh = torch.rand(64, 1024, generator=generator).to(device)[:, ::2]
                    actual = manager.run_policy({f"{funcA.__name__}/inputA": fresh})[
                        output_key
                    ]
                    self.assertEqual(actual.device.type, device)
                    torch.testing.assert_close(
                        actual.cpu(), fresh.cpu() @ weight, rtol=1e-4, atol=1e-5
                    )

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend_constants(self):
        """A legacy node is exported from its Python function, so a value it
        builds from constants alone reaches export. XNNPACK cannot run arange,
        so the program runs whole in the delegate only because export folds
        it."""

        @annotate._method(export_with="executorch")
        def legacy_constants(inputL):
            return inputL + torch.arange(3, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        legacy_constants(torch.ones(3))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {legacy_constants.__name__: True},
        )
        plan = _execution_plan(
            os.path.join(self.TEST_GRAPH_NAME, f"{legacy_constants.__name__}.pte")
        )
        self.assertEqual([op.name for op in plan.operators], [])

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend_written_buffers(self):
        """ExecuTorch leaves a buffer the program writes uninitialized unless told to
        keep its value, so these nodes would return zeros where the buffers held
        20 and 30. One declares its buffer, the other's module owns two. On a GPU
        machine, CPU inputs still give CPU outputs."""

        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("count", torch.tensor([10.0, 20.0, 30.0]))
                # torch.export keeps a persistent=False buffer's value in constants.
                self.register_buffer(
                    "total", torch.tensor([10.0, 20.0, 30.0]), persistent=False
                )

        class Overwriter:
            def __init__(self):
                self.values = torch.tensor([10.0, 20.0, 30.0])
                self.counter = Counter()

            @annotate._method(
                export_with="executorch", register_buffers=["self.values"]
            )
            def legacy_declared(self, inputL):
                self.values[0] = inputL[0]
                return self.values * 1.0

            @annotate._method(export_with="executorch")
            def legacy_module(self, inputM):
                self.counter.count[0] = inputM[0]
                self.counter.total[0] = inputM[0]
                return (self.counter.count + self.counter.total) / 2

        overwriter = Overwriter()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for value in (1.0, 2.0):
            overwriter.legacy_declared(torch.full((3,), value))
            overwriter.legacy_module(torch.full((3,), value))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"legacy_declared": True, "legacy_module": True},
        )
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        seven = torch.full((3,), 7.0)
        outputs = leapp.InferenceManager(yaml_path).run_policy(
            {"legacy_declared/inputL": seven, "legacy_module/inputM": seven}
        )
        self.assertEqual(len(outputs), 2)
        for actual in outputs.values():
            torch.testing.assert_close(actual, torch.tensor([7.0, 20.0, 30.0]))

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_cuda_written_buffer(self):
        """A CUDA node that returns the buffer it writes leaves the buffer outside
        the delegate, where it must still start from 20 and 30, not zeros."""

        class Overwriter:
            def __init__(self):
                self.values = torch.tensor([10.0, 20.0, 30.0])

            @annotate._method(
                export_with="executorch",
                register_buffers=["self.values"],
                # With device_io, ExecuTorch cannot place a written buffer the
                # program returns on the GPU.
                backend_params={"delegate": "cuda", "device_io": False},
            )
            def legacy_cuda(self, inputC):
                self.values[0] = inputC[0]
                return self.values

        overwriter = Overwriter()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for value in (1.0, 2.0):
            overwriter.legacy_cuda(torch.full((3,), value))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"legacy_cuda": True}
        )
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        (actual,) = (
            leapp.InferenceManager(yaml_path)
            .run_policy({"legacy_cuda/inputC": torch.full((3,), 7.0)})
            .values()
        )
        torch.testing.assert_close(actual.cpu(), torch.tensor([7.0, 20.0, 30.0]))

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend_no_inputs(self):
        """A node whose output reads only a buffer loses its input when the graph
        is pruned. Its output must still reach a TorchScript node on the node's
        device: the GPU when there is one, else the CPU. The CUDA case also shows
        that a CUDA program with no inputs runs."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = torch.tensor([1.0, 2.0, 3.0], device=device)
        delegates = ("xnnpack", "cuda") if device == "cuda" else ("xnnpack",)
        for delegate in delegates:
            with self.subTest(delegate=delegate):
                leapp.start(name=self.TEST_GRAPH_NAME)
                annotate.input_tensors("funcK", {"x": torch.ones(3, device=device)})
                buffer = annotate.register_buffer(
                    "funcK", {"buffer": torch.tensor([4.0, 5.0, 6.0], device=device)}
                )
                y = annotate.output_tensors(
                    "funcK",
                    {"y": buffer * 2},
                    export_with="executorch",
                    backend_params={"delegate": delegate},
                )
                c = annotate.input_tensors("funcL", {"c": y})
                annotate.output_tensors("funcL", {"out": c * weight}, export_with="jit")
                leapp.stop()
                leapp.compile_graph(visualize=False)
                outputs = leapp.InferenceManager(yaml_path).run_policy({})
                torch.testing.assert_close(
                    outputs["funcL/out"],
                    torch.tensor([8.0, 20.0, 36.0], device=device),
                )

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_devices(self):
        """Covers nodes on different devices: a GPU node that returns an input
        unchanged, a GPU node traced on the GPU, two GPU nodes with their own
        weights files in one graph, an XNNPACK node on the CPU feeding a
        TorchScript node, and a GPU default device during lowering."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        delegates = ["cuda", "tensorrt"] if _has_tensorrt() else ["cuda"]
        for delegate in delegates:
            with self.subTest(delegate=delegate):
                leapp.start(name=self.TEST_GRAPH_NAME)
                a, b = annotate.input_tensors(
                    "funcE", {"a": torch.ones(3), "b": torch.ones(3)}
                )
                annotate.output_tensors(
                    "funcE",
                    {"same": a, "doubled": b * 2},
                    export_with="executorch",
                    backend_params={"delegate": delegate},
                )
                leapp.stop()
                leapp.compile_graph(visualize=False)
                a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
                b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
                outputs = leapp.InferenceManager(yaml_path).run_policy(
                    {"funcE/a": a, "funcE/b": b}
                )
                torch.testing.assert_close(outputs["funcE/same"], a)
                torch.testing.assert_close(outputs["funcE/doubled"], b * 2)

        # Traced on the GPU, with a cast that torch.export records with its device.
        @annotate.method(export_with="executorch", backend_params={"delegate": "cuda"})
        def funcH(inputH: torch.Tensor):
            return inputH.float() / 255

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcH(torch.arange(3, dtype=torch.uint8, device="cuda"))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {funcH.__name__: True},
        )

        # Different weights, so a node reading the other's .ptd gives wrong values.
        weight_k = torch.rand(3, 3, device="cuda")
        weight_l = torch.rand(3, 3, device="cuda")

        @annotate.method(export_with="executorch", backend_params={"delegate": "cuda"})
        def funcK(inputK: torch.Tensor):
            return inputK @ weight_k

        @annotate.method(export_with="executorch", backend_params={"delegate": "cuda"})
        def funcL(inputL: torch.Tensor):
            return inputL @ weight_l

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcL(funcK(torch.ones(2, 3, device="cuda")))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {funcK.__name__: True, funcL.__name__: True},
        )
        with open(yaml_path) as f:
            models = yaml.safe_load(f)["models"]
        self.assertEqual(
            {models[name]["parameters"]["data_path"] for name in ("funcK", "funcL")},
            {"funcK.ptd", "funcL.ptd"},
        )
        value = torch.rand(2, 3, device="cuda")
        (actual,) = (
            leapp.InferenceManager(yaml_path)
            .run_policy({f"{funcK.__name__}/inputK": value})
            .values()
        )
        torch.testing.assert_close(
            actual, value @ weight_k @ weight_l, rtol=1e-4, atol=1e-5
        )

        weight = torch.tensor([1.0, 2.0, 3.0])

        @annotate.method(export_with="executorch")
        def funcF(inputF: torch.Tensor):
            return inputF + 1

        @annotate.method(export_with="jit")
        def funcG(inputG: torch.Tensor):
            return inputG * weight

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcG(funcF(torch.ones(3)))
        leapp.stop()
        leapp.compile_graph(visualize=False)
        manager = leapp.InferenceManager(yaml_path)
        (actual,) = manager.run_policy(
            {f"{funcF.__name__}/inputF": torch.ones(3, device="cuda")}
        ).values()
        torch.testing.assert_close(actual.cpu(), 2 * weight)
        # Mock inputs live on the first node's device, which the next node shares.
        mock_input = manager.get_mock_input()
        (actual,) = manager.run_policy(mock_input).values()
        (value,) = mock_input.values()
        torch.testing.assert_close(actual.cpu(), (value.cpu() + 1) * weight)

        # The caller's default device must not reach either lowering.
        @annotate.method(export_with="executorch", backend_params={"delegate": "cuda"})
        def funcM(inputM: torch.Tensor):
            return inputM * 2 + 1

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcM(funcF(torch.ones(3)))
        leapp.stop()
        run_lower = ExecuTorchExportBackend._lower
        lowered_on = []

        def lower(backend, *args):
            lowered_on.append(torch.get_default_device())
            return run_lower(backend, *args)

        torch.set_default_device("cuda")
        try:
            with mock.patch.object(ExecuTorchExportBackend, "_lower", lower):
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {funcF.__name__: True, funcM.__name__: True},
                )
        finally:
            torch.set_default_device(None)
        self.assertEqual(lowered_on, [torch.device("cpu")] * 2)

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend_errors(self):
        """Covers the error paths, the checksum of the ``.pte`` file, and that
        validation runs the program."""
        from executorch.runtime import Runtime

        with self.assertRaises(ValueError):
            ExecuTorchExportBackend(None, {"delegate": "unknown"})
        with self.assertRaises(TypeError):
            ExecuTorchExportBackend(None, {"delegate": "cuda", "device_io": "False"})
        with self.assertRaisesRegex(TypeError, "tensorrt_options must be a dict"):
            ExecuTorchExportBackend(
                None, {"delegate": "tensorrt", "tensorrt_options": ["fp16"]}
            )
        for key in (
            "exported_program",
            "inputs",
            "arg_inputs",
            "kwarg_inputs",
            "device",
            "dryrun",
        ):
            with self.assertRaisesRegex(ValueError, f"\\['{key}'\\] cannot be set"):
                ExecuTorchExportBackend(
                    None, {"delegate": "tensorrt", "tensorrt_options": {key: None}}
                )
        with self.assertRaisesRegex(ValueError, "release_weights_after_build"):
            ExecuTorchExportBackend(
                None,
                {
                    "delegate": "tensorrt",
                    "tensorrt_options": {"release_weights_after_build": True},
                },
            )
        ExecuTorchExportBackend(
            None, {"delegate": "tensorrt", "tensorrt_options": {"min_block_size": 3}}
        )
        for delegate in ("xnnpack", "cuda"):
            with self.assertRaisesRegex(ValueError, 'only to delegate "tensorrt"'):
                ExecuTorchExportBackend(
                    None,
                    {"delegate": delegate, "tensorrt_options": {"min_blok_size": 3}},
                )
        node = mock.Mock()
        node.name = "never_compiled"
        with self.assertRaises(RuntimeError):
            ExecuTorchExportBackend(node).save(self.TEST_GRAPH_NAME)

        @annotate.method(export_with="executorch")
        def funcB(inputB: torch.Tensor):
            return inputB * 2

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcB(torch.ones(3))
        leapp.stop()
        leapp.compile_graph(visualize=False)
        with open(
            os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        ) as f:
            sha256sum = yaml.safe_load(f)["models"][funcB.__name__]["parameters"][
                "sha256sum"
            ]
        model = os.path.join(self.TEST_GRAPH_NAME, f"{funcB.__name__}.pte")
        with self.assertRaises(ValueError):
            ExecuTorchExportBackend(None).load(model, "0" * 64)
        with (
            mock.patch.dict("sys.modules", {"executorch.runtime": None}),
            self.assertRaisesRegex(ImportError, "pip install"),
        ):
            ExecuTorchExportBackend(None).load(model, sha256sum)
        # Only a TensorRT program needs Torch-TensorRT's runtime.
        with mock.patch.dict(
            "sys.modules", {"torch_tensorrt_executorch_runtime": None}
        ):
            ExecuTorchExportBackend(None).load(model, sha256sum)
        backend = ExecuTorchExportBackend(None)
        with mock.patch.object(
            Runtime, "load_program", autospec=True, side_effect=Runtime.load_program
        ) as load_program:
            backend.load(model, sha256sum)
        # So XNNPACK packs a weight once, not once per use.
        self.assertEqual(
            load_program.call_args.kwargs["backend_options"],
            {"XnnpackBackend": {"weight_cache_enabled": True}},
        )
        with self.assertRaisesRegex(ValueError, "Expected 1 inputs, got 2"):
            backend.compiled_model(torch.ones(3), torch.ones(3))

        # A failed lowering leaves no temporary folder behind.
        made = []
        mkdtemp = tempfile.mkdtemp

        def recorded_mkdtemp(*args, **kwargs):
            made.append(mkdtemp(*args, **kwargs))
            return made[-1]

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcB(torch.ones(3))
        leapp.stop()
        with (
            mock.patch("tempfile.mkdtemp", recorded_mkdtemp),
            mock.patch.object(
                ExecuTorchExportBackend, "_lower", side_effect=RuntimeError("lowering")
            ),
            self.assertRaisesRegex(Exception, "lowering"),
        ):
            leapp.compile_graph(visualize=False)
        self.assertTrue(made)
        self.assertEqual([path for path in made if os.path.exists(path)], [])

        # Validation runs the program, so a program that returns other values fails it.
        run = ExecuTorchProgram.__call__

        def off_by_one(program, *args):
            return tuple(output + 1 for output in run(program, *args))

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcB(torch.ones(3))
        leapp.stop()
        with (
            mock.patch.object(ExecuTorchProgram, "__call__", off_by_one),
            self.assertRaisesRegex(Exception, "validation failed"),
        ):
            leapp.compile_graph(visualize=False, validate=True)

        for delegate in ("cuda", "tensorrt"):

            @annotate.method(
                export_with="executorch", backend_params={"delegate": delegate}
            )
            def funcI(inputI: torch.Tensor):
                return inputI * 2

            with self.subTest(delegate=delegate):
                leapp.start(name=self.TEST_GRAPH_NAME)
                funcI(torch.ones(3))
                leapp.stop()
                with (
                    mock.patch("torch.cuda.is_available", return_value=False),
                    self.assertRaisesRegex(
                        RuntimeError, f'"{delegate}" needs a CUDA GPU'
                    ),
                ):
                    leapp.compile_graph(visualize=False)

        # With 0 rows the input has no elements, with 1 row only the output.
        for delegate in ("cuda", "tensorrt"):
            for device_io in (True, False):

                @annotate.method(
                    export_with="executorch",
                    backend_params={"delegate": delegate, "device_io": device_io},
                )
                def funcK(inputK: torch.Tensor):
                    return inputK[1:] * 2

                for rows in (0, 1):
                    with self.subTest(
                        delegate=delegate, rows=rows, device_io=device_io
                    ):
                        leapp.start(name=self.TEST_GRAPH_NAME)
                        funcK(torch.ones(rows, 3))
                        leapp.stop()
                        with (
                            mock.patch("torch.cuda.is_available", return_value=True),
                            self.assertRaisesRegex(
                                RuntimeError,
                                f'elements, which ExecuTorch delegate "{delegate}"',
                            ),
                        ):
                            leapp.compile_graph(visualize=False)

        # An installed runtime or Torch-TensorRT that lacks a dependency keeps its
        # own error.
        class Broken(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == broken:
                    raise ModuleNotFoundError(
                        "No module named 'tensorrt_libs'", name="tensorrt_libs"
                    )

        @annotate.method(
            export_with="executorch", backend_params={"delegate": "tensorrt"}
        )
        def funcR(inputR: torch.Tensor):
            return inputR * 2

        runtime = types.ModuleType("runtime")
        for broken, modules in (
            ("torch_tensorrt_executorch_runtime", {}),
            ("torch_tensorrt", {"torch_tensorrt_executorch_runtime": runtime}),
        ):
            with self.subTest(broken=broken):
                leapp.start(name=self.TEST_GRAPH_NAME)
                funcR(torch.ones(3))
                leapp.stop()
                with (
                    mock.patch("torch.cuda.is_available", return_value=True),
                    mock.patch.dict("sys.modules", modules),
                    mock.patch.object(sys, "meta_path", [Broken(), *sys.meta_path]),
                    self.assertRaisesRegex(ImportError, "tensorrt_libs"),
                ):
                    sys.modules.pop(broken, None)
                    leapp.compile_graph(visualize=False)

        # Without the runtime, or without a Torch-TensorRT that can build the
        # program, the export stops with an install hint before anything is built.
        no_runtime = {"torch_tensorrt_executorch_runtime": None}
        no_compiler = {
            "torch_tensorrt_executorch_runtime": runtime,
            "torch_tensorrt": None,
        }
        # An early release candidate has torch_tensorrt.dynamo but no
        # compile_and_save.
        early = types.ModuleType("torch_tensorrt")
        early.dynamo = types.ModuleType("torch_tensorrt.dynamo")
        old_compiler = {
            "torch_tensorrt_executorch_runtime": runtime,
            "torch_tensorrt": early,
        }
        install = r'pip install "torch_tensorrt\[executorch\]>=2\.15\.0"'
        for modules, message in (
            (no_runtime, f"its ExecuTorch runtime: {install}"),
            (no_compiler, f"Torch-TensorRT 2.15.0 or newer: {install}"),
            (old_compiler, f"release candidate: pip uninstall .*, then {install}"),
        ):
            with self.subTest(message=message, compiler=modules.get("torch_tensorrt")):
                leapp.start(name=self.TEST_GRAPH_NAME)
                funcR(torch.ones(3))
                leapp.stop()
                with (
                    mock.patch("torch.cuda.is_available", return_value=True),
                    mock.patch.dict("sys.modules", modules),
                    mock.patch("torch.export.export") as export,
                    self.assertRaisesRegex(ImportError, message),
                ):
                    leapp.compile_graph(visualize=False)
                export.assert_not_called()

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_tensorrt_errors(self):
        """Covers the TensorRT error paths and the timing cache, which need a GPU
        and Torch-TensorRT."""
        if not _has_tensorrt():
            self.skipTest(NO_TENSORRT)

        @annotate.method(
            export_with="executorch", backend_params={"delegate": "tensorrt"}
        )
        def funcT(inputT: torch.Tensor):
            return inputT * 2 + 1

        # TensorRT has no float64.
        leapp.start(name=self.TEST_GRAPH_NAME)
        funcT(torch.ones(3, dtype=torch.float64, device="cuda"))
        leapp.stop()
        with self.assertRaisesRegex(ValueError, "float64"):
            leapp.compile_graph(visualize=False)

        # A saved TensorRT program cannot load without the runtime either.
        no_runtime = {"torch_tensorrt_executorch_runtime": None}
        leapp.start(name=self.TEST_GRAPH_NAME)
        funcT(torch.ones(3, device="cuda"))
        leapp.stop()
        leapp.compile_graph(visualize=False)
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with (
            mock.patch.dict("sys.modules", no_runtime),
            self.assertRaisesRegex(ImportError, "pip install"),
        ):
            leapp.InferenceManager(yaml_path)

        # A buffer or parameter the node writes, and a held value it returns
        # unchanged, are refused before the engines are built.
        class Steps(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.steps = torch.nn.Parameter(
                    torch.tensor([10.0, 20.0, 30.0], device="cuda"),
                    requires_grad=False,
                )

        class Overwriter:
            def __init__(self):
                self.values = torch.tensor([10.0, 20.0, 30.0], device="cuda")
                self.module = Steps()

            @annotate._method(
                export_with="executorch",
                register_buffers=["self.values"],
                backend_params={"delegate": "tensorrt", "device_io": False},
            )
            def legacy_buffer(self, inputW):
                self.values[0] = inputW[0]
                return self.values * 2.0

            @annotate._method(
                export_with="executorch", backend_params={"delegate": "tensorrt"}
            )
            def legacy_parameter(self, inputW):
                self.module.steps.add_(1)
                return inputW * self.module.steps

            @annotate._method(
                export_with="executorch",
                register_buffers=["self.values"],
                backend_params={"delegate": "tensorrt"},
            )
            def legacy_returned(self, inputW):
                return inputW * 2, self.values

            # Torch-TensorRT removes the detach, so this returns the buffer too.
            @annotate._method(
                export_with="executorch",
                register_buffers=["self.values"],
                backend_params={"delegate": "tensorrt"},
            )
            def legacy_detached(self, inputW):
                return inputW * 2, self.values.detach()

            @annotate._method(
                export_with="executorch", backend_params={"delegate": "tensorrt"}
            )
            def legacy_returned_parameter(self, inputW):
                return inputW * 2, self.module.steps

            @annotate._method(
                export_with="executorch", backend_params={"delegate": "tensorrt"}
            )
            def legacy_constant(self, inputW):
                return inputW * 2, table.detach()

        table = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        import torch_tensorrt

        for name, message in (
            ("legacy_buffer", 'writes .* delegate "cuda"'),
            ("legacy_parameter", "writes .* with another backend\\."),
            ("legacy_returned", "returns \\['values'\\] unchanged"),
            ("legacy_detached", "returns \\['values'\\] unchanged"),
            ("legacy_returned_parameter", "returns \\['module.steps'\\] unchanged"),
            ("legacy_constant", "returns \\['lifted_tensor_0'\\] unchanged"),
        ):
            with self.subTest(node=name):
                leapp.start(name=self.TEST_GRAPH_NAME)
                getattr(Overwriter(), name)(torch.ones(3, device="cuda"))
                leapp.stop()
                with (
                    mock.patch.object(
                        torch_tensorrt, "compile_and_save"
                    ) as compile_trt,
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    leapp.compile_graph(visualize=False)
                compile_trt.assert_not_called()

        with self.subTest(node="static output"):
            leapp.start(name=self.TEST_GRAPH_NAME)
            inputs = annotate.input_tensors(
                "static", {"x": torch.ones(3, device="cuda")}
            )
            annotate.output_tensors(
                "static",
                {"y": inputs * 2},
                static_outputs={"bias": torch.tensor([4.0, 5.0, 6.0], device="cuda")},
                export_with="executorch",
                backend_params={"delegate": "tensorrt"},
            )
            leapp.stop()
            with (
                mock.patch.object(torch_tensorrt, "compile_and_save") as compile_trt,
                self.assertRaisesRegex(RuntimeError, "returns .* unchanged"),
            ):
                leapp.compile_graph(visualize=False)
            compile_trt.assert_not_called()

        # A name the CUDA backend refuses for its data file fails before the
        # engines are built.
        @annotate.method(
            export_with="executorch", backend_params={"delegate": "tensorrt"}
        )
        def aux(inputA: torch.Tensor):
            return inputA * 2 + 1

        leapp.start(name=self.TEST_GRAPH_NAME)
        aux(torch.ones(3, device="cuda"))
        leapp.stop()
        with (
            mock.patch.object(torch_tensorrt, "compile_and_save") as compile_trt,
            self.assertRaisesRegex(ValueError, "Invalid data name"),
        ):
            leapp.compile_graph(visualize=False)
        compile_trt.assert_not_called()

        # A misspelled setting is refused before the engines are built.
        @annotate.method(
            export_with="executorch",
            backend_params={
                "delegate": "tensorrt",
                "tensorrt_options": {"min_blok_size": 3, "kwargs": {}},
            },
        )
        def funcU(inputU: torch.Tensor):
            return inputU * 2 + 1

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcU(torch.ones(3, device="cuda"))
        leapp.stop()
        with (
            mock.patch.object(torch_tensorrt, "compile_and_save") as compile_trt,
            self.assertRaisesRegex(ValueError, "\\['kwargs', 'min_blok_size'\\]"),
        ):
            leapp.compile_graph(visualize=False)
        compile_trt.assert_not_called()

        # The timing cache is private to the build and removed after it, whether
        # the build fails or not. A given path is left alone. A build that returns
        # without a program fails later, at the load.
        builds = []

        def build(exported_program, program_path, **kwargs):
            folder = os.path.dirname(kwargs["timing_cache_path"])
            # save() copies the program's folder, so the cache must be elsewhere.
            if folder == os.path.dirname(program_path):
                raise AssertionError("timing cache in the program folder")
            builds.append((folder, os.stat(folder).st_mode & 0o777))
            if fails:
                raise RuntimeError("stopped before the build")

        given = tempfile.TemporaryDirectory()
        self.addCleanup(given.cleanup)
        given_cache = os.path.join(given.name, "timing_cache.bin")
        with open(given_cache, "wb") as f:
            f.write(b"timings")
        for options, fails in (
            ({}, True),
            ({}, False),
            ({"timing_cache_path": given_cache}, True),
            ({"timing_cache_path": given_cache}, False),
        ):
            with self.subTest(options=options, fails=fails):

                @annotate.method(
                    export_with="executorch",
                    backend_params={
                        "delegate": "tensorrt",
                        "tensorrt_options": options,
                    },
                )
                def funcV(inputV: torch.Tensor):
                    return inputV * 2 + 1

                leapp.start(name=self.TEST_GRAPH_NAME)
                funcV(torch.ones(3, device="cuda"))
                leapp.stop()
                with (
                    mock.patch.object(torch_tensorrt, "compile_and_save", build),
                    self.assertRaisesRegex(
                        Exception, "stopped before the build" if fails else "funcV.pte"
                    ),
                ):
                    leapp.compile_graph(visualize=False)
        *made, (given_folder, _), (given_folder_success, _) = builds
        for folder, mode in made:
            self.assertEqual(mode, 0o700)
            self.assertFalse(os.path.exists(folder))
        self.assertNotEqual(made[0][0], made[1][0])
        self.assertEqual(given_folder, given.name)
        self.assertEqual(given_folder_success, given.name)
        with open(given_cache, "rb") as f:
            self.assertEqual(f.read(), b"timings")

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_tensorrt_options(self):
        """Covers what TensorRT does not take: the CUDA backend runs it, an
        operator TensorRT runs through a PyTorch plugin still exports, a program
        built with ``require_full_compilation`` still saves, and
        ``tensorrt_options`` override LEAPP's own settings."""
        if not _has_tensorrt():
            self.skipTest(NO_TENSORRT)
        import torch_tensorrt

        compile_trt = torch_tensorrt.compile_and_save
        settings = []

        def spy(*args, **kwargs):
            settings.append(kwargs)
            return compile_trt(*args, **kwargs)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        weight = torch.rand(8, 8, generator=torch.Generator().manual_seed(0))
        options = {
            "torch_executed_ops": {"torch.ops.aten.mul.Tensor"},
            "fallback_pytorch_plugin_ops": False,
        }
        for device_io in (True, False):
            with self.subTest(device_io=device_io):

                @annotate.method(
                    export_with="executorch",
                    backend_params={
                        "delegate": "tensorrt",
                        "device_io": device_io,
                        "tensorrt_options": options,
                    },
                )
                def funcS(inputS: torch.Tensor):
                    return (inputS @ weight.to(inputS.device)) * 2

                settings.clear()
                leapp.start(name=self.TEST_GRAPH_NAME)
                funcS(torch.rand(4, 8, device="cuda"))
                leapp.stop()
                with mock.patch.object(torch_tensorrt, "compile_and_save", spy):
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {funcS.__name__: True},
                    )
                (split,) = settings
                self.assertEqual(
                    split["torch_executed_ops"], options["torch_executed_ops"]
                )
                self.assertFalse(split["fallback_pytorch_plugin_ops"])

                plan = _execution_plan(
                    os.path.join(self.TEST_GRAPH_NAME, f"{funcS.__name__}.pte")
                )
                self.assertIn("CudaBackend", [d.id for d in plan.delegates])
                engines = [d for d in plan.delegates if d.id == "TensorRTBackend"]
                self.assertTrue(engines)
                for engine in engines:
                    specs = {spec.key: spec.value for spec in engine.compile_specs}
                    self.assertEqual(specs["use_cuda_graphs"], b"1")
                with open(yaml_path) as f:
                    parameters = yaml.safe_load(f)["models"][funcS.__name__][
                        "parameters"
                    ]
                self.assertEqual(parameters["data_path"], f"{funcS.__name__}.ptd")
                self.assertEqual(
                    parameters["torch_tensorrt_version"],
                    importlib.metadata.version("torch_tensorrt"),
                )
                # save() copies the whole program folder, which must not hold the
                # private timing cache.
                self.assertEqual(
                    [f for f in os.listdir(self.TEST_GRAPH_NAME) if "timing" in f], []
                )

                fresh = torch.rand(4, 8, device="cuda")
                outputs = leapp.InferenceManager(yaml_path).run_policy(
                    {f"{funcS.__name__}/inputS": fresh}
                )
                torch.testing.assert_close(
                    next(iter(outputs.values())).cpu(),
                    (fresh.cpu() @ weight) * 2,
                    rtol=1e-4,
                    atol=1e-5,
                )

        index = torch.tensor([0, 2, 2, 4, 0])

        # Accumulating index_put runs through such a plugin by default.
        @annotate.method(
            export_with="executorch", backend_params={"delegate": "tensorrt"}
        )
        def funcP(inputP: torch.Tensor):
            return torch.zeros(6, device=inputP.device).index_put(
                (index.to(inputP.device),), inputP, accumulate=True
            )

        # Runs in float32 instead, as the guide advises.
        @annotate.method(
            export_with="executorch",
            backend_params={
                "delegate": "tensorrt",
                "tensorrt_options": {"truncate_double": True},
            },
        )
        def funcD(inputD: torch.Tensor):
            return inputD * 2 + 1

        # TensorRT takes this whole, so it builds one engine.
        @annotate.method(
            export_with="executorch",
            backend_params={
                "delegate": "tensorrt",
                "tensorrt_options": {"require_full_compilation": True},
            },
        )
        def funcF(inputF: torch.Tensor):
            return inputF * 2 + 1

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcP(torch.arange(5, dtype=torch.float32, device="cuda"))
        funcD(torch.ones(3, dtype=torch.float64, device="cuda"))
        funcF(torch.ones(3, device="cuda"))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {funcP.__name__: True, funcD.__name__: True, funcF.__name__: True},
        )
        outputs = leapp.InferenceManager(yaml_path).run_policy(
            {
                f"{funcP.__name__}/inputP": torch.tensor(
                    [1.0, 10, 100, 1000, 10000], device="cuda"
                ),
                f"{funcD.__name__}/inputD": torch.full(
                    (3,), 2.0, dtype=torch.float64, device="cuda"
                ),
                f"{funcF.__name__}/inputF": torch.full((3,), 2.0, device="cuda"),
            }
        )
        by_node = {key.split("/")[0]: value for key, value in outputs.items()}
        self.assertEqual(by_node[funcP.__name__].tolist(), [10001, 0, 110, 0, 1000, 0])
        self.assertEqual(by_node[funcD.__name__].dtype, torch.float64)
        self.assertEqual(by_node[funcD.__name__].tolist(), [5.0, 5.0, 5.0])
        self.assertEqual(by_node[funcF.__name__].tolist(), [5.0, 5.0, 5.0])

        # A new process, as in a deployment, has not yet imported the runtime
        # that registers the TensorRT backend.
        with open(yaml_path) as f:
            sha256sum = yaml.safe_load(f)["models"][funcF.__name__]["parameters"][
                "sha256sum"
            ]
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, torch\n"
                    "from leapp.backends.executorch_backend import "
                    "ExecuTorchExportBackend\n"
                    "backend = ExecuTorchExportBackend(None, "
                    "{'delegate': 'tensorrt'})\n"
                    "backend.load(sys.argv[1], sys.argv[2])\n"
                    "x = torch.full((3,), 2.0, device='cuda')\n"
                    "print(backend.compiled_model(x)[0].tolist())"
                ),
                os.path.join(self.TEST_GRAPH_NAME, f"{funcF.__name__}.pte"),
                sha256sum,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertEqual(loaded.stdout.split()[-3:], ["[5.0,", "5.0,", "5.0]"])

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    def test_executorch_backend_export_again(self):
        """A program still loaded from the output folder keeps its values when the
        graph is exported again into that folder."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        index = torch.tensor([0, 1])
        managers = []
        for value in (1.0, 7.0):
            # XNNPACK does not take an embedding, so the program reads the
            # table from its file.
            table = torch.full((4, 3), value)

            @annotate.method(export_with="executorch")
            def funcM(inputM: torch.Tensor):
                return torch.nn.functional.embedding(inputM, table)

            leapp.start(name=self.TEST_GRAPH_NAME)
            funcM(index)
            leapp.stop()
            leapp.compile_graph(visualize=False)
            managers.append(leapp.InferenceManager(yaml_path))
        for manager, value in zip(managers, (1.0, 7.0)):
            device = manager.nodes[funcM.__name__].device
            (actual,) = manager.run_policy(
                {f"{funcM.__name__}/inputM": index.to(device)}
            ).values()
            torch.testing.assert_close(actual.cpu(), torch.full((2, 3), value))

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_gpu_weights(self):
        """Covers weights captured on another device than the delegate's, and the
        CUDA backend's ``.ptd`` file: its checksum and the name the YAML gives."""

        # XNNPACK runs on the CPU, so a weight captured on the GPU must move.
        weight = torch.tensor([1.0, 2.0, 3.0], device="cuda")

        @annotate.method(export_with="executorch")
        def funcD(inputD: torch.Tensor):
            return inputD * weight + 1

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcD(torch.ones(3, device="cuda"))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {funcD.__name__: True},
        )

        # A traced module's weights come in as buffers that require grad.
        linear = torch.jit.trace(torch.nn.Linear(3, 3), torch.ones(2, 3))

        @annotate.method(
            export_with="executorch",
            backend_params={"delegate": "cuda"},
            environment_constants=["linear"],
        )
        def funcC(inputC: torch.Tensor):
            return linear(inputC)

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcC(torch.ones(2, 3))
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {funcC.__name__: True},
        )
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(yaml_path) as f:
            parameters = yaml.safe_load(f)["models"][funcC.__name__]["parameters"]
        self.assertIn("data_sha256sum", parameters)
        leapp.InferenceManager(yaml_path)
        # Loading reads the file the YAML names.
        data_path = os.path.join(self.TEST_GRAPH_NAME, "renamed.ptd")
        os.rename(
            os.path.join(self.TEST_GRAPH_NAME, parameters["data_path"]), data_path
        )
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        config["models"][funcC.__name__]["parameters"]["data_path"] = "renamed.ptd"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        leapp.InferenceManager(yaml_path)
        with open(data_path, "ab") as f:
            f.write(b"\x00")
        with self.assertRaises(ValueError):
            leapp.InferenceManager(yaml_path)

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_failed_save(self):
        """A copy that fails while a CUDA program and its ``.ptd`` are saved
        leaves the node's previous files whole and no staged copies behind."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        copy2 = shutil.copy2
        copies = []

        def copy_then_fail(src, dst, *args, **kwargs):
            copies.append(dst)
            if len(copies) == 2:
                # A disk that fills up leaves part of the copy behind.
                with open(dst, "wb") as f:
                    f.write(b"partial")
                raise OSError(errno.ENOSPC, "No space left on device", dst)
            return copy2(src, dst, *args, **kwargs)

        for value in (1.0, 7.0):
            weight = torch.full((3, 3), value, device="cuda")

            @annotate.method(
                export_with="executorch", backend_params={"delegate": "cuda"}
            )
            def funcN(inputN: torch.Tensor):
                return inputN @ weight

            leapp.start(name=self.TEST_GRAPH_NAME)
            funcN(torch.ones(2, 3, device="cuda"))
            leapp.stop()
            if value == 1.0:
                leapp.compile_graph(visualize=False)
                continue
            with (
                mock.patch.object(shutil, "copy2", copy_then_fail),
                self.assertRaises(OSError),
            ):
                leapp.compile_graph(visualize=False)
            self.assertEqual(
                [f for f in os.listdir(self.TEST_GRAPH_NAME) if f.endswith(".tmp")], []
            )
        (actual,) = (
            leapp.InferenceManager(yaml_path)
            .run_policy({f"{funcN.__name__}/inputN": torch.ones(2, 3, device="cuda")})
            .values()
        )
        torch.testing.assert_close(actual, torch.full((2, 3), 3.0, device="cuda"))

    @unittest.skipUnless(HAS_EXECUTORCH, "ExecuTorch is not installed")
    @unittest.skipUnless(torch.cuda.device_count() > 1, "Needs 2 CUDA GPUs")
    def test_executorch_backend_second_gpu(self):
        """Covers a GPU program exported with the second GPU current and as the
        default device: it is built for the first GPU and runs there whichever
        GPU is current at run time."""
        from executorch.runtime import Runtime

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        previous = torch.cuda.current_device()
        cases = [("cuda", True), ("cuda", False)]
        if _has_tensorrt():
            cases += [("tensorrt", True), ("tensorrt", False)]
        for delegate, device_io in cases:
            with self.subTest(delegate=delegate, device_io=device_io):
                backend_params = {"delegate": delegate, "device_io": device_io}
                if delegate == "tensorrt":
                    # The CUDA backend runs the mul, so the program has a part
                    # that runs on the current GPU.
                    backend_params["tensorrt_options"] = {
                        "torch_executed_ops": {"torch.ops.aten.mul.Tensor"}
                    }

                @annotate.method(
                    export_with="executorch", backend_params=backend_params
                )
                def funcJ(inputJ: torch.Tensor):
                    return inputJ * 2 + 1

                try:
                    torch.cuda.set_device(1)
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    funcJ(torch.ones(3))
                    leapp.stop()
                    # ExecuTorch lowers onto the current GPU, which must be GPU
                    # 0, with the CPU as the default device.
                    lowered_on = []
                    name = (
                        "_lower_with_tensorrt" if delegate == "tensorrt" else "_lower"
                    )

                    def lower(
                        backend, *args, run=getattr(ExecuTorchExportBackend, name)
                    ):
                        lowered_on.append(
                            (torch.cuda.current_device(), torch.get_default_device())
                        )
                        return run(backend, *args)

                    with (
                        torch.device("cuda", 1),
                        mock.patch.object(ExecuTorchExportBackend, name, lower),
                    ):
                        self.assertEqual(
                            leapp.compile_graph(visualize=False, validate=True),
                            {funcJ.__name__: True},
                        )
                    self.assertEqual(lowered_on, [(0, torch.device("cpu"))])
                    self.assertEqual(torch.cuda.current_device(), 1)
                    if delegate == "tensorrt":
                        plan = _execution_plan(
                            os.path.join(self.TEST_GRAPH_NAME, f"{funcJ.__name__}.pte")
                        )
                        delegate_ids = [d.id for d in plan.delegates]
                        self.assertIn("CudaBackend", delegate_ids)
                        self.assertIn("TensorRTBackend", delegate_ids)
                    if device_io:
                        meta = (
                            Runtime.get()
                            .load_program(
                                os.path.join(
                                    self.TEST_GRAPH_NAME, f"{funcJ.__name__}.pte"
                                )
                            )
                            .metadata("forward")
                        )
                        self.assertEqual(
                            torch.device(meta.input_tensor_meta(0).device()),
                            torch.device("cuda", 0),
                        )
                    for current in (0, 1):
                        torch.cuda.set_device(current)
                        manager = leapp.InferenceManager(yaml_path)
                        # A program loaded without GPU 0 current can still get
                        # one call right.
                        for _ in range(3):
                            value = torch.ones(3, device="cuda")
                            (actual,) = manager.run_policy(
                                {f"{funcJ.__name__}/inputJ": value}
                            ).values()
                            self.assertEqual(
                                actual.device, torch.device("cuda", current)
                            )
                            self.assertEqual(torch.cuda.current_device(), current)
                            torch.testing.assert_close(actual, value * 2 + 1)
                finally:
                    torch.cuda.set_device(previous)

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
        self.verify_all_models_exist("complex_math")

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
        self.verify_all_models_exist("complex_math")

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
        self.verify_all_models_exist("reduction_ops")

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
            [x], [expected], "tensor_method_norm"
        )

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
        self.verify_all_models_exist("reduction_ops")

    # -----------------------------------------------------------------
    # Dict / list I/O
    # -----------------------------------------------------------------

    def test_torch_trace_dict_inputs_to_list_outputs(self):
        """Test jit-trace export with dict inputs and list outputs."""

        @annotate.method(export_with="jit-trace")
        def dict_to_list(inputs: dict):
            a = inputs["a"]
            b = inputs["b"]
            sum_result = a + b
            diff_result = a - b
            return [sum_result, diff_result]

        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        dict_to_list(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("dict_to_list")

    def test_torch_trace_list_inputs_to_dict_outputs(self):
        """Test jit-trace export with list inputs and dict outputs."""

        @annotate.method(export_with="jit-trace")
        def list_to_dict(inputs: list):
            a = inputs[0]
            b = inputs[1]
            return {
                "sum": a + b,
                "product": a * b,
            }

        input_list = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        list_to_dict(input_list)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("list_to_dict")

    def test_torch_trace_nested_dict_input(self):
        """Test jit-trace export with nested dict input."""

        @annotate.method(export_with="jit-trace")
        def process_nested_dict(nested: dict):
            a = nested["level1"]["a"]
            b = nested["level1"]["b"]
            c = nested["level2"]["c"]
            return a + b + c

        nested_input = {
            "level1": {
                "a": torch.randn(3, 4, dtype=torch.float32),
                "b": torch.randn(3, 4, dtype=torch.float32),
            },
            "level2": {
                "c": torch.randn(3, 4, dtype=torch.float32),
            },
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_dict(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_nested_dict")

    def test_torch_trace_nested_list_input(self):
        """Test jit-trace export with nested list input."""

        @annotate.method(export_with="jit-trace")
        def process_nested_list(nested: list):
            a = nested[0][0]
            b = nested[0][1]
            c = nested[1][0]
            return a * b + c

        nested_input = [
            [
                torch.randn(3, 4, dtype=torch.float32),
                torch.randn(3, 4, dtype=torch.float32),
            ],
            [torch.randn(3, 4, dtype=torch.float32)],
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_nested_list(nested_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_nested_list")

    def test_torch_trace_mixed_dict_list_input(self):
        """Test jit-trace export with mixed dict and list inputs."""

        @annotate.method(export_with="jit-trace")
        def process_mixed_inputs(dict_input: dict, list_input: list):
            a = dict_input["a"]
            b = dict_input["b"]
            c = list_input[0]
            d = list_input[1]
            return a + b, c * d

        dict_input = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_mixed_inputs(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_mixed_inputs")

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
        self.verify_all_models_exist("create_nested_list_output")

    def test_torch_trace_dict_of_lists_output(self):
        """Test jit-trace export with dict containing list values as output."""

        @annotate.method(export_with="jit-trace")
        def create_dict_of_lists(x: torch.Tensor):
            return {
                "group_a": [x[0:1], x[1:2]],
                "group_b": [x[2:3], x[3:4]],
            }

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_dict_of_lists(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("create_dict_of_lists")

    def test_torch_trace_list_of_dicts_output(self):
        """Test jit-trace export with list containing dict values as output."""

        @annotate.method(export_with="jit-trace")
        def create_list_of_dicts(x: torch.Tensor):
            return [
                {"a": x[0:1], "b": x[1:2]},
                {"a": x[2:3], "b": x[3:4]},
            ]

        input_tensor = torch.randn(4, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        create_list_of_dicts(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("create_list_of_dicts")

    def test_torch_trace_mixed_return_types(self):
        """Test jit-trace export with mixed return types: tensor, list, and dict."""

        @annotate.method(export_with="jit-trace")
        def mixed_outputs(x: torch.Tensor):
            single = x[0:1]
            list_out = [x[1:2], x[2:3]]
            dict_out = {"a": x[3:4], "b": x[4:5]}
            return single, list_out, dict_out

        input_tensor = torch.randn(5, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        mixed_outputs(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("mixed_outputs")

    def test_torch_trace_bidirectional_dict_list_io(self):
        """Test jit-trace export with dict and list inputs producing list and dict outputs."""

        @annotate.method(export_with="jit-trace")
        def bidirectional_io(dict_input: dict, list_input: list):
            list_out = [v + 1 for v in dict_input.values()]
            dict_out = {f"item_{i}": v + 1 for i, v in enumerate(list_input)}
            return list_out, dict_out

        dict_input = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
            "c": torch.randn(3, 4, dtype=torch.float32),
        }
        list_input = [
            torch.randn(3, 4, dtype=torch.float32),
            torch.randn(3, 4, dtype=torch.float32),
        ]

        leapp.start(name=self.TEST_GRAPH_NAME)
        bidirectional_io(dict_input, list_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("bidirectional_io")

    # -----------------------------------------------------------------
    # Deeply nested and stress tests
    # -----------------------------------------------------------------

    def test_torch_trace_deeply_nested_dict_input(self):
        """Test jit-trace export with deeply nested dict input (3+ levels)."""

        @annotate.method(export_with="jit-trace")
        def process_deep_dict(nested: dict):
            a = nested["l1"]["l2"]["l3"]["a"]
            b = nested["l1"]["l2"]["l3"]["b"]
            return a + b

        deep_input = {
            "l1": {
                "l2": {
                    "l3": {
                        "a": torch.randn(3, 4, dtype=torch.float32),
                        "b": torch.randn(3, 4, dtype=torch.float32),
                    }
                }
            }
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_deep_dict(deep_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_deep_dict")

    def test_torch_trace_many_tensor_inputs(self):
        """Test jit-trace export with many individual tensor inputs."""

        @annotate.method(export_with="jit-trace")
        def sum_many_tensors(
            t0: torch.Tensor,
            t1: torch.Tensor,
            t2: torch.Tensor,
            t3: torch.Tensor,
            t4: torch.Tensor,
            t5: torch.Tensor,
            t6: torch.Tensor,
            t7: torch.Tensor,
            t8: torch.Tensor,
            t9: torch.Tensor,
        ):
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

        tensors = [torch.randn(3, 4, dtype=torch.float32) for _ in range(10)]

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_many_tensors(*tensors)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("sum_many_tensors")

    def test_torch_trace_many_dict_tensor_inputs(self):
        """Test jit-trace export with dict containing many tensors."""

        @annotate.method(export_with="jit-trace")
        def sum_dict_tensors(data: dict):
            total = data["t0"]
            for i in range(1, 10):
                total = total + data[f"t{i}"]
            return total

        input_dict = {
            f"t{i}": torch.randn(3, 4, dtype=torch.float32) for i in range(10)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        sum_dict_tensors(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("sum_dict_tensors")

    def test_torch_trace_split_tensor_to_many_outputs(self):
        """Test jit-trace export that splits one tensor into many outputs."""

        @annotate.method(export_with="jit-trace")
        def split_to_many(x: torch.Tensor):
            return [x[i : i + 1] for i in range(8)]

        input_tensor = torch.randn(8, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        split_to_many(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("split_to_many")

    def test_torch_trace_large_nested_structure(self):
        """Test jit-trace export with large nested dict/list structure."""

        @annotate.method(export_with="jit-trace")
        def process_large_structure(data: dict):
            results = []
            for i in range(4):
                group = data[f"group_{i}"]
                for j in range(2):
                    results.append(group[j] * 2.0)
            return torch.stack(results).sum(dim=0)

        large_input = {
            f"group_{i}": [torch.randn(3, 4, dtype=torch.float32) for _ in range(2)]
            for i in range(4)
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        process_large_structure(large_input)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process_large_structure")

    def test_torch_trace_complex_nested_dict_with_extra_input(self):
        """Test jit-trace export with nested dict input plus additional tensor input."""

        @annotate.method(export_with="jit-trace")
        def nested_with_extra(nested_input: dict, extra_tensor: torch.Tensor):
            nested = nested_input[0]["nested"]
            a = nested["a"]
            b = nested["b"]
            return [a + extra_tensor, b * extra_tensor]

        nested = [
            {
                "nested": {
                    "a": torch.randn(3, 4, dtype=torch.float32),
                    "b": torch.randn(3, 4, dtype=torch.float32),
                }
            }
        ]
        extra = torch.randn(3, 4, dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        nested_with_extra(nested, extra)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("nested_with_extra")

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
        self.verify_all_models_exist("step_a", "step_b")

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
        self.verify_all_models_exist("step_a", "step_b")

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
                a = inputs["a"]
                b = inputs["b"]
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process")

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
        self.verify_all_models_exist("split_process")

    def test_torch_script_nn_module_with_dict_input(self):
        """Test jit-script export with nn.Module method that takes dict input."""

        class ProcessingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            @annotate.method(export_with="jit")
            def process(self, inputs: dict):
                a = inputs["a"]
                b = inputs["b"]
                return self.linear(a + b)

        module = ProcessingModule()
        input_dict = {
            "a": torch.randn(3, 4, dtype=torch.float32),
            "b": torch.randn(3, 4, dtype=torch.float32),
        }

        leapp.start(name=self.TEST_GRAPH_NAME)
        module.process(input_dict)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("process")

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
        self.verify_all_models_exist("split_process")

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
        self.verify_all_models_exist("multi_output")

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
        self.verify_all_models_exist("multi_output")

    def test_torch_trace_tensor_broadcasting(self):
        """Test jit-trace export with tensor broadcasting operations."""

        @annotate.method(export_with="jit-trace")
        def broadcast_ops(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = inputA * 2.0
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0, scaled, torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        broadcast_ops(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("broadcast_ops")

    def test_torch_script_tensor_broadcasting(self):
        """Test jit-script export with tensor broadcasting operations."""

        @annotate.method(export_with="jit")
        def broadcast_ops(inputA: torch.Tensor, mask: torch.Tensor):
            scaled = inputA * 2.0
            masked_output = torch.where(
                mask.unsqueeze(-1) > 0, scaled, torch.zeros_like(scaled)
            )
            return masked_output

        input_tensor = torch.randn(2, 4, dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        broadcast_ops(input_tensor, mask)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_all_models_exist("broadcast_ops")

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
            [x, y], [expected], "compute"
        )

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
            [x, y], [expected], "compute"
        )

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
            [x], [expected_half1, expected_half2], "split_and_scale"
        )


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
