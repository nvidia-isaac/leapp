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
import copy
import errno
import importlib.util
import os
import json
import shutil
import subprocess
import sys
import unittest
import tempfile
from unittest import mock
import torch
import yaml
import onnx
import onnxruntime as ort
from onnxscript.rewriter import _matcher
import leapp
from leapp.leapp import _MANAGER as annotate
from leapp import TensorSemantics, InputKindEnum, OutputKindEnum
from leapp.backends.export_backend import SimplifiedONNXProgram
from leapp.leapp_graph.leapp_node import LeappNode
from .base import LEAPPFunctionalTestBase
import pytest

# The ExecuTorch backend refuses an older torch when it is imported.
HAS_EXECUTORCH = importlib.util.find_spec("executorch") is not None and tuple(
    int(part) for part in torch.__version__.split(".")[:2]
) >= (2, 15)
if HAS_EXECUTORCH:
    from leapp.backends.executorch_backend import (
        ExecuTorchExportBackend,
        ExecuTorchProgram,
    )


def _execution_plan(program_path):
    """Return the execution plan of an ExecuTorch ``.pte`` file."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    with open(program_path, "rb") as f:
        (plan,) = deserialize_pte_binary(f.read()).program.execution_plan
    return plan


class TestBackendImports(unittest.TestCase):
    """Optional runtime requirements do not prevent using other backends."""

    def _run(self, torch_version, code):
        script = (
            "import torch\n"
            f"torch.__version__ = torch.torch_version.TorchVersion({torch_version!r})\n"
            f"{code}"
        )
        root = os.path.dirname(os.path.dirname(leapp.__file__))
        env = {
            **os.environ,
            "PYTHONPATH": root,
            "CUDA_VISIBLE_DEVICES": "",
            # Python 3.13+ colors tracebacks under FORCE_COLOR and breaks the checks.
            "PYTHON_COLORS": "0",
        }
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    def test_executorch_backend_needs_torch_2_15(self):
        for torch_version in ("2.14.0", "2.7.0"):
            with self.subTest(torch_version=torch_version):
                result = self._run(
                    torch_version, "import leapp.backends.executorch_backend"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "ImportError: LEAPP's ExecuTorch backend needs torch 2.15 and "
                    "Python 3.11 or newer",
                    result.stderr,
                )
                self.assertIn(f"found torch {torch_version}", result.stderr)

        for torch_version in (
            "2.15.0",
            "2.15.0+cu132",
            "2.15.0.dev20260909+cu132",
            "2.16.0",
        ):
            with self.subTest(torch_version=torch_version):
                result = self._run(
                    torch_version, "import leapp.backends.executorch_backend"
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_other_backends_load_on_older_torch(self):
        code = (
            "import leapp\n"
            "from leapp.inference_manager import NodeManager\n"
            "for name in ('jit', 'onnx', 'pt2'):\n"
            "    print(type(NodeManager._create_backend(None, name, {})).__name__)\n"
            "try:\n"
            "    NodeManager._create_backend(None, 'pte', {})\n"
            "except ImportError as exc:\n"
            "    print('pte:', exc)\n"
        )
        result = self._run("2.7.0", code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "TorchExportBackend",
                "ONNXExportBackend",
                "ExportedProgramExportBackend",
                "pte: LEAPP's ExecuTorch backend needs torch 2.15 and Python 3.11 "
                "or newer, as ExecuTorch 1.6 does; found torch 2.7.0",
            ],
        )

    def test_import_leapp_registers_the_onnx_copy_lowering(self):
        code = (
            "import os, sys, tempfile\n"
            "import leapp\n"
            "assert 'leapp.backends.onnx_export_backend' not in sys.modules\n"
            "class Copy(torch.nn.Module):\n"
            "    def forward(self, dst, src):\n"
            "        return torch.ops.aten.copy.default(dst, src)\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    torch.onnx.export(torch.fx.symbolic_trace(Copy()),\n"
            "        (torch.zeros(2), torch.ones(2)), os.path.join(tmp, 'm.onnx'),\n"
            "        dynamo=False, opset_version=17)\n"
        )
        result = self._run(torch.__version__, code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tensorrt_is_optional(self):
        code = (
            "import sys\n"
            "sys.modules['tensorrt'] = None\n"
            "import leapp\n"
            "from leapp.inference_manager import NodeManager\n"
            "from leapp.backends.export_backend import NoneExportBackend\n"
            "assert 'leapp.backends.tensorrt_backend' not in sys.modules\n"
            "for name in ('jit', 'onnx', 'pt2', 'trt'):\n"
            "    NodeManager._create_backend(None, name, {})\n"
        )
        result = self._run(torch.__version__, code)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestTensorRTBackend(LEAPPFunctionalTestBase):
    @unittest.skipUnless(torch.cuda.is_available(), "TensorRT execution requires CUDA")
    def test_external_engine_validation_and_relocated_pipeline(self):
        """Exercise real TRT execution, strict cached validation, artifact copying,
        and relocated YAML reload in a mixed TRT/TorchScript pipeline. Unit tests
        cannot establish CUDA stream ordering or numerical execution correctness.
        """
        trt = pytest.importorskip("tensorrt")
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        x = network.add_input("x", trt.float32, (2, 3))
        y = network.add_input("y", trt.float32, (2, 3))
        difference = network.add_elementwise(
            x, y, trt.ElementWiseOperation.SUB
        ).get_output(0)
        total = network.add_elementwise(x, y, trt.ElementWiseOperation.SUM).get_output(
            0
        )
        difference.name, total.name = "difference", "total"
        network.mark_output(difference)
        network.mark_output(total)
        serialized = builder.build_serialized_network(
            network, builder.create_builder_config()
        )
        self.assertIsNotNone(serialized)

        with tempfile.TemporaryDirectory() as tmp:
            engine_path = os.path.join(tmp, "external.trt")
            with open(engine_path, "wb") as f:
                f.write(bytes(serialized))
            leapp.start(name=self.TEST_GRAPH_NAME, max_cached_io=2)
            try:
                for offset in (0.0, 4.0):
                    x, y = annotate.input_tensors(
                        "engine",
                        {
                            "x": torch.arange(6, dtype=torch.float32).reshape(2, 3)
                            + offset,
                            "y": torch.full((2, 3), 2.0),
                        },
                    )
                    difference, total = x - y, x + y
                    annotate.output_tensors(
                        "engine",
                        {"difference": difference, "total": total},
                        export_with=None,
                        backend_params={
                            "model_path": engine_path,
                            "copy_original_model": True,
                        },
                    )
                    difference, total = annotate.input_tensors(
                        "post", {"difference": difference, "total": total}
                    )
                    annotate.output_tensors(
                        "post",
                        {"result": difference * 2 + total},
                        export_with="jit-trace",
                    )
            finally:
                leapp.stop()
            leapp.compile_graph(visualize=False, validate=True, strict=True)
            yaml_path = os.path.join(
                self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml"
            )
            with open(yaml_path) as f:
                self.assertEqual(
                    yaml.safe_load(f)["models"]["engine"]["parameters"]["backend"],
                    "trt",
                )

            relocated = shutil.copytree(
                self.TEST_GRAPH_NAME, os.path.join(tmp, "relocated")
            )
            os.remove(engine_path)
            shutil.rmtree(self.TEST_GRAPH_NAME)
            manager = leapp.InferenceManager(
                os.path.join(relocated, f"{self.TEST_GRAPH_NAME}.yaml")
            )
            for device in ("cpu", "cuda"):
                # Use a nondefault stream, noncontiguous inputs, and changing
                # values to catch stale bindings and stream/buffer lifetime errors.
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    x = (
                        torch.arange(6, dtype=torch.float32, device=device)
                        .reshape(3, 2)
                        .T
                    )
                    y = torch.full((2, 3), 3.0, device=device)
                    result = manager.run_policy({"engine/x": x, "engine/y": y})[
                        "post/result"
                    ]
                    torch.testing.assert_close(result, 3 * x - y)
                    saved = result.clone()
                    changed = manager.run_policy(
                        {"engine/x": x + 7, "engine/y": y - 2}
                    )["post/result"]
                    torch.testing.assert_close(changed, 3 * (x + 7) - (y - 2))
                    torch.testing.assert_close(result, saved)


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

    def test_constant_type_as_export_and_replay(self):
        """Check initializer naming through real Dynamo export and ORT reload.

        FX conversion equivalence alone cannot detect aliased initializer
        renaming, especially when the constant is also a graph output.
        """
        constant = torch.linspace(-0.7, 1.3, 12).reshape(3, 4)
        double = constant.double()

        def compute(x):
            converted = constant.type_as(x)
            return {
                "noop": x + converted,
                "cast": x + double.type_as(x),
                "constant": converted,
                "reused": x + converted + constant.type_as(x),
            }

        leapp.start(self.TEST_GRAPH_NAME, max_cached_io=2)
        try:
            for shift in (0.0, 0.5):
                x = leapp.annotate.input_tensors(
                    "policy", {"x": torch.zeros(2, 3, 4) + shift}
                )
                leapp.annotate.output_tensors("policy", compute(x), export_with="onnx")
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        manager = leapp.InferenceManager(
            os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        )
        for shift in (-2.0, 3.0):
            fresh = torch.arange(24.0).reshape(2, 3, 4) + shift
            actual = manager.run_policy({"policy/x": fresh})
            for name, expected in compute(fresh).items():
                torch.testing.assert_close(
                    actual[f"policy/{name}"], expected, rtol=0, atol=0
                )

    def test_integer_sort_ties_and_mismatch_diagnostics(self):
        """Cover real CUDA ONNX sorting and integer failure reporting in one test.

        Export once and replay changed duplicate keys, including large int64 keys.
        A separate deliberate mismatch must report statistics without changing
        integer outputs to floating point.
        """
        if (
            not torch.cuda.is_available()
            or "CUDAExecutionProvider" not in ort.get_available_providers()
        ):
            self.skipTest("requires CUDA and GPU ONNX Runtime")
        keys = (
            torch.arange(2 * 3632, dtype=torch.int64, device="cuda").reshape(2, -1) % 11
            - 5
        )
        keys[:, 0] = torch.iinfo(torch.int64).min
        # ORT CUDA TopK separately mishandles its exact INT64_MAX sentinel.
        keys[:, 1] = torch.iinfo(torch.int64).max - 1
        leapp.start(self.TEST_GRAPH_NAME, global_patching=False)
        try:
            x = annotate.input_tensors("sort", {"keys": keys})
            annotate.output_tensors(
                "sort", {"indices": torch.argsort(x, dim=-1)}, export_with="onnx"
            )
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=False)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = os.path.join(self.TEST_GRAPH_NAME, "sort_profile")
        session = ort.InferenceSession(
            os.path.join(self.TEST_GRAPH_NAME, "sort.onnx"),
            options,
            providers=["CUDAExecutionProvider"],
        )
        session.disable_fallback()
        with self.subTest("CUDA sort parity"):
            for index, sample in enumerate((keys, keys.roll(317, dims=-1))):
                actual = session.run(None, {"keys": sample.cpu().numpy()})[0]
                if index == 0:
                    with open(session.end_profiling()) as stream:
                        events = json.load(stream)
                    providers = {
                        event.get("args", {}).get("provider")
                        for event in events
                        if event.get("args", {}).get("op_name") == "TopK"
                    }
                    self.assertIn("CUDAExecutionProvider", providers)
                torch.testing.assert_close(
                    torch.from_numpy(actual),
                    torch.argsort(sample, dim=-1).cpu(),
                    rtol=0,
                    atol=0,
                )
        with self.subTest("integer mismatch diagnostics"):
            node = LeappNode("integer_mismatch")
            node.add_output("indices", "indices", torch.tensor([0, 1]))
            node.export_backend.compiled_model = lambda: torch.tensor([0, 2])
            with self.assertLogs("leapp", level="ERROR") as logs:
                self.assertEqual(
                    node.validate_compiled_model(rtol=0, atol=0), (False, None)
                )
            self.assertIn(
                "integer_mismatch/indices sample 0: Mismatch detected",
                "\n".join(logs.output),
            )

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
        """Exercise real export with the matcher optimization active beforehand."""

        @annotate.method(export_with="onnx")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        matcher = _matcher.SimplePatternMatcher
        stock = matcher.match
        if getattr(stock, "_leapp_root_guard", False):
            stock = stock.__wrapped__
        original_export = torch.onnx.export

        def export(*args, **kwargs):
            self.assertTrue(matcher.match._leapp_root_guard)
            self.assertIs(matcher.match.__wrapped__, stock)
            self.assertTrue(kwargs["optimize"])
            return original_export(*args, **kwargs)

        # Start unpatched even if another test exported first; restore on exit.
        with (
            mock.patch.object(matcher, "match", stock),
            mock.patch("torch.onnx.export", side_effect=export) as export_mock,
        ):
            leapp.start(name=self.TEST_GRAPH_NAME)
            funcA(input_tensor)
            leapp.stop()
            leapp.compile_graph(visualize=False)
            export_mock.assert_called_once()

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

    def test_pt2_reload_uses_runtime_device(self):
        """Relocate saved transfers, factories and constants as one program.

        Exercise CPU export to CUDA and mixed-device export to either target;
        cached validation alone cannot cover runtime relocation. The combined
        device/dtype conversion must retain its dtype on changed-input replay.
        """
        cases = [("cpu", "cpu")]
        if torch.cuda.is_available():
            cases += [("cpu", "cuda"), ("cuda", "cuda"), ("cuda", "cpu")]
        for device, runtime_device in cases:
            with self.subTest(device=device, runtime_device=runtime_device):
                gain = torch.tensor([2.0, 0.5], device=device)
                lower = torch.tensor([-1.0, 0.0])
                upper = torch.tensor([1.0, 2.0])
                leapp.start(name=self.TEST_GRAPH_NAME, max_cached_io=2)
                try:
                    for shift in (0.0, 0.5):
                        x, offset = leapp.annotate.input_tensors(
                            "policy",
                            {
                                "x": torch.tensor([0.5, 3.0], device=device) + shift,
                                "offset": torch.tensor([0.1, -0.2]) + shift,
                            },
                        )
                        result = (x * gain).to(device="cpu", dtype=torch.float64)
                        result = result.clamp(lower, upper) + offset
                        result = result + torch.arange(2, device="cpu")
                        leapp.annotate.output_tensors(
                            "policy", {"result": result}, export_with="pt2"
                        )
                finally:
                    leapp.stop()
                leapp.compile_graph(visualize=False, validate=True, strict=True)
                with mock.patch(
                    "leapp.backends.export_backend.ExportBackend._select_runtime_device",
                    return_value=runtime_device,
                ):
                    manager = leapp.InferenceManager(
                        os.path.join(
                            self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml"
                        )
                    )
                for values in ([-2.0, 1.0], [0.2, 5.0]):
                    fresh = torch.tensor(values, device=device)
                    offset = torch.tensor([0.3, 0.4])
                    actual = manager.run_policy(
                        {
                            "policy/x": fresh.to(runtime_device),
                            "policy/offset": offset.to(runtime_device),
                        }
                    )["policy/result"]
                    expected = (fresh * gain).to(device="cpu", dtype=torch.float64)
                    expected = expected.clamp(lower, upper) + offset + torch.arange(2)
                    self.assertEqual(actual.device.type, runtime_device)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_pt2_reload_retargets_device_arguments(self):
        """Cover positional and keyword device arguments without requiring CUDA.

        A meta transfer has no stored tensor data to migrate; after relocation,
        its operation must compute real values on the requested CPU device.
        """

        class Convert(torch.nn.Module):
            def forward(self, x):
                return x.to(device="meta", dtype=torch.float64) + 1

        node = LeappNode("relocate")
        node.setup_backend("pt2", {})
        backend = node.export_backend
        program = torch.export.export(Convert(), (torch.ones(2),))
        with tempfile.TemporaryDirectory() as directory:
            for keyword_device in (False, True):
                with self.subTest(keyword_device=keyword_device):
                    if keyword_device:
                        for fx_node in program.graph.nodes:
                            if fx_node.target is torch.ops.aten.to.device:
                                fx_node.kwargs = {
                                    "device": fx_node.args[1],
                                    "dtype": fx_node.args[2],
                                }
                                fx_node.args = fx_node.args[:1]
                        program.graph_module.recompile()
                        program.validate()
                    backend.exported_program = program
                    path, _, sha256 = backend.save(directory)
                    with mock.patch.object(
                        backend, "_select_runtime_device", return_value="cpu"
                    ):
                        backend.load(path, sha256)
                    for values in ([2.0, 3.0], [-4.0, 0.25]):
                        fresh = torch.tensor(values)
                        actual = backend.compiled_model(fresh)
                        torch.testing.assert_close(
                            actual, fresh.double() + 1, rtol=0, atol=0
                        )

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
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
                # Only the CUDA backend writes a .ptd, so an earlier case's file
                # must not be recorded.
                self.assertEqual("data_path" in parameters, delegate == "cuda")
                program_path = os.path.join(
                    self.TEST_GRAPH_NAME, f"{funcA.__name__}.pte"
                )
                plan = _execution_plan(program_path)
                backend_id = {"xnnpack": "XnnpackBackend", "cuda": "CudaBackend"}
                self.assertEqual([d.id for d in plan.delegates], [backend_id[delegate]])
                # The delegate runs the whole graph; only the copies between host
                # and GPU memory run outside it.
                copies = {"et_copy::_h2d_copy", "et_copy::_d2h_copy"}
                self.assertEqual(
                    {op.name for op in plan.operators},
                    copies if delegate == "cuda" and not device_io else set(),
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
                    "cuda" if delegate == "cuda" and device_io else "cpu",
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

    def _export_executorch_module(
        self, name, module, backend_params=None, *, validate=False
    ):
        # Direct modules retain mutable buffers that FX capture makes functional.
        node = LeappNode(name)
        node._max_cached_io = int(validate)
        node.add_input("obs", "obs", torch.ones(3))
        if validate:
            reference = copy.deepcopy(module)
            node.add_output("action", "action", reference(torch.ones(3)))
            second = torch.full((3,), 2.0)
            node.validate_input_and_update_sources("obs", "obs", second)
            node.validate_output_and_update_sources(
                "action", "action", reference(second)
            )
            node.increment_cache_idx()
        backend = ExecuTorchExportBackend(node, backend_params)
        node.export_backend = backend
        node.backend = "executorch"
        backend.compile(module)
        if validate:
            passed, hint = node.validate_compiled_model()
            self.assertTrue(passed, hint)
        os.makedirs(self.TEST_GRAPH_NAME, exist_ok=True)
        path, _, checksum = backend.save(self.TEST_GRAPH_NAME)
        loader = ExecuTorchExportBackend(None, backend.get_backend_metadata())
        loader.load(path, checksum)
        return loader.compiled_model, path

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    def test_executorch_backend_constants(self):
        """Fold constant-only operators before lowering to a delegate that lacks them."""

        class ConstantPolicy(torch.nn.Module):
            def forward(self, obs):
                return obs + torch.arange(3, dtype=torch.float32)

        program, path = self._export_executorch_module("constants", ConstantPolicy())
        self.assertEqual([op.name for op in _execution_plan(path).operators], [])
        for offset in (1.0, -4.0):
            fresh = torch.full((3,), offset)
            (actual,) = program(fresh)
            torch.testing.assert_close(actual, fresh + torch.arange(3))

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    def test_executorch_backend_written_buffers(self):
        """Retain persistent/non-persistent initial values and reset saved counters."""

        class BufferPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("count", torch.tensor([10.0, 20.0, 30.0]))
                self.register_buffer(
                    "total", torch.tensor([10.0, 20.0, 30.0]), persistent=False
                )

            def forward(self, obs):
                self.count[0] = obs[0]
                self.total[0] = obs[0]
                return (self.count + self.total) / 2

        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("steps", torch.zeros(3))

            def forward(self, obs):
                self.steps += 1
                return obs + self.steps

        program, _ = self._export_executorch_module("buffers", BufferPolicy())
        for value in (7.0, -2.0):
            fresh = torch.full((3,), value)
            (actual,) = program(fresh)
            self.assertEqual(actual.device, fresh.device)
            torch.testing.assert_close(actual, torch.tensor([value, 20.0, 30.0]))

        counter, _ = self._export_executorch_module("counter", Counter(), validate=True)
        fresh = torch.full((3,), 7.0)
        for step in (1, 2):
            (actual,) = counter(fresh)
            torch.testing.assert_close(actual, fresh + step)
        counter.reload()
        (actual,) = counter(fresh)
        torch.testing.assert_close(actual, fresh + 1)

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_cuda_written_buffer(self):
        """Initialize CUDA buffers returned outside the delegate and reset counters."""

        class BufferPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("values", torch.tensor([10.0, 20.0, 30.0]))

            def forward(self, obs):
                self.values[0] = obs[0]
                return self.values

        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("steps", torch.zeros(3))

            def forward(self, obs):
                self.steps += 1
                return obs + self.steps

        program, _ = self._export_executorch_module(
            "cuda_buffers", BufferPolicy(), {"delegate": "cuda", "device_io": False}
        )
        for value in (7.0, -2.0):
            fresh = torch.full((3,), value)
            (actual,) = program(fresh)
            self.assertEqual(actual.device, fresh.device)
            torch.testing.assert_close(actual, torch.tensor([value, 20.0, 30.0]))

        counter, _ = self._export_executorch_module(
            "cuda_counter", Counter(), {"delegate": "cuda"}, validate=True
        )
        fresh = torch.full((3,), 7.0)
        for step in (1, 2):
            (actual,) = counter(fresh)
            self.assertEqual(actual.device, fresh.device)
            torch.testing.assert_close(actual, fresh + step)
        counter.reload()
        (actual,) = counter(fresh)
        torch.testing.assert_close(actual, fresh + 1)

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
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

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_devices(self):
        """Covers nodes on different devices: a GPU node that returns an input
        unchanged, a GPU node traced on the GPU, two GPU nodes with their own
        weights files in one graph, an XNNPACK node on the CPU feeding a
        TorchScript node, and lowering while the caller's default device is the
        GPU."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        leapp.start(name=self.TEST_GRAPH_NAME)
        a, b = annotate.input_tensors("funcE", {"a": torch.ones(3), "b": torch.ones(3)})
        annotate.output_tensors(
            "funcE",
            {"same": a, "doubled": b * 2},
            export_with="executorch",
            backend_params={"delegate": "cuda"},
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

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    def test_executorch_backend_errors(self):
        """Covers the error paths, the checksum of the ``.pte`` file, that
        validation runs the program, and that temporary folders are removed."""
        from executorch.runtime import Runtime

        with self.assertRaises(ValueError):
            ExecuTorchExportBackend(None, {"delegate": "unknown"})
        with self.assertRaises(TypeError):
            ExecuTorchExportBackend(None, {"delegate": "cuda", "device_io": "False"})
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
        # A loaded program removes its folder when it is deleted.
        folder = tempfile.mkdtemp()
        shutil.copy2(model, folder)
        program = ExecuTorchProgram(
            os.path.join(folder, os.path.basename(model)), temp_dir=folder
        )
        torch.testing.assert_close(program(torch.ones(3))[0], torch.full((3,), 2.0))
        del program
        self.assertFalse(os.path.exists(folder))

        # A program that fails to lower or to load leaves no temporary folder behind.
        made = []
        mkdtemp = tempfile.mkdtemp

        def recorded_mkdtemp(*args, **kwargs):
            made.append(mkdtemp(*args, **kwargs))
            return made[-1]

        for owner, step in (
            (ExecuTorchExportBackend, "_lower"),
            (ExecuTorchProgram, "reload"),
        ):
            with self.subTest(step=step):
                made.clear()
                leapp.start(name=self.TEST_GRAPH_NAME)
                funcB(torch.ones(3))
                leapp.stop()
                with (
                    mock.patch("tempfile.mkdtemp", recorded_mkdtemp),
                    mock.patch.object(owner, step, side_effect=RuntimeError(step)),
                    self.assertRaisesRegex(Exception, step),
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

        @annotate.method(export_with="executorch", backend_params={"delegate": "cuda"})
        def funcI(inputI: torch.Tensor):
            return inputI * 2

        leapp.start(name=self.TEST_GRAPH_NAME)
        funcI(torch.ones(3))
        leapp.stop()
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "needs a CUDA GPU"),
        ):
            leapp.compile_graph(visualize=False)

        # With 0 rows the input has no elements, with 1 row only the output.
        for device_io in (True, False):

            @annotate.method(
                export_with="executorch",
                backend_params={"delegate": "cuda", "device_io": device_io},
            )
            def funcK(inputK: torch.Tensor):
                return inputK[1:] * 2

            for rows in (0, 1):
                with self.subTest(rows=rows, device_io=device_io):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    funcK(torch.ones(rows, 3))
                    leapp.stop()
                    with (
                        mock.patch("torch.cuda.is_available", return_value=True),
                        self.assertRaisesRegex(RuntimeError, "has no elements"),
                    ):
                        leapp.compile_graph(visualize=False)

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    def test_executorch_backend_export_again(self):
        """A program still loaded from the output folder keeps its values when the
        graph is exported again into that folder, also when that export fails."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        index = torch.tensor([0, 1])
        managers = []
        replace = os.replace

        def rename_into_export_fails(src, dst, *args, **kwargs):
            if os.path.dirname(os.path.abspath(dst)) == os.path.abspath(
                self.TEST_GRAPH_NAME
            ) and src.endswith((".tmp", ".old")):
                raise OSError(errno.EIO, "Input/output error", dst)
            return replace(src, dst, *args, **kwargs)

        for value in (1.0, 7.0, 9.0):
            # XNNPACK does not take an embedding, so the program reads the
            # table from its file.
            table = torch.full((4, 3), value)

            @annotate.method(export_with="executorch")
            def funcM(inputM: torch.Tensor):
                return torch.nn.functional.embedding(inputM, table)

            leapp.start(name=self.TEST_GRAPH_NAME)
            funcM(index)
            leapp.stop()
            if value == 9.0:
                # A rename into place that keeps failing leaves the previous .pte
                # in place, since a program with one file is never moved aside.
                with (
                    mock.patch.object(os, "replace", rename_into_export_fails),
                    self.assertRaises(OSError),
                ):
                    leapp.compile_graph(visualize=False)
            else:
                leapp.compile_graph(visualize=False)
            managers.append(leapp.InferenceManager(yaml_path))
        for manager, value in zip(managers, (1.0, 7.0, 7.0)):
            device = manager.nodes[funcM.__name__].device
            (actual,) = manager.run_policy(
                {f"{funcM.__name__}/inputM": index.to(device)}
            ).values()
            torch.testing.assert_close(actual.cpu(), torch.full((2, 3), value))

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
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

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    @unittest.skipUnless(torch.cuda.is_available(), "Needs a CUDA GPU")
    def test_executorch_backend_failed_save(self):
        """A copy or a rename that fails while a CUDA program and its ``.ptd`` are
        saved leaves the node's previous files whole and no staged copies behind. A
        staged copy that cannot be removed is the only file left, and the save's own
        error is still raised. A failure to remove a moved-aside file does not fail
        the save, and the next save removes it."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        copy2 = shutil.copy2
        copies = []
        renamed = []
        replace = os.replace

        def rename_into_export_fails(n):
            # Saving 2 files over 2 does 3 renames into the folder: moving the
            # first old file aside, then renaming each new file into place.
            def fail(src, dst, *args, **kwargs):
                # The compilers rename their own cache files, which must still work.
                if os.path.dirname(os.path.abspath(dst)) == os.path.abspath(
                    self.TEST_GRAPH_NAME
                ):
                    renamed.append(dst)
                    if len(renamed) == n:
                        raise OSError(errno.EIO, "Input/output error", dst)
                return replace(src, dst, *args, **kwargs)

            return fail

        def copy_then_fail(src, dst, *args, **kwargs):
            # Fail the second staged copy, not a copy the compilers make.
            if os.path.dirname(os.path.abspath(dst)) == os.path.abspath(
                self.TEST_GRAPH_NAME
            ) and dst.endswith(".tmp"):
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
            failures = (
                mock.patch.object(shutil, "copy2", copy_then_fail),
                *(
                    mock.patch.object(os, "replace", rename_into_export_fails(n))
                    for n in (1, 2, 3)
                ),
            )
            for failure in failures:
                renamed.clear()
                with failure, self.assertRaises(OSError):
                    leapp.compile_graph(visualize=False)
                self.assertEqual(
                    [
                        f
                        for f in os.listdir(self.TEST_GRAPH_NAME)
                        if f.endswith((".tmp", ".old"))
                    ],
                    [],
                )
        # A staged copy that cannot be removed does not hide the save's own
        # error, and the other staged copy is still removed.
        remove = os.remove
        stuck = []

        def remove_first_staged_fails(path, *args, **kwargs):
            if (
                not stuck
                and os.path.dirname(os.path.abspath(path))
                == os.path.abspath(self.TEST_GRAPH_NAME)
                and path.endswith(".tmp")
            ):
                stuck.append(path)
                raise OSError(errno.EACCES, "Permission denied", path)
            return remove(path, *args, **kwargs)

        renamed.clear()
        with (
            mock.patch.object(os, "replace", rename_into_export_fails(1)),
            mock.patch.object(os, "remove", remove_first_staged_fails),
            self.assertRaises(OSError) as caught,
        ):
            leapp.compile_graph(visualize=False)
        self.assertEqual(caught.exception.errno, errno.EIO)
        self.assertEqual(
            [
                f
                for f in os.listdir(self.TEST_GRAPH_NAME)
                if f.endswith((".tmp", ".old"))
            ],
            [os.path.basename(stuck[0])],
        )
        os.remove(stuck[0])
        (actual,) = (
            leapp.InferenceManager(yaml_path)
            .run_policy({f"{funcN.__name__}/inputN": torch.ones(2, 3, device="cuda")})
            .values()
        )
        torch.testing.assert_close(actual, torch.full((2, 3), 3.0, device="cuda"))

        # A save removes the file it moved aside, and a failure to remove it does
        # not fail the save, since the new files are already in place.
        def remove_in_export_fails(path, *args, **kwargs):
            if os.path.dirname(os.path.abspath(path)) == os.path.abspath(
                self.TEST_GRAPH_NAME
            ):
                raise OSError(errno.EIO, "Input/output error", path)
            return remove(path, *args, **kwargs)

        leapp.compile_graph(visualize=False)
        self.assertEqual(
            [f for f in os.listdir(self.TEST_GRAPH_NAME) if f.endswith(".old")], []
        )
        with mock.patch.object(os, "remove", remove_in_export_fails):
            leapp.compile_graph(visualize=False)
        self.assertTrue(
            [f for f in os.listdir(self.TEST_GRAPH_NAME) if f.endswith(".old")]
        )
        # The next save replaces that leftover and removes it.
        leapp.compile_graph(visualize=False)
        self.assertEqual(
            [f for f in os.listdir(self.TEST_GRAPH_NAME) if f.endswith(".old")], []
        )
        (actual,) = (
            leapp.InferenceManager(yaml_path)
            .run_policy({f"{funcN.__name__}/inputN": torch.ones(2, 3, device="cuda")})
            .values()
        )
        torch.testing.assert_close(actual, torch.full((2, 3), 21.0, device="cuda"))

    @unittest.skipUnless(HAS_EXECUTORCH, "Needs ExecuTorch and torch 2.15")
    @unittest.skipUnless(torch.cuda.device_count() > 1, "Needs 2 CUDA GPUs")
    def test_executorch_backend_second_gpu(self):
        """Covers a CUDA program exported with the second GPU current and as the
        default device: it is built for the first GPU and runs there whichever
        GPU is current at run time."""
        from executorch.runtime import Runtime

        run_lower = ExecuTorchExportBackend._lower
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        previous = torch.cuda.current_device()
        for device_io in (True, False):
            with self.subTest(device_io=device_io):

                @annotate.method(
                    export_with="executorch",
                    backend_params={"delegate": "cuda", "device_io": device_io},
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

                    def lower(backend, *args):
                        lowered_on.append(
                            (torch.cuda.current_device(), torch.get_default_device())
                        )
                        return run_lower(backend, *args)

                    with (
                        torch.device("cuda", 1),
                        mock.patch.object(ExecuTorchExportBackend, "_lower", lower),
                    ):
                        self.assertEqual(
                            leapp.compile_graph(visualize=False, validate=True),
                            {funcJ.__name__: True},
                        )
                    self.assertEqual(lowered_on, [(0, torch.device("cpu"))])
                    self.assertEqual(torch.cuda.current_device(), 1)
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
                        value = torch.ones(3, device="cuda")
                        (actual,) = (
                            leapp.InferenceManager(yaml_path)
                            .run_policy({f"{funcJ.__name__}/inputJ": value})
                            .values()
                        )
                        self.assertEqual(actual.device, torch.device("cuda", current))
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
