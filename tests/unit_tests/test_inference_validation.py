# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description and runtime validation reject invalid model artifacts and inputs."""

import contextlib
import hashlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import yaml

from leapp import InferenceManager
from leapp.backends.export_backend import NoneExportBackend
from leapp.backends.tensorrt_backend import TensorRTProgram
from leapp.backends import tensorrt_backend
from leapp.inference_manager import NodeManager


@pytest.fixture
def description():
    return {
        "models": {
            "policy": {
                "inputs": [],
                "outputs": [],
                "parameters": {
                    "model_path": "unused.pt",
                    "md5sum": "unused",
                    "sha256sum": "unused",
                    "backend": "jit",
                },
            }
        },
        "pipeline": {},
        "system information": {},
    }


def test_missing_description_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Leapp description file not found"):
        InferenceManager(str(tmp_path / "missing.yaml"))


def test_description_requires_yaml_extension(tmp_path):
    path = tmp_path / "graph.txt"
    path.write_text("")
    with pytest.raises(ValueError, match=r"must end with \.yaml"):
        InferenceManager(str(path))


@pytest.mark.parametrize("missing_key", ["models", "pipeline", "system information"])
def test_description_requires_top_level_keys(tmp_path, description, missing_key):
    description.pop(missing_key)
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(description))
    with pytest.raises(
        ValueError,
        match="Leapp description file must contain models, pipeline, and system_info",
    ):
        InferenceManager(str(path))


@pytest.mark.parametrize(
    "missing_key", ["model_path", "md5sum", "sha256sum", "backend"]
)
def test_description_requires_model_parameters(tmp_path, description, missing_key):
    description["models"]["policy"]["parameters"].pop(missing_key)
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(description))
    with pytest.raises(
        ValueError,
        match="Model description must contain model_path, md5sum, sha256sum, and backend",
    ):
        InferenceManager(str(path))


@pytest.mark.parametrize("suffix", ["engine", "plan", "trt"])
def test_tensorrt_external_artifact_type(suffix):
    backend = NoneExportBackend(None, {"model_path": f"model.{suffix}"})
    assert backend.get_backend_model_type() == "trt"


def test_tensorrt_dependency_is_only_required_when_loading(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    backend = NodeManager._create_backend(None, "trt", {})
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA256 checksum mismatch"):
        backend.load(str(path), "incorrect")
    with pytest.raises(
        ImportError, match="Install the TensorRT version and CUDA variant"
    ):
        backend.load(str(path), checksum)
    # The external-model path must reach the same optional runtime guard.
    backend = NoneExportBackend(None, {"model_path": str(path)})
    with pytest.raises(ImportError, match="requires the TensorRT Python package"):
        backend.compile()


def test_tensorrt_external_artifact_copy_and_node_reload(tmp_path, monkeypatch):
    path = tmp_path / "model.trt"
    path.write_bytes(b"engine")
    program = mock.Mock(device=torch.device("cpu"))
    monkeypatch.setattr(tensorrt_backend, "TensorRTProgram", lambda _: program)
    backend = NoneExportBackend(
        None, {"model_path": str(path), "copy_original_model": True}
    )
    backend.compile()
    assert backend.compiled_model is program
    assert backend.runtime_device == program.device
    model_path, md5sum, sha256sum = backend.save(str(tmp_path / "bundle"))
    path.unlink()
    assert (tmp_path / "bundle" / "model.trt").read_bytes() == b"engine"
    node = NodeManager(
        "engine",
        model_path,
        [],
        [],
        {"backend": "trt", "md5sum": md5sum, "sha256sum": sha256sum},
    )
    assert node.model is program
    assert node.device == program.device


@pytest.fixture
def trt_engine(tmp_path, monkeypatch):
    engine = mock.Mock(num_io_tensors=2)
    engine.get_tensor_name.side_effect = ["x", "y"]
    engine.get_tensor_shape.return_value = (2, 3)
    engine.get_tensor_location.return_value = "device"
    engine.get_tensor_format.return_value = "linear"
    engine.get_tensor_dtype.return_value = "float32"
    engine.get_tensor_mode.side_effect = ["input", "output"]
    trt = SimpleNamespace(
        Logger=mock.Mock(),
        Runtime=mock.Mock(),
        init_libnvinfer_plugins=mock.Mock(),
        TensorLocation=SimpleNamespace(DEVICE="device"),
        TensorFormat=SimpleNamespace(LINEAR="linear"),
        TensorIOMode=SimpleNamespace(INPUT="input"),
        float32="float32",
    )
    trt.Runtime.return_value.deserialize_cuda_engine.return_value = engine
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    path = tmp_path / "model.engine"
    path.write_bytes(b"engine")
    return path, engine, trt


@pytest.mark.parametrize(
    "problem,error,message",
    [
        ("cuda", RuntimeError, "CUDA-enabled GPU"),
        ("deserialize", RuntimeError, "Could not deserialize"),
        ("context", RuntimeError, "Could not create"),
        ("dynamic", ValueError, "nonempty fixed shape"),
        ("empty", ValueError, "nonempty fixed shape"),
        ("host", ValueError, "linear device storage"),
        ("packed", ValueError, "linear device storage"),
        ("dtype", TypeError, "Unsupported TensorRT dtype"),
    ],
)
def test_tensorrt_rejects_unusable_engines(
    trt_engine, monkeypatch, problem, error, message
):
    path, engine, trt = trt_engine
    if problem == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif problem == "deserialize":
        trt.Runtime.return_value.deserialize_cuda_engine.return_value = None
    elif problem == "context":
        engine.create_execution_context.return_value = None
    elif problem in ("dynamic", "empty"):
        engine.get_tensor_shape.return_value = (
            (-1, 3) if problem == "dynamic" else (0, 3)
        )
    elif problem == "host":
        engine.get_tensor_location.return_value = "host"
    elif problem == "packed":
        engine.get_tensor_format.return_value = "packed"
    elif problem == "dtype":
        engine.get_tensor_dtype.return_value = "unsupported"
    with pytest.raises(error, match=message):
        TensorRTProgram(path)


def test_tensorrt_input_checks_and_pointer_binding(trt_engine, monkeypatch):
    path, engine, _ = trt_engine
    program = TensorRTProgram(path)
    assert program.device == torch.device("cuda:0")
    # Exercise buffer preparation and bindings with CPU storage. Numerical CUDA
    # execution is covered separately by the real-engine functional test.
    program.device = torch.device("cpu")
    stream = mock.Mock(cuda_stream=123)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    context = engine.create_execution_context.return_value
    with pytest.raises(ValueError, match="Expected 1 inputs"):
        program()
    with pytest.raises(TypeError, match="requires a torch.float32 tensor"):
        program(torch.ones(2, 3, dtype=torch.int32))
    with pytest.raises(ValueError, match="requires shape"):
        program(torch.ones(3, 2))
    context.execute_async_v3.assert_not_called()

    x = torch.arange(6, dtype=torch.float32).reshape(3, 2).T
    (output,) = program(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device
    bindings = dict(call.args for call in context.set_tensor_address.call_args_list)
    assert bindings["x"] != x.data_ptr()  # A noncontiguous input needs a copy.
    assert bindings["y"] == output.data_ptr()
    context.execute_async_v3.assert_called_once_with(123)
    stream.synchronize.assert_called_once()
    context.execute_async_v3.return_value = False
    with pytest.raises(RuntimeError, match="execution failed"):
        program(x)
    assert stream.synchronize.call_count == 2
    context.set_tensor_address.return_value = False
    with pytest.raises(RuntimeError, match="Could not bind"):
        program(x)
