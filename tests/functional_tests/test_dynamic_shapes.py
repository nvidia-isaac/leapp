#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Variable dimensions through export, serialization, and runtime inference."""

import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml

import leapp
from leapp import Axis, InferenceManager, TensorSemantics, Variable, annotate
from leapp.backends.export_backend import ExportBackend
from leapp.export_manager import ExportManager
from leapp.leapp import _MANAGER


@pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
def test_dynamic_shapes_survive_export_and_reload(device, dynamic_session):
    """Round-trip inferred axes through YAML, connected models, and runtime checks."""
    x = annotate.input_tensors(
        "scale",
        leapp.TensorSemantics(
            "x",
            torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8),
            axes=[
                leapp.Axis(kind="element", names=["left", "right"]),
                leapp.Axis(
                    variable=leapp.Variable("length", min=4, max=32, multiple_of=2)
                ),
            ],
        ),
    )
    y = annotate.output_tensors("scale", {"y": x * 2}, export_with="pt2")
    y = annotate.input_tensors("pairs", {"y": y})
    annotate.output_tensors(
        "pairs", {"z": y.reshape(2, y.size(1) // 2, 2)}, export_with="pt2"
    )
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    with open(dynamic_session) as file:
        models = yaml.safe_load(file)["models"]
    for model in models.values():
        for desc in model["inputs"] + model["outputs"]:
            assert all(isinstance(size, int) for size in desc["shape"])
            assert {
                "example_shape",
                "dimension_ranges",
                "dimension_expressions",
            }.isdisjoint(desc)
            assert desc["axes"][1]["variable"]["min"] == (
                2 if desc["name"] == "z" else 4
            )
    assert models["scale"]["inputs"][0]["axes"][0]["names"] == ["left", "right"]
    assert "expression" in models["pairs"]["outputs"][0]["axes"][1]["variable"]
    manager = InferenceManager(str(dynamic_session))
    assert tuple(manager.get_mock_input()["scale/x"].shape) == (2, 8)
    for length in (4, 12, 32):
        fresh = torch.arange(2 * length, dtype=torch.float32, device=device).reshape(
            2, length
        )
        actual = manager({"scale/x": fresh})["pairs/z"]
        torch.testing.assert_close(actual, (fresh * 2).reshape(2, length // 2, 2))
    for length in (5, 34):
        with pytest.raises(ValueError):
            manager({"scale/x": torch.ones(2, length, device=device)})


@pytest.fixture(autouse=True)
def device(request, monkeypatch):
    device = getattr(request, "param", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LEAPP_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA is required by this test run but unavailable")
        pytest.skip("CUDA is unavailable")
    if device == "cpu":
        # Runtime defaults prefer CUDA; keep CPU cases on CPU even on GPU runners.
        monkeypatch.setattr(ExportBackend, "_select_runtime_device", lambda self: "cpu")
        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
        )
    return device


@pytest.fixture
def dynamic_session(tmp_path):
    leapp.start("dynamic_shapes", save_path=str(tmp_path))
    try:
        yield tmp_path / "dynamic_shapes" / "dynamic_shapes.yaml"
    finally:
        if ExportManager.is_interpret_graph_enabled():
            leapp.stop()
        _MANAGER.reset_tracing_lock()
        _MANAGER.reset_nodes()
        _MANAGER.set_dry_run_and_non_traced(False, [])


def _camera_frame(height, width, device="cpu"):
    return (
        torch.arange(3 * height * width, dtype=torch.float32, device=device).reshape(
            1, 3, height, width
        )
        / 255
    )


def _annotated_frame(image, name="image", min_height=4):
    return TensorSemantics(
        name,
        image,
        axes=[
            None,
            None,
            Axis(variable=Variable("height", min=min_height, max=12)),
            Axis(variable=Variable("width", min=4, max=12)),
        ],
    )


@pytest.fixture
def image_pipeline(dynamic_session, request):
    backend = getattr(request, "param", "pt2")

    @annotate.method(export_with=backend)
    def preprocess(image):
        features = image * 0.5
        return features

    @annotate.method(export_with=backend)
    def combine(features, residual):
        result = features + residual
        return result

    return preprocess, combine


def _compile_image_pipeline(path):
    leapp.stop()
    results = leapp.compile_graph(visualize=False, validate=True, strict=True)
    assert results and all(result is True for result in results.values())
    return InferenceManager(str(path))


@pytest.mark.parametrize(
    "backend", ["pt2", "onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"]
)
@pytest.mark.parametrize(
    "library,device",
    [
        pytest.param(torch, "cpu", id="torch-cpu"),
        pytest.param(torch, "cuda", id="torch-cuda"),
        pytest.param(np, "cpu", id="numpy-cpu"),
    ],
    indirect=["device"],
)
def test_exported_models_accept_different_input_shapes(
    device, dynamic_session, backend, library
):
    """Check saved backend artifacts on sizes never used for capture."""

    if device == "cuda" and backend.startswith("onnx"):
        assert "CUDAExecutionProvider" in ort.get_available_providers()
    array_options = {"device": device} if library is torch else {}

    def pair_tokens(x):
        return (x * 2).reshape(2, x.shape[1] // 2, 2)

    x = annotate.input_tensors(
        "pairs",
        TensorSemantics(
            "x",
            library.arange(16, dtype=library.float32, **array_options).reshape(2, 8),
            axes=[
                None,
                Axis(variable=Variable("length", min=4, max=32, multiple_of=2)),
            ],
        ),
    )
    annotate.output_tensors("pairs", {"y": pair_tokens(x)}, export_with=backend)
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(dynamic_session))
    for length in (4, 12, 32):
        fresh = library.arange(
            2 * length, dtype=library.float32, **array_options
        ).reshape(2, length)
        torch.testing.assert_close(
            manager({"pairs/x": torch.as_tensor(fresh)})["pairs/y"],
            torch.as_tensor(pair_tokens(fresh)),
        )

    if backend.startswith("onnx"):
        providers = manager.nodes["pairs"].model._session.get_providers()
        expected_provider = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        assert providers[0] == expected_provider


@pytest.mark.parametrize("factory_name", ["zeros", "ones", "full", "arange"])
@pytest.mark.parametrize("library", [torch, np], ids=["torch", "numpy"])
def test_factories_use_runtime_shape(dynamic_session, factory_name, library):
    """Check factory capture and each library's output values, shapes, and dtypes."""

    def factory(length):
        if factory_name == "full":
            return library.full((length,), 3)
        return getattr(library, factory_name)(length)

    x = annotate.input_tensors(
        "factory",
        TensorSemantics(
            "x",
            library.ones(8, dtype=library.float32),
            axes=[Axis(variable=Variable("length", min=4, max=32))],
        ),
    )
    annotate.output_tensors("factory", {"y": factory(x.shape[0])}, export_with="pt2")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(dynamic_session))
    for length in (4, 12, 32):
        actual = manager({"factory/x": torch.ones(length)})["factory/y"]
        torch.testing.assert_close(actual, torch.as_tensor(factory(length)))


@pytest.mark.parametrize(
    "other",
    [
        Variable("length", min=4, max=64),
        Variable("length", min=4, max=32, multiple_of=2),
    ],
    ids=["bounds", "divisibility"],
)
def test_conflicting_axis_declarations_are_rejected(dynamic_session, other):
    """Compilation rejects conflicting declarations across otherwise valid nodes."""
    for name, variable in (
        ("first", Variable("length", min=4, max=32)),
        ("second", other),
    ):
        x = annotate.input_tensors(
            name, TensorSemantics("x", torch.ones(8), axes=[Axis(variable=variable)])
        )
        annotate.output_tensors(name, {"y": x + 1}, export_with="pt2")
    leapp.stop()
    with pytest.raises(
        ValueError, match="Conflicting bounds for shared variable length"
    ):
        leapp.compile_graph(visualize=False, validate=True, strict=True)


def _export_two_input_model(function, inputs, path):
    left, right = annotate.input_tensors("combine", inputs)
    annotate.output_tensors("combine", {"y": function(left, right)}, export_with="pt2")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    return InferenceManager(str(path))


def test_inputs_share_a_dynamic_dimension(dynamic_session):
    """Reloaded models enforce the same runtime length across both inputs."""
    axis = Axis(variable=Variable("length", min=4, max=32))
    manager = _export_two_input_model(
        torch.mul,
        [
            TensorSemantics("left", torch.ones(2, 8, 3), axes=[None, axis, None]),
            TensorSemantics("right", torch.ones(2, 8, 3), axes=[None, axis, None]),
        ],
        dynamic_session,
    )
    for length in (4, 12, 32):
        left = torch.arange(6 * length, dtype=torch.float32).reshape(2, length, 3)
        right = left + 1
        actual = manager({"combine/left": left, "combine/right": right})["combine/y"]
        torch.testing.assert_close(actual, left * right)
    with pytest.raises(ValueError, match="shared dimension length"):
        manager(
            {"combine/left": torch.ones(2, 12, 3), "combine/right": torch.ones(2, 8, 3)}
        )


def test_inputs_have_independent_dynamic_dimensions(dynamic_session):
    """Reloaded models derive output length from independently varying inputs."""

    def concatenate(left, right):
        return torch.cat((left, right), dim=1)

    manager = _export_two_input_model(
        concatenate,
        [
            TensorSemantics(
                "left",
                torch.ones(2, 8, 3),
                axes=[None, Axis(variable=Variable("length", min=4, max=32)), None],
            ),
            TensorSemantics(
                "right",
                torch.ones(2, 8, 3),
                axes=[
                    None,
                    Axis(variable=Variable("right_length", min=4, max=32)),
                    None,
                ],
            ),
        ],
        dynamic_session,
    )
    for left_length, right_length in ((4, 8), (12, 8), (12, 32)):
        left = torch.arange(6 * left_length, dtype=torch.float32).reshape(
            2, left_length, 3
        )
        right = torch.arange(6 * right_length, dtype=torch.float32).reshape(
            2, right_length, 3
        )
        actual = manager({"combine/left": left, "combine/right": right})["combine/y"]
        torch.testing.assert_close(actual, concatenate(left, right))


def test_dynamic_input_with_static_input(dynamic_session):
    """Reloaded models allow variable lengths while enforcing the static bias shape."""
    bias = torch.arange(3, dtype=torch.float32)
    manager = _export_two_input_model(
        torch.add,
        [
            TensorSemantics(
                "left",
                torch.ones(2, 8, 3),
                axes=[None, Axis(variable=Variable("length", min=4, max=32)), None],
            ),
            TensorSemantics("right", bias, axes=[None]),
        ],
        dynamic_session,
    )
    for length in (4, 12, 32):
        left = torch.arange(6 * length, dtype=torch.float32).reshape(2, length, 3)
        actual = manager({"combine/left": left, "combine/right": bias})["combine/y"]
        torch.testing.assert_close(actual, left + bias)
    with pytest.raises(ValueError, match="expected fixed dimension 3"):
        manager({"combine/left": torch.ones(2, 12, 3), "combine/right": torch.ones(4)})


def _check_shape_export(path, function, library, backend, sample_shape, axes, shapes):
    sample = library.arange(int(np.prod(sample_shape)), dtype=library.float32).reshape(
        sample_shape
    )
    x = annotate.input_tensors("shape", TensorSemantics("x", sample, axes=axes))
    annotate.output_tensors("shape", {"y": function(x)}, export_with=backend)
    leapp.stop()
    captured = _MANAGER.get_nodes()["shape"].m
    for shape in shapes:
        value = library.arange(int(np.prod(shape)), dtype=library.float32).reshape(
            shape
        )
        torch.testing.assert_close(
            captured(torch.as_tensor(value)), torch.as_tensor(function(value))
        )
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(path))
    for shape in shapes:
        value = library.arange(int(np.prod(shape)), dtype=library.float32).reshape(
            shape
        )
        torch.testing.assert_close(
            manager({"shape/x": torch.as_tensor(value)})["shape/y"],
            torch.as_tensor(function(value)),
        )


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize(
    "library,reshape",
    [
        pytest.param(torch, lambda x: x.reshape(x.shape[0], -1), id="torch-shape"),
        pytest.param(torch, lambda x: x.reshape(x.size(0), -1), id="torch-size"),
        pytest.param(np, lambda x: x.reshape(x.shape[0], -1), id="numpy-shape"),
    ],
)
def test_reshape_uses_runtime_first_dimension(
    dynamic_session, backend, library, reshape
):
    """The original batch-reshape regression stays correct in FX and saved models."""
    _check_shape_export(
        dynamic_session,
        reshape,
        library,
        backend,
        (2, 3, 4),
        [Axis(variable=Variable("batch", min=2, max=8)), None, None],
        [(2, 3, 4), (5, 3, 4)],
    )


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize(
    "library,reshape",
    [
        pytest.param(torch, lambda x: x.reshape(-1, x.shape[2]), id="torch-shape"),
        pytest.param(torch, lambda x: x.reshape(-1, x.size(2)), id="torch-size"),
        pytest.param(np, lambda x: x.reshape(-1, x.shape[2]), id="numpy-shape"),
    ],
)
def test_reshape_uses_runtime_third_dimension(
    dynamic_session, backend, library, reshape
):
    """A variable third axis remains dynamic while the leading axes stay fixed."""
    _check_shape_export(
        dynamic_session,
        reshape,
        library,
        backend,
        (2, 3, 128),
        [None, None, Axis(variable=Variable("length", min=4, max=512))],
        [(2, 3, 128), (2, 3, 511)],
    )


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize("library", [torch, np], ids=["torch", "numpy"])
def test_slice_uses_runtime_shape(dynamic_session, backend, library):
    """Reloaded slices preserve half-length bounds over an explicit even-length domain."""

    def first_half(x):
        return x[..., : x.shape[2] // 2]

    _check_shape_export(
        dynamic_session,
        first_half,
        library,
        backend,
        (2, 3, 128),
        [None, None, Axis(variable=Variable("length", min=4, max=512, multiple_of=2))],
        [(2, 3, 128), (2, 3, 10), (2, 3, 510)],
    )


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize(
    "library,reshape",
    [
        pytest.param(
            torch, lambda x: (x + 1).reshape(-1).reshape(x.shape), id="torch-shape"
        ),
        pytest.param(
            torch, lambda x: (x + 1).reshape(-1).reshape(x.size()), id="torch-size"
        ),
        pytest.param(
            np, lambda x: (x + 1).reshape(-1).reshape(x.shape), id="numpy-shape"
        ),
    ],
)
def test_reshape_uses_full_runtime_shape(dynamic_session, backend, library, reshape):
    """Full shape tuples preserve independently varying batch and length dimensions."""
    _check_shape_export(
        dynamic_session,
        reshape,
        library,
        backend,
        (2, 3, 8),
        [
            Axis(variable=Variable("batch", min=2, max=8)),
            None,
            Axis(variable=Variable("length", min=4, max=32)),
        ],
        [(5, 3, 8), (2, 3, 11), (5, 3, 11)],
    )


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize(
    "library,flatten",
    [
        pytest.param(torch, lambda x: x.reshape(x.numel()), id="torch-numel"),
        pytest.param(np, lambda x: x.reshape(x.size), id="numpy-size"),
    ],
)
def test_reshape_uses_runtime_element_count(dynamic_session, backend, library, flatten):
    """Total element counts remain executable dependencies through export/reload."""
    _check_shape_export(
        dynamic_session,
        flatten,
        library,
        backend,
        (2, 3, 128),
        [None, None, Axis(variable=Variable("length", min=4, max=512))],
        [(2, 3, 128), (2, 3, 511)],
    )


@pytest.mark.parametrize(
    "backend", ["pt2", "onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"]
)
def test_zero_and_one_sizes_after_fresh_process_reload(dynamic_session, backend):
    """A fresh interpreter replays an admitted domain including empty and singleton axes."""
    x = annotate.input_tensors(
        "boundary",
        TensorSemantics(
            "x",
            torch.ones(2, 3, 4),
            axes=[None, None, Axis(variable=Variable("length", min=0, max=8))],
        ),
    )
    annotate.output_tensors("boundary", {"y": x * 2 + 1}, export_with=backend)
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    script = """
import sys
import torch
from leapp import InferenceManager
manager = InferenceManager(sys.argv[1])
for length in (0, 1, 5, 8):
    value = torch.arange(6 * length, dtype=torch.float32).reshape(2, 3, length)
    actual = manager({"boundary/x": value})["boundary/y"]
    torch.testing.assert_close(actual, value * 2 + 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(dynamic_session)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "corrupt_last_sample", [False, True], ids=["correct", "incorrect-reference"]
)
def test_different_size_capture_samples_are_validated(
    dynamic_session, corrupt_last_sample
):
    """Default caching validates later original outputs, not only the capture sample."""
    for length in (8, 4, 12, 32):
        x = annotate.input_tensors(
            "cached",
            TensorSemantics(
                "x",
                torch.arange(2 * length, dtype=torch.float32).reshape(2, length),
                axes=[None, Axis(variable=Variable("length", min=4, max=32))],
            ),
        )
        annotate.output_tensors("cached", {"y": x * 2}, export_with="pt2")
    leapp.stop()
    node = _MANAGER.get_nodes()["cached"]
    samples = node._build_validation_examples()
    assert [inputs[0].shape[1] for _, inputs, _ in samples] == [8, 4, 12, 32]
    for _, inputs, outputs in samples:
        torch.testing.assert_close(outputs[0], inputs[0] * 2)
    if corrupt_last_sample:
        node.outputs[0].cached_values[2] = torch.zeros_like(samples[-1][2][0])
        with pytest.raises(Exception, match="Model validation failed"):
            leapp.compile_graph(visualize=False, validate=True, strict=True)
    else:
        leapp.compile_graph(visualize=False, validate=True, strict=True)


@pytest.mark.parametrize("port", ["input", "output"])
@pytest.mark.parametrize("backend", ["onnx-dynamo", "onnx-torchscript"])
def test_saved_artifact_cannot_freeze_a_variable_dimension(
    dynamic_session, monkeypatch, port, backend
):
    """Final artifact checks run even when numerical output validation is disabled."""
    from leapp.backends.export_backend import SimplifiedONNXProgram

    save = SimplifiedONNXProgram.save

    def save_with_fixed_dimension(self, path, *args, **kwargs):
        result = save(self, path, *args, **kwargs)
        model = onnx.load(path)
        value = getattr(model.graph, port)[0]
        value.type.tensor_type.shape.dim[1].dim_value = 8
        onnx.save(model, path)
        return result

    monkeypatch.setattr(SimplifiedONNXProgram, "save", save_with_fixed_dimension)
    x = annotate.input_tensors(
        "artifact",
        TensorSemantics(
            "x",
            torch.ones(2, 8),
            axes=[None, Axis(variable=Variable("length", min=4, max=32))],
        ),
    )
    annotate.output_tensors("artifact", {"y": x * 2}, export_with=backend)
    leapp.stop()
    with pytest.raises(ValueError, match="ONNX artifact dimension 1 is fixed at 8"):
        leapp.compile_graph(visualize=False, validate=False)


def _generated_arrays(samples, shapes, seed):
    return {
        name: (
            torch.full(
                shape, seed + 3, dtype=samples[name].dtype, device=samples[name].device
            )
            if isinstance(samples[name], torch.Tensor)
            else np.full(shape, seed + 3, dtype=samples[name].dtype)
        )
        for name, shape in shapes.items()
    }


@pytest.mark.parametrize("backend", ["pt2", "onnx-dynamo"])
@pytest.mark.parametrize(
    "library,device",
    [
        pytest.param(torch, "cpu", id="torch-cpu"),
        pytest.param(torch, "cuda", id="torch-cuda"),
        pytest.param(np, "cpu", id="numpy-cpu"),
    ],
    indirect=["device"],
)
def test_annotation_validates_generated_input_shapes(
    device, dynamic_session, monkeypatch, library, backend
):
    """Generated references use the original callable and validation reloads the saved artifact."""
    executions = []

    @annotate.method(export_with=backend)
    def encode(image):
        executions.append((ExportManager.is_interpret_graph_enabled(), image.shape[1]))
        result = (image * 2).reshape(2, image.shape[1], 1)
        return result

    encode(
        TensorSemantics(
            "pixels",
            library.ones(
                (2, 8),
                dtype=library.float32,
                **({"device": device} if library is torch else {}),
            ),
            axes=[None, Axis(variable=Variable("length", min=4, max=12))],
        )
    )
    leapp.stop()
    node = _MANAGER.get_nodes()["encode"]
    loaded = []
    load = node.export_backend.load

    def record_load(path, sha256):
        result = load(path, sha256)
        loaded.append((path, node.compiled_model))
        return result

    monkeypatch.setattr(node.export_backend, "load", record_load)
    assert leapp.compile_graph(visualize=False, validate=True, strict=True) == {
        "encode": True
    }
    assert [path for path, _ in loaded] == [node.model_path]
    if backend == "onnx-dynamo":
        providers = loaded[0][1]._session.get_providers()
        expected = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        assert providers[0] == expected
    assert (False, 4) in executions and (False, 12) in executions
    assert sum(tracing for tracing, _ in executions) == 1
    assert node.method_validation is None
    assert list(_MANAGER.get_nodes()) == ["encode"]


def test_validation_rejects_frozen_shape_branch(dynamic_session):
    """A branch absent from FX is exposed by executing the original function at length 12."""

    @annotate.method(export_with="pt2")
    def branch(x):
        if x.shape[1] == 12:
            result = x * 2
        else:
            result = x + 1
        return result

    branch(
        TensorSemantics(
            "x",
            torch.ones(2, 8),
            axes=[None, Axis(variable=Variable("length", min=4, max=12))],
        )
    )
    leapp.stop()
    with pytest.raises(Exception, match="Model validation failed"):
        leapp.compile_graph(visualize=False, validate=True, strict=True)
    assert _MANAGER.get_nodes()["branch"].method_validation is None


def test_validation_rejects_frozen_shape_loop(dynamic_session, image_pipeline):
    """A downstream loop over inherited image height must fail generated-output comparison."""
    preprocess, _ = image_pipeline

    @annotate.method(export_with="pt2")
    def accumulate(features):
        result = torch.zeros_like(features[:, :, 0])
        for index in range(features.shape[2]):
            result = result + features[:, :, index]
        return result

    accumulate(preprocess(_annotated_frame(_camera_frame(8, 8), min_height=8)))
    leapp.stop()
    results = leapp.compile_graph(visualize=False, validate=True, strict=False)
    assert results["preprocess"] is True
    assert isinstance(results["accumulate"], ValueError)
    assert "Exported outputs differ" in str(results["accumulate"])


def test_validation_preserves_function_arguments(dynamic_session):
    """Reference replay restores a bound method, positional-only inputs and keyword configuration."""

    class Encoder:
        offset = 3

        @annotate.method(export_with="pt2", validation_inputs=_generated_arrays)
        def encode(self, image, /, *, mask, scale=2, shift=1):
            result = image * scale + mask + shift + self.offset
            return result

    Encoder().encode(
        TensorSemantics(
            "image",
            torch.ones(2, 8),
            axes=[None, Axis(variable=Variable("length", min=4, max=12))],
        ),
        mask=TensorSemantics(
            "attention_mask",
            torch.ones(2, 8, dtype=torch.int64),
            axes=[None, Axis(variable=Variable("length", min=4, max=12))],
        ),
        scale=4,
    )
    leapp.stop()
    assert leapp.compile_graph(visualize=False, validate=True, strict=True) == {
        "encode": True
    }


@pytest.mark.parametrize(
    "failure", ["shape", "dtype", "generator", "reference", "automatic_reference"]
)
def test_validation_reports_invalid_generated_inputs(dynamic_session, failure):
    """Invalid generated inputs and reference failures are errors, never successful validation."""

    def generate(samples, shapes, seed):
        if failure == "generator":
            raise RuntimeError("generator failed")
        if failure == "shape":
            return samples
        values = _generated_arrays(samples, shapes, seed)
        if failure == "dtype":
            values["x"] = values["x"].to(torch.int64)
        return values

    @annotate.method(
        export_with="pt2",
        validation_inputs=None if failure == "automatic_reference" else generate,
    )
    def encode(x):
        if failure in ("reference", "automatic_reference") and x.shape[1] != 8:
            raise RuntimeError("reference failed")
        result = x * 2
        return result

    encode(
        TensorSemantics(
            "x",
            torch.ones(2, 8),
            axes=[None, Axis(variable=Variable("length", min=4, max=12))],
        )
    )
    leapp.stop()
    results = leapp.compile_graph(visualize=False, validate=True, strict=False)
    assert isinstance(results["encode"], ValueError)
    assert "generated sample" in str(results["encode"])
    assert {
        "shape": "wrong shape",
        "dtype": "wrong shape or dtype",
        "generator": "generator failed",
        "reference": "Original function",
        "automatic_reference": "provide validation_inputs",
    }[failure] in str(results["encode"])


def test_static_annotation_uses_cached_samples(dynamic_session):
    """Static annotations retain cached-sample validation without original-function replay."""
    calls = []

    @annotate.method(export_with="pt2")
    def encode(x):
        calls.append(x.shape[1])
        result = x * 2
        return result

    for length in (8, 8):
        encode(
            TensorSemantics(
                "x",
                torch.ones(2, length),
            )
        )
    leapp.stop()
    assert leapp.compile_graph(visualize=False, validate=True, strict=True) == {
        "encode": True
    }
    assert calls == [8, 8]


@pytest.mark.parametrize("custom_generator", [False, True])
def test_disabled_validation_does_not_generate_inputs(
    dynamic_session, image_pipeline, custom_generator
):
    """validate=False skips reference replay and callbacks across connected dynamic nodes."""
    preprocess, _ = image_pipeline
    calls = []

    def generate(samples, shapes, seed):
        raise AssertionError("Validation should be disabled")

    @annotate.method(
        export_with="pt2", validation_inputs=generate if custom_generator else None
    )
    def encode(x):
        calls.append(x.shape[2])
        result = x * 2
        return result

    encode(preprocess(_annotated_frame(_camera_frame(8, 8))))
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=False)
    assert calls == [8]


@pytest.mark.parametrize("device", ["cpu", "cuda"], indirect=True)
@pytest.mark.parametrize(
    "image_pipeline",
    ["pt2", "onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"],
    indirect=True,
)
def test_image_pipeline_validates_shared_and_independent_axes(
    dynamic_session, image_pipeline, device
):
    """Reload connected artifacts and check shared input axes at independently changed image sizes."""
    preprocess, combine = image_pipeline
    image = _camera_frame(8, 8, device)
    residual = image.flip((-1,))
    combine(
        preprocess(_annotated_frame(image)),
        _annotated_frame(residual, "residual"),
    )
    manager = _compile_image_pipeline(dynamic_session)
    for height, width in ((4, 7), (11, 5), (12, 12)):
        image = _camera_frame(height, width, device)
        residual = image.flip((-1,))
        expected = combine.__wrapped__(preprocess.__wrapped__(image), residual)
        actual = manager({"preprocess/image": image, "combine/residual": residual})
        torch.testing.assert_close(actual["combine/result"], expected)


@pytest.mark.parametrize(
    "make_alias",
    [
        pytest.param(lambda image: image, id="same-object"),
        pytest.param(lambda image: image.view_as(image), id="shared-storage-view"),
    ],
)
def test_image_pipeline_preserves_aliased_arguments(
    dynamic_session, image_pipeline, make_alias
):
    """One merged graph port must still supply both original function parameters at fresh sizes."""
    preprocess, combine = image_pipeline
    image = _camera_frame(8, 8)
    alias = make_alias(image)
    preprocess(
        combine(
            _annotated_frame(image, "features"), _annotated_frame(alias, "residual")
        )
    )
    inputs = _MANAGER.get_nodes()["combine"].inputs
    assert len(inputs) == 1
    input_port = f"combine/{inputs[0].name}"
    manager = _compile_image_pipeline(dynamic_session)
    for height, width in ((4, 7), (11, 5)):
        image = _camera_frame(height, width)
        expected = preprocess.__wrapped__(combine.__wrapped__(image, image))
        actual = manager({input_port: image})
        torch.testing.assert_close(actual["preprocess/features"], expected)


def test_image_pipeline_preserves_static_input_values(dynamic_session, image_pipeline):
    """A broadcast bias stays fixed while the producer's two dynamic dimensions change."""
    preprocess, combine = image_pipeline
    bias = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
    combine(preprocess(_annotated_frame(_camera_frame(8, 8))), bias)
    manager = _compile_image_pipeline(dynamic_session)
    for height, width in ((4, 7), (11, 5)):
        image = _camera_frame(height, width)
        expected = combine.__wrapped__(preprocess.__wrapped__(image), bias)
        actual = manager({"preprocess/image": image, "combine/residual": bias})
        torch.testing.assert_close(actual["combine/result"], expected)


def test_image_pipeline_does_not_skip_pruned_dynamic_inputs(
    dynamic_session, image_pipeline
):
    """A frozen branch must not lose its validation obligation when its inherited input is pruned."""
    preprocess, _ = image_pipeline

    @annotate.method(export_with="pt2")
    def summarize(features, bias):
        if features.shape[2] > 8:
            result = bias + 1
        else:
            result = bias * 2
        return result

    bias = torch.full((1, 3), 3.0)
    summarize(preprocess(_annotated_frame(_camera_frame(8, 8))), bias)
    assert "features" in _MANAGER.get_nodes()["summarize"].trimmed_inputs
    leapp.stop()
    results = leapp.compile_graph(visualize=False, validate=True, strict=False)
    assert results["preprocess"] is True
    error = results["summarize"]
    assert isinstance(error, ValueError)
    assert "variable input" in str(error) or "Exported outputs differ" in str(error)


def test_image_pipeline_accepts_shape_branch_constant_over_domain(
    dynamic_session, image_pipeline
):
    """A branch that stays constant throughout the declared domain must retain output parity."""
    preprocess, _ = image_pipeline

    @annotate.method(export_with="pt2")
    def bounded(features):
        if features.shape[2] >= 4:
            result = features + 1
        else:
            result = features * 2
        return result

    bounded(preprocess(_annotated_frame(_camera_frame(8, 8))))
    manager = _compile_image_pipeline(dynamic_session)
    for height in (4, 7, 12):
        image = _camera_frame(height, 8)
        expected = bounded.__wrapped__(preprocess.__wrapped__(image))
        actual = manager({"preprocess/image": image})
        torch.testing.assert_close(actual["bounded/result"], expected)


@pytest.mark.parametrize(
    "capture_context",
    [
        pytest.param(torch.enable_grad, id="grad-enabled"),
        pytest.param(torch.no_grad, id="no-grad"),
        pytest.param(torch.inference_mode, id="inference-mode"),
        pytest.param(
            lambda: torch.autocast("cpu", dtype=torch.bfloat16), id="autocast"
        ),
    ],
)
def test_image_pipeline_replays_capture_context(
    dynamic_session, image_pipeline, capture_context
):
    """Original reference replay must use the capture context, even when compilation runs outside it."""
    preprocess, _ = image_pipeline

    @annotate.method(export_with="pt2")
    def contextual_features(features):
        scale = 2 if torch.is_grad_enabled() else 3
        offset = 1 if torch.is_inference_mode_enabled() else 0
        offset += 4 if torch.is_autocast_enabled("cpu") else 0
        result = features * scale + offset
        return result

    with capture_context():
        contextual_features(preprocess(_annotated_frame(_camera_frame(8, 8))))
    manager = _compile_image_pipeline(dynamic_session)
    for height, width in ((4, 7), (11, 5)):
        image = _camera_frame(height, width)
        with capture_context():
            expected = contextual_features.__wrapped__(preprocess.__wrapped__(image))
        actual = manager({"preprocess/image": image})
        torch.testing.assert_close(actual["contextual_features/result"], expected)


@pytest.mark.parametrize("compile_in_inference_mode", [False, True])
def test_validation_restores_inference_buffers(
    dynamic_session, compile_in_inference_mode
):
    """Captured inference buffers can be restored regardless of compilation context."""
    with torch.inference_mode():
        bias = torch.tensor([3.0])

        @annotate.method(export_with="pt2")
        def encode(x):
            result = x + bias
            return result

        encode(
            TensorSemantics(
                "x",
                torch.ones(2, 8),
                axes=[None, Axis(variable=Variable("length", min=4, max=12))],
            )
        )
    leapp.stop()
    with torch.inference_mode(compile_in_inference_mode):
        assert leapp.compile_graph(visualize=False, validate=True, strict=True) == {
            "encode": True
        }
