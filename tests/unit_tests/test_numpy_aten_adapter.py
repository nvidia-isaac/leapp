# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NumPy parity and runtime dependence for the automatic ATen capture."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

import leapp
from leapp.leapp_graph.datatypes import TracedNpArray
from leapp.leapp_graph.datatypes.numpy.patching import NumpyPatchBackend
from leapp.leapp_graph.datatypes.torch.patching import TorchPatchBackend
from leapp.leapp_graph.traced_node import TracedTensorNode


OPERATIONS = [
    pytest.param(np.linalg.inv, id="inverse"),
    pytest.param(lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True), id="norm"),
    pytest.param(
        lambda x: np.cross(x, np.array([1.0, 2.0, 3.0], dtype=np.float32)), id="cross"
    ),
    pytest.param(lambda x: np.where(np.isclose(x, 3.0), x, -x), id="isclose_where"),
    pytest.param(np.sinc, id="sinc"),
    pytest.param(lambda x: np.diff(x, axis=-1), id="diff"),
    pytest.param(lambda x: np.quantile(x, 0.25, axis=1), id="quantile"),
    pytest.param(np.cbrt, id="cbrt"),
    pytest.param(lambda x: np.pad(x, ((1, 0), (0, 2)), constant_values=-2), id="pad"),
    pytest.param(lambda x: np.nanmean(x, axis=-1, keepdims=True), id="nanmean"),
    pytest.param(lambda x: np.sort(x, axis=None), id="sort"),
    pytest.param(lambda x: np.argsort(x, axis=0), id="argsort"),
]


def capture(fn, *inputs):
    ctx = TracedTensorNode(name="numpy_aten", node_index=0)
    result = fn(
        *(ctx.create_input(x.copy(), name=f"x{i}") for i, x in enumerate(inputs))
    )
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    assert all(isinstance(x, TracedNpArray) for x in outputs)
    ctx.compile_trace({f"y{i}": x for i, x in enumerate(outputs)})
    return ctx.m


@pytest.mark.parametrize("fn", OPERATIONS)
@pytest.mark.parametrize("backend", ["fx", "jit", "pt2"])
def test_changed_input_replay(fn, backend):
    x = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]], dtype=np.float32)
    model = capture(fn, x)
    if backend == "jit":
        model = torch.jit.script(model)
    elif backend == "pt2":
        model = torch.export.export(
            model, (torch.from_numpy(x),), strict=False
        ).module()
    corner = x.copy()
    corner[0, 0] += 4
    for sample in (x, x + 2, corner, -x):
        torch.testing.assert_close(
            model(torch.from_numpy(sample)), torch.from_numpy(fn(sample))
        )


@pytest.mark.parametrize("equal_nan", [False, True])
def test_isclose_special_values_and_broadcasting(equal_nan):
    fn = lambda x, y: np.isclose(x, y, rtol=1e-4, atol=1e-6, equal_nan=equal_nan)
    x = np.array([[0.0, 1.00009, np.nan, np.inf, -np.inf]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, np.inf, np.inf], dtype=np.float64)
    model = capture(fn, x, y)
    changed = np.array([[1e-5, 1.0002, 3.0, -np.inf, -np.inf]], dtype=np.float64)
    for a, b in ((x, y), (changed, y), (changed, changed[0])):
        torch.testing.assert_close(
            model(torch.from_numpy(a), torch.from_numpy(b)), torch.from_numpy(fn(a, b))
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cbrt_signed_zero_and_special_values(dtype):
    x = np.array([-np.inf, -8.0, -0.0, 0.0, 27.0, np.inf, np.nan], dtype=dtype)
    model = capture(np.cbrt, x)
    for sample in (x, -x):
        actual = model(torch.from_numpy(sample))
        expected = torch.from_numpy(np.cbrt(sample))
        torch.testing.assert_close(actual, expected, equal_nan=True)
        torch.testing.assert_close(
            torch.signbit(actual[2:4]), torch.signbit(expected[2:4])
        )


@pytest.mark.parametrize("fn", [np.mean, np.nanmean])
@pytest.mark.parametrize("backend", ["fx", "jit", "pt2"])
def test_integer_reduction_uses_numpy_promotion(fn, backend):
    operation = lambda x: fn(x, axis=-1, keepdims=True)
    x = np.arange(6, dtype=np.int32).reshape(2, 3)
    model = capture(operation, x)
    if backend == "jit":
        model = torch.jit.script(model)
    elif backend == "pt2":
        model = torch.export.export(
            model, (torch.from_numpy(x),), strict=False
        ).module()
    for sample in (x, x + 3):
        torch.testing.assert_close(
            model(torch.from_numpy(sample)), torch.from_numpy(operation(sample))
        )


@pytest.mark.parametrize("axis", [None, -1, (0, 1), ()])
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_nanmean_nan_slices_and_explicit_dtype(axis):
    fn = lambda x: np.nanmean(x, axis=axis, dtype=np.float64, keepdims=True)
    x = np.array([[np.nan, np.nan], [1.0, np.nan]], dtype=np.float32)
    model = capture(fn, x)
    changed = np.array([[2.0, 3.0], [np.nan, 4.0]], dtype=np.float32)
    for sample in (x, changed, changed + 2):
        torch.testing.assert_close(
            model(torch.from_numpy(sample)),
            torch.from_numpy(fn(sample)),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: np.sort(x, stable=True),
        lambda x: np.argsort(x, kind="mergesort"),
        lambda x: np.pad(x, 1, mode="reflect"),
        lambda x: np.pad(x, 1, constant_values=(1, 2)),
    ],
)
def test_override_rejects_unimplemented_signatures(fn):
    ctx = TracedTensorNode(name="unsupported_override", node_index=0)
    x = ctx.create_input(np.array([3.0, 1.0, 2.0]), name="x")
    before = tuple(ctx.graph.nodes)
    with pytest.raises(NotImplementedError):
        fn(x)
    assert tuple(ctx.graph.nodes) == before


def test_structured_outputs_and_constant_snapshot():
    constant = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fn = lambda x: (np.cross(x, constant), *np.gradient(x))
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    model = capture(fn, x)
    expected = fn(x + 2)
    constant[:] = 100
    for actual, reference in zip(model(torch.from_numpy(x + 2)), expected):
        torch.testing.assert_close(actual, torch.from_numpy(reference))


@pytest.mark.parametrize(
    "fn, message",
    [
        (np.fft.fft, "dtype"),
        (np.spacing, "no torch._numpy lowering"),
        (lambda x: np.allclose(x, x), "ATen capture/parity check failed"),
    ],
)
def test_unsupported_capture_is_explicit_and_atomic(fn, message):
    ctx = TracedTensorNode(name="unsupported", node_index=0)
    x = ctx.create_input(np.ones(3, dtype=np.float32), name="x")
    before = tuple(ctx.graph.nodes)
    with pytest.raises(NotImplementedError, match=message):
        fn(x)
    assert tuple(ctx.graph.nodes) == before


@pytest.mark.parametrize("positional", [False, True])
def test_reject_out_before_mutating_storage(positional):
    ctx = TracedTensorNode(name="out", node_index=0)
    x = ctx.create_input(np.array([2.0, 1.0, 3.0]), name="x")
    out = np.zeros(2)
    with pytest.raises(NotImplementedError, match="mutation"):
        if positional:
            np.take(x, [0, 2], None, out)
        else:
            np.take(x, [0, 2], out=out)
    np.testing.assert_array_equal(out, np.zeros(2))


def test_reject_mutating_call_before_changing_input():
    ctx = TracedTensorNode(name="mutation", node_index=0)
    original = np.array([np.nan, 1.0], dtype=np.float32)
    x = ctx.create_input(original, name="x")
    with pytest.raises(NotImplementedError, match="ATen capture/parity check failed"):
        np.nan_to_num(x, copy=False)
    np.testing.assert_array_equal(original, [np.nan, 1.0])


def test_inactive_carrier_keeps_native_behavior():
    ctx = TracedTensorNode(name="transit", node_index=0)
    x = ctx.create_input(np.array([-8.0, 1.0]), name="x")
    ctx.compile_trace({"y": x})
    result = np.cbrt(x)
    assert type(result) is np.ndarray
    np.testing.assert_array_equal(result, [-2.0, 1.0])


def test_eager_view_keeps_numpy_storage():
    ctx = TracedTensorNode(name="view", node_index=0)
    x = ctx.create_input(np.arange(3, dtype=np.float32), name="x")
    y = np.broadcast_to(x, (2, 3))
    assert np.shares_memory(x.data, y.data)
    assert not y.flags.writeable


def test_conversion_patches_during_capture():
    numpy_patch, torch_patch = NumpyPatchBackend(), TorchPatchBackend()
    try:
        numpy_patch.install()
        torch_patch.install()
        fn = lambda x: np.linalg.norm(x, axis=-1)
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        model = capture(fn, x)
    finally:
        torch_patch.uninstall()
        numpy_patch.uninstall()
    torch.testing.assert_close(
        model(torch.from_numpy(x + 2)), torch.from_numpy(fn(x + 2))
    )


def test_public_lifecycle_export_and_replay(tmp_path):
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    leapp.start(name="numpy_aten", save_path=str(tmp_path))
    try:
        traced = leapp.annotate.input_tensors("norm", {"x": x})
        result = np.linalg.norm(traced, axis=-1)
        leapp.annotate.output_tensors("norm", {"y": result}, export_with="jit")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False)
    model = torch.jit.load(str(next(tmp_path.rglob("norm.pt"))))
    torch.testing.assert_close(
        model(torch.from_numpy(x + 2)), torch.from_numpy(np.linalg.norm(x + 2, axis=-1))
    )


@pytest.mark.parametrize("fn", OPERATIONS[1:4])
def test_onnx_changed_input_replay(fn):
    x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    model = capture(fn, x)
    exported = torch.onnx.export(model, (torch.from_numpy(x),), dynamo=True)
    session = ort.InferenceSession(
        exported.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for sample in (x, x + 2, -x):
        actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
        expected = fn(sample)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
