# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reverse reads retain NumPy slice semantics in exported tensor graphs."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from leapp.leapp_graph.datatypes import TracedNpArray
from leapp.leapp_graph.traced_node import TracedTensorNode


CASES = [
    pytest.param(slice(None, None, -1), id="full_reverse"),
    pytest.param(slice(None, None, -2), id="negative_step"),
    pytest.param(slice(3, 0, -2), id="bounded_step"),
    pytest.param(slice(-2, None, -1), id="negative_start"),
    pytest.param(slice(None, -1, -1), id="explicit_minus_one_stop_is_empty"),
    pytest.param(slice(100, -100, -3), id="clipped_bounds"),
    pytest.param(slice(-100, 100, -1), id="empty_clipped_bounds"),
    pytest.param(slice(np.int64(3), None, np.int64(-2)), id="numpy_integer_bounds"),
    pytest.param(
        (slice(None, None, -1), slice(1, 5, 2), slice(None, None, -2)),
        id="multiple_axes",
    ),
    pytest.param(
        (1, slice(None, None, -2), slice(None, None, -1)), id="integer_before_slices"
    ),
    pytest.param((slice(None, None, -1), 2, slice(None)), id="integer_after_slice"),
    pytest.param((Ellipsis, slice(None, None, -1)), id="ellipsis"),
    pytest.param((None, Ellipsis, slice(5, 0, -2), None), id="new_axes"),
    pytest.param((1, None, slice(None, None, -1), Ellipsis, None), id="mixed_basic"),
]


def capture(key, x):
    ctx = TracedTensorNode(name="reverse_slice", node_index=0)
    traced = ctx.create_input(x.copy(), name="x")
    result = traced[key]
    assert isinstance(result, TracedNpArray)
    ctx.compile_trace({"y": result})
    return ctx.m


@pytest.mark.parametrize("key", CASES)
@pytest.mark.parametrize("backend", ["fx", "jit", "pt2"])
def test_reverse_slice_changed_input_replay(key, backend):
    x = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    model = capture(key, x)
    if backend == "jit":
        model = torch.jit.script(model)
    elif backend == "pt2":
        model = torch.export.export(
            model, (torch.from_numpy(x),), strict=False
        ).module()
    for sample in (x, x * -2 + 7, np.roll(x, 1, axis=-1)):
        expected = torch.from_numpy(np.array(sample[key], copy=True))
        torch.testing.assert_close(model(torch.from_numpy(sample)), expected)


@pytest.mark.parametrize("key", CASES[::3])
def test_reverse_slice_onnx_replay(key):
    x = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    model = capture(key, x)
    exported = torch.onnx.export(model, (torch.from_numpy(x),), dynamo=True)
    session = ort.InferenceSession(
        exported.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for sample in (x, x * -2 + 7):
        actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
        expected = sample[key]
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("size", [0, 1, 7])
@pytest.mark.parametrize("key", [slice(None, None, -1), slice(None, None, -3)])
def test_empty_and_singleton_dimensions(size, key):
    x = np.arange(size, dtype=np.int64)
    model = capture(key, x)
    torch.testing.assert_close(
        model(torch.from_numpy(x + 3)),
        torch.from_numpy(np.array((x + 3)[key], copy=True)),
    )


def test_lowering_and_chained_numpy_operations():
    x = np.arange(30, dtype=np.float32).reshape(5, 6)
    full = capture((slice(None, None, -1), Ellipsis), x)
    assert any(node.target is torch.flip for node in full.graph.nodes)
    partial = capture((Ellipsis, slice(4, 0, -2)), x)
    assert any(node.target is torch.index_select for node in partial.graph.nodes)

    ctx = TracedTensorNode(name="reverse_chain", node_index=0)
    traced = ctx.create_input(x.copy(), name="x")
    result = np.sin(traced[::-1][:, ::-2]) + 1
    ctx.compile_trace({"y": result})
    sample = x * -2 + 7
    torch.testing.assert_close(
        ctx.m(torch.from_numpy(sample)),
        torch.from_numpy(np.sin(sample[::-1][:, ::-2]) + 1),
    )


def test_eager_and_inactive_reverse_views():
    ctx = TracedTensorNode(name="reverse_view", node_index=0)
    x = ctx.create_input(np.arange(8, dtype=np.float32), name="x")
    y = x[::-2]
    assert np.shares_memory(x.data, y.data)
    assert y.strides == x.data[::-2].strides
    ctx.compile_trace({"y": y})
    inactive = x[::-1]
    assert type(inactive) is np.ndarray
    np.testing.assert_array_equal(inactive, x.data[::-1])


def test_unsupported_combinations_fail_during_capture():
    ctx = TracedTensorNode(name="reverse_unsupported", node_index=0)
    x = ctx.create_input(np.arange(12, dtype=np.float32).reshape(3, 4), name="x")
    bound = ctx.create_input(np.array(2, dtype=np.int64), name="bound")
    for key in ((slice(None, None, -1), [1, 3]), (slice(bound, None, -1), slice(None))):
        before = tuple(ctx.graph.nodes)
        with pytest.raises(NotImplementedError, match="static basic indices"):
            x[key]
        assert tuple(ctx.graph.nodes) == before


def test_zero_step_preserves_numpy_error():
    ctx = TracedTensorNode(name="zero_step", node_index=0)
    x = ctx.create_input(np.arange(4, dtype=np.float32), name="x")
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        x[::0]
