#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for TracedNpArray operations."""

import pathlib
import tempfile
import unittest
import warnings

import pytest

import numpy as np
import onnxruntime as ort
import torch

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedNpArray
from leapp.leapp_graph.datatypes.numpy import aten_adapter
from leapp.leapp_graph.datatypes.patching import TracingPatcher

from tests.unit_tests.export_format_validation import (
    verify_exported_program,
    verify_onnx_dynamo,
)

warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
warnings.filterwarnings("ignore", message=".*on empty non-base types in `__init__`.*")
warnings.filterwarnings("ignore", message=".*feature will be removed.*")
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")


def _assert_numpy_output(actual, expected, atol=1e-5):
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list))
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_numpy_output(value, reference, atol)
        return
    torch.testing.assert_close(
        actual,
        torch.from_numpy(np.array(expected, copy=True)),
        rtol=1e-7,
        atol=atol,
        equal_nan=True,
    )


def _capture_numpy(fn, *inputs):
    expected = fn(*(x.copy() for x in inputs))
    ctx = TracedTensorNode(name="numpy_operation", node_index=0)
    result = fn(
        *(ctx.create_input(x.copy(), name=f"x{i}") for i, x in enumerate(inputs))
    )
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    assert all(isinstance(x, TracedNpArray) for x in outputs)
    references = expected if isinstance(expected, (tuple, list)) else (expected,)
    assert len(outputs) == len(references)
    for output, reference in zip(outputs, references):
        _assert_numpy_output(
            torch.from_numpy(np.array(output.data, copy=True)), reference
        )
    ctx.compile_trace({f"y{i}": x for i, x in enumerate(outputs)})
    return ctx.m


class TestTracedNpArray(unittest.TestCase):
    """Test TracedNpArray operations."""

    # ==================== Basic Property Tests ====================

    def test_array_properties(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(60, dtype=np.int32).reshape(3, 4, 5), name="x")
        self.assertEqual(x.shape, (3, 4, 5))
        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.dtype, np.int32)
        self.assertEqual(len(x), 3)
        self.assertEqual(x.size, 60)

    def test_T_property(self):
        """Test .T transpose property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4), name="x")
        result = x.T
        self.assertEqual(result.shape, (4, 3))
        self.assertIsInstance(result, TracedNpArray)

    # ==================== Arithmetic Operation Tests ====================

    def test_rtruediv(self):
        """Test reverse division."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 4.0]), name="x")
        y = 8 / x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([8.0, 4.0, 2.0]))

    def test_abs(self):
        """Test absolute value."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([-1.0, 2.0, -3.0]), name="x")
        y = abs(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_matmul(self):
        """Test matrix multiplication operator."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        b = np.array([4.0, 5.0, 6.0])
        y = x @ b
        self.assertIsInstance(y, (TracedNpArray, np.floating, float))

    # ==================== Comparison Operator Tests ====================

    def test_comparison_gt(self):
        """Test greater than."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x > 1.5
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_equal(y, np.array([False, True, True]))

    def test_comparison_lt(self):
        """Test less than."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x < 2.5
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_equal(y, np.array([True, True, False]))

    def test_comparison_eq(self):
        """Test equality."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x == 2.0
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_equal(y, np.array([False, True, False]))

    # ==================== NumPy Ufunc Tests ====================

    def test_np_sin(self):
        """Test np.sin ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([0.0, np.pi / 2, np.pi]), name="x")
        y = np.sin(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([0.0, 1.0, 0.0]), decimal=5)

    def test_np_cos(self):
        """Test np.cos ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([0.0, np.pi / 2, np.pi]), name="x")
        y = np.cos(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 0.0, -1.0]), decimal=5)

    def test_np_exp(self):
        """Test np.exp ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([0.0, 1.0, 2.0]), name="x")
        y = np.exp(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.exp(np.array([0.0, 1.0, 2.0])))

    def test_np_log(self):
        """Test np.log ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, np.e, np.e**2]), name="x")
        y = np.log(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([0.0, 1.0, 2.0]))

    def test_np_sqrt(self):
        """Test np.sqrt ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 4.0, 9.0]), name="x")
        y = np.sqrt(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_np_abs(self):
        """Test np.abs ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([-1.0, 2.0, -3.0]), name="x")
        y = np.abs(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    # ==================== NumPy Function Tests ====================

    def test_np_concatenate(self):
        """Test np.concatenate function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        b = np.array([4.0, 5.0, 6.0])
        y = np.concatenate([x, b])
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (6,))

    def test_np_stack(self):
        """Test np.stack function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        b = np.array([4.0, 5.0, 6.0])
        y = np.stack([x, b])
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (2, 3))

    def test_np_clip(self):
        """Test np.clip function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 5.0, 10.0]), name="x")
        y = np.clip(x, 2.0, 8.0)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 5.0, 8.0]))

    def test_np_where(self):
        """Test np.where function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        condition = np.array([True, False, True])
        y = np.where(condition, x, 0.0)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 0.0, 3.0]))

    def test_np_nonzero(self):
        """Test np.nonzero function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([0.0, 1.0, 0.0, 2.0, 3.0]), name="x")
        y = np.nonzero(x)
        # nonzero returns a tuple of arrays
        self.assertIsInstance(y, tuple)
        np.testing.assert_array_equal(y[0], np.array([1, 3, 4]))

    # ==================== Indexing Tests ====================

    def test_getitem_single_index(self):
        """Test single element indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x[1]
        self.assertAlmostEqual(float(y), 2.0)

    def test_getitem_slice(self):
        """Test slice indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        y = x[1:4]
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 3.0, 4.0]))

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x[-1]
        self.assertAlmostEqual(float(y), 3.0)

    def test_getitem_step(self):
        """Test step indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        y = x[::2]
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 3.0, 5.0]))

    def test_getitem_2d(self):
        """Test 2D indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(12).reshape(3, 4).astype(float), name="x")
        y = x[1, 2]
        self.assertAlmostEqual(float(y), 6.0)

    def test_getitem_2d_slice(self):
        """Test 2D slice indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(12).reshape(3, 4).astype(float), name="x")
        y = x[1:3, 1:3]
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (2, 2))

    # ==================== In-place Operation Tests ====================

    def test_inplace_add(self):
        """Test in-place addition."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        x += 1
        self.assertIsInstance(x, TracedNpArray)
        np.testing.assert_array_almost_equal(x, np.array([2.0, 3.0, 4.0]))

    def test_inplace_sub(self):
        """Test in-place subtraction."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        x -= 1
        self.assertIsInstance(x, TracedNpArray)
        np.testing.assert_array_almost_equal(x, np.array([0.0, 1.0, 2.0]))

    def test_inplace_mul(self):
        """Test in-place multiplication."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        x *= 2
        self.assertIsInstance(x, TracedNpArray)
        np.testing.assert_array_almost_equal(x, np.array([2.0, 4.0, 6.0]))

    def test_inplace_div(self):
        """Test in-place division."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([2.0, 4.0, 6.0]), name="x")
        x /= 2
        self.assertIsInstance(x, TracedNpArray)
        np.testing.assert_array_almost_equal(x, np.array([1.0, 2.0, 3.0]))

    # ==================== Method Tests ====================

    def test_method_sum(self):
        """Test .sum() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x.sum()
        self.assertAlmostEqual(float(y), 6.0)

    def test_method_mean(self):
        """Test .mean() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x.mean()
        self.assertAlmostEqual(float(y), 2.0)

    def test_method_reshape(self):
        """Test .reshape() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), name="x")
        y = x.reshape(2, 3)
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (2, 3))

    def test_method_transpose(self):
        """Test .transpose() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4), name="x")
        y = x.transpose()
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (4, 3))

    def test_method_copy(self):
        """Test .copy() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x.copy()
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y.data, x.data)

    def test_method_astype(self):
        """Test .astype() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x.astype(np.float32)
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.dtype, np.float32)
        np.testing.assert_array_almost_equal(y.data, x.data)

    def test_method_astype_int(self):
        """Test .astype() to integer type."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.5, 2.7, 3.2]), name="x")
        y = x.astype(np.int32)
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.dtype, np.int32)
        np.testing.assert_array_equal(y, np.array([1, 2, 3]))

    def test_method_view(self):
        """Test .view() method returns TracedNpArray with same data."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0]), name="x")
        # View as same dtype should return TracedNpArray
        y = x.view(np.float64)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y.data, x.data)

    def test_method_copy_compiled(self):
        """Test .copy() method compiles correctly."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x.copy() * 2  # Use copied array in computation

        ctx.compile_trace({"output": y})

        # Test compiled graph works
        input_tensor = torch.tensor([4.0, 5.0, 6.0])
        expected = input_tensor.clone() * 2  # clone is torch equivalent of copy
        output = ctx.m(input_tensor)
        self.assertTrue(torch.allclose(output, expected))

    # ==================== Graph Building Tests ====================

    def test_graph_basic_operations(self):
        """Test that basic operations create an FX graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x + 1
        z = y * 2

        # Compile and verify graph exists
        ctx.compile_trace({"output": z})
        self.assertIsNotNone(ctx.m)
        self.assertIsNotNone(ctx.graph)
        self.assertGreater(len(list(ctx.graph.nodes)), 0)

    def test_graph_multiple_inputs(self):
        """Test graph with multiple inputs."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = ctx.create_input(np.array([4.0, 5.0, 6.0]), name="y")
        z = x + y

        ctx.compile_trace({"output": z})
        self.assertIsNotNone(ctx.m)

    # ==================== Operations After Compile Tests ====================

    def test_operations_after_compile_return_native_arrays(self):
        """A value derived from a finished node is new data, so it stays native."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x * 2

        ctx.compile_trace({"y": y})

        # After compilation, is_tracing should be False
        self.assertFalse(ctx.is_tracing)

        patcher = TracingPatcher()
        patcher.install()
        try:
            derived = [
                x + 1,
                x.copy(),
                x.astype(np.float32),
                np.array(x),
                np.cbrt(x),
            ]
            # These alias the boundary value rather than producing new data.
            aliases = [np.asarray(x), np.asanyarray(x)]
        finally:
            patcher.uninstall()

        for result in derived:
            self.assertNotIsInstance(result, TracedNpArray)
            self.assertIsInstance(result, np.ndarray)

        for result in aliases:
            self.assertIs(result, x)

        np.testing.assert_array_equal(derived[-1], np.cbrt(x.data))

        # The boundary value itself is untouched and still carries its node.
        self.assertIsInstance(x, TracedNpArray)
        self.assertIs(x.context_obj, ctx)

    # ==================== Conversion Patch Tests ====================

    def test_conversion_copy_property(self):
        """Test that np.array copies and np.asarray aliases, in eager and graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")

        patcher = TracingPatcher()
        patcher.install()
        try:
            aliased = np.asarray(x)
            never_copy = np.asarray(x, copy=False)
            copied = np.array(x)
            forced = np.asarray(x, copy=True)
        finally:
            patcher.uninstall()

        # np.asarray copies only when needed, so an unchanged request aliases
        self.assertIs(aliased, x)
        self.assertIs(never_copy, x)

        # np.array copies by default, and copy=True forces one either way
        for result in (copied, forced):
            self.assertIsInstance(result, TracedNpArray)
            self.assertIsNot(result, x)
            self.assertFalse(
                np.shares_memory(result.view(np.ndarray), x.view(np.ndarray))
            )
            self.assertTrue(np.array_equal(result.view(np.ndarray), [1.0, 2.0, 3.0]))

        ctx.compile_trace({"aliased": aliased, "copied": copied})

        # Each copy is recorded, while the alias reuses its source proxy
        targets = [node.target for node in ctx.graph.nodes]
        self.assertEqual(targets.count(torch.clone), 1)

        alias_out, copy_out = ctx.m(torch.tensor([1.0, 2.0, 3.0]))
        self.assertFalse(alias_out.data_ptr() == copy_out.data_ptr())

    def test_conversion_dtype_swap_property(self):
        """Test that a dtype swap casts eagerly and is recorded in the graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")

        patcher = TracingPatcher()
        patcher.install()
        try:
            # dtype is accepted positionally as well as by keyword
            positional = np.asarray(x, np.float32)
            keyword = np.asarray(x, dtype=np.float32)
            widened = np.asarray(x, dtype=np.int64)
        finally:
            patcher.uninstall()

        for result, expected in (
            (positional, np.float32),
            (keyword, np.float32),
            (widened, np.int64),
        ):
            self.assertIsInstance(result, TracedNpArray)
            self.assertEqual(result.dtype, expected)

        ctx.compile_trace({"positional": positional, "widened": widened})

        # Each output cast is a real node rather than a reuse of the source proxy
        casts = [node for node in ctx.graph.nodes if node.target == "to"]
        self.assertEqual([node.args[1] for node in casts], [torch.float32, torch.int64])

        # A float64 input proves the recorded cast changes dtype rather than
        # coincidentally matching torch's float32 default.
        float_out, int_out = ctx.m(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
        self.assertEqual(float_out.dtype, torch.float32)
        self.assertEqual(int_out.dtype, torch.int64)

    def test_getitem_after_compile_returns_native_array(self):
        """Inactive indexing produces new data, so it stays native."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        # Use slice to get array output (scalar indexing returns np.float64, not array)
        y = x[0:2]

        ctx.compile_trace({"y": y})

        z = x[2:4]
        self.assertNotIsInstance(z, TracedNpArray)
        self.assertIsInstance(z, np.ndarray)
        self.assertIsInstance(x, TracedNpArray)

    # ==================== Context Validation Tests ====================

    def test_multiple_contexts_raises_error(self):
        """Test that mixing contexts raises an error."""
        ctx1 = TracedTensorNode(name="test1", node_index=0)
        ctx2 = TracedTensorNode(name="test2", node_index=1)

        x = ctx1.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = ctx2.create_input(np.array([4.0, 5.0, 6.0]), name="y")

        # Mixing arrays from different contexts should raise
        with self.assertRaises(Exception):
            _ = x + y

    # ==================== Chained Operations Tests ====================

    def test_chained_operations(self):
        """Test chained operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")

        result = ((x + 1) * 2 - 3) / 4

        self.assertIsInstance(result, TracedNpArray)
        expected = ((np.array([1.0, 2.0, 3.0]) + 1) * 2 - 3) / 4
        np.testing.assert_array_almost_equal(result, expected)

    def test_complex_computation(self):
        """Test complex computation chain."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")

        # Multiple operations
        y = np.sin(x)
        z = np.cos(x)
        w = y**2 + z**2  # Should be ~1.0

        self.assertIsInstance(w, TracedNpArray)
        np.testing.assert_array_almost_equal(w, np.ones(3), decimal=5)


# Suppress TracerWarning from setitem tests where torch.tensor/torch.as_tensor constants
# are baked into the trace (e.g. "torch.as_tensor results are registered as constants in the trace").
# This is expected behavior for constant index/value assignments like x[0] = 10.0.
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
class TestCompiledGraphExecution(unittest.TestCase):
    """Test that traced numpy operations compile and execute correctly.

    These tests verify that:
    1. The FX graph module produces correct output
    2. TorchScript export works and produces correct output
    3. ONNX export works and produces correct output
    4. ExportedProgram export works and produces correct output
    """

    def _test_operation(
        self,
        op_name,
        np_func,
        input_data,
        expected_output,
        atol=1e-5,
        *,
        backends=("fx", "jit", "onnx", "pt2"),
        replay_inputs=None,
    ):
        """Capture once; check eager and changed-input replay without dtype/shape coercion."""
        graph_module = _capture_numpy(np_func, input_data)
        input_tensor = torch.from_numpy(input_data.copy())
        samples = (input_data,) + tuple(
            (input_data + 2,) if replay_inputs is None else replay_inputs
        )
        for backend in backends:
            with self.subTest(operation=op_name, backend=backend):
                model = graph_module
                if backend == "jit":
                    model = torch.jit.script(graph_module)
                elif backend == "jit-trace":
                    model = torch.jit.trace(graph_module, (input_tensor.clone(),))
                elif backend == "pt2":
                    model = torch.export.export(
                        graph_module, (input_tensor.clone(),), strict=False
                    ).module()
                elif backend in ("onnx", "onnx-dynamo"):
                    if backend == "onnx-dynamo":
                        exported = torch.onnx.export(
                            graph_module, (input_tensor.clone(),), dynamo=True
                        )
                        session = ort.InferenceSession(
                            exported.model_proto.SerializeToString(),
                            providers=["CPUExecutionProvider"],
                        )
                    else:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            path = pathlib.Path(tmpdir) / f"{op_name}.onnx"
                            torch.onnx.export(
                                graph_module,
                                (input_tensor.clone(),),
                                path,
                                dynamo=False,
                                opset_version=17,
                                input_names=["input"],
                                output_names=["output"],
                            )
                            session = ort.InferenceSession(
                                str(path), providers=["CPUExecutionProvider"]
                            )
                for index, sample in enumerate(samples):
                    with self.subTest(sample=index):
                        reference = (
                            expected_output if index == 0 else np_func(sample.copy())
                        )
                        if backend in ("onnx", "onnx-dynamo"):
                            actual = torch.from_numpy(
                                session.run(
                                    None, {session.get_inputs()[0].name: sample.copy()}
                                )[0]
                            )
                        else:
                            actual = model(torch.from_numpy(sample.copy()))
                        _assert_numpy_output(actual, reference, atol)

    def test_numpy_function_replay(self):
        x = np.array(
            [[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]], dtype=np.float32
        )
        corner = x.copy()
        corner[0, 0] += 4
        operations = {
            "inverse": np.linalg.inv,
            "det": np.linalg.det,
            "slogdet": lambda x: tuple(np.linalg.slogdet(x)),
            "cholesky": np.linalg.cholesky,
            "solve": lambda x: np.linalg.solve(x, np.ones(3, dtype=np.float32)),
            "matrix_power": lambda x: np.linalg.matrix_power(x, 3),
            "svd_values": lambda x: np.linalg.svd(x, compute_uv=False),
            "eigvalsh": np.linalg.eigvalsh,
            "pinv": np.linalg.pinv,
            "norm": lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True),
            "cross": lambda x: np.cross(x, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            "isclose_where": lambda x: np.where(np.isclose(x, 3.0), x, -x),
            "sinc": np.sinc,
            "degrees": np.degrees,
            "radians": np.radians,
            "logaddexp": lambda x: np.logaddexp(x, x + 1),
            "heaviside": lambda x: np.heaviside(x - 4, 0.5),
            "copysign": lambda x: np.copysign(x, -1.0),
            "gradient": lambda x: tuple(np.gradient(x)),
            "repeat": lambda x: np.repeat(x, 2, axis=1),
            "tile": lambda x: np.tile(x, (2, 1)),
            "take": lambda x: np.take(x, [0, 2], axis=1),
            "take_along_axis": lambda x: np.take_along_axis(
                x, np.array([[2, 0], [0, 1], [1, 2]]), axis=1
            ),
            "broadcast_to": lambda x: np.broadcast_to(x, (2, 3, 3)),
            "median": lambda x: np.median(x, axis=1),
            "ptp": lambda x: np.ptp(x, axis=1),
            "unique": np.unique,
            "diff": lambda x: np.diff(x, axis=-1),
            "quantile": lambda x: np.quantile(x, 0.25, axis=1),
            "cbrt": np.cbrt,
            "pad": lambda x: np.pad(x, ((1, 0), (0, 2)), constant_values=-2),
            "nanmean": lambda x: np.nanmean(x, axis=-1, keepdims=True),
        }
        for name, fn in operations.items():
            backends = ("fx", "jit", "jit-trace", "pt2")
            if name in ("norm", "cross", "isclose_where"):
                backends += ("onnx-dynamo",)
            with self.subTest(operation=name):
                self._test_operation(
                    name,
                    fn,
                    x,
                    fn(x),
                    backends=backends,
                    # Cholesky requires positive-definite inputs.
                    replay_inputs=(x + 2, corner)
                    if name == "cholesky"
                    else (x + 2, corner, -x),
                )

    def test_integer_mean(self):
        x = np.arange(6, dtype=np.int32).reshape(2, 3)
        for fn in (np.mean, np.nanmean):
            operation = lambda x: fn(x, axis=-1, keepdims=True)
            self._test_operation(
                fn.__name__,
                operation,
                x,
                operation(x),
                backends=("fx", "jit", "pt2"),
                replay_inputs=(x + 3,),
            )

    def test_method_and_arithmetic_chains(self):
        x = np.array([2.0, 3.0, 4.0])
        operations = {
            "scalar_arithmetic": lambda x: (((x + 1) - 2) * 3 / 4) ** 5,
            "reverse_arithmetic": lambda x: 5 ** (4 / (3 * (2 - (1 + x)))),
            "array_arithmetic": lambda x: (
                x + 2 * np.array([4.0, 5.0, 6.0]) - 0.5 * np.array([7.0, 8.0, 9.0])
            ),
            "sum_method": lambda x: x.sum(),
            "mean_method": lambda x: x.mean(),
            "reshape_method": lambda x: (x.reshape(1, 3) * 2).reshape(3),
            "reshape_function": lambda x: np.reshape(np.reshape(x, (1, 3)) * 2, (3,)),
        }
        for name, fn in operations.items():
            self._test_operation(name, fn, x, fn(x))

    def test_add_scalar(self):
        """Test: x + 1"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data + 1
        self._test_operation("add_scalar", lambda x: x + 1, input_data, expected)

    def test_add_array(self):
        """Test: x + y (array + array)"""
        input_data = np.array([1.0, 2.0, 3.0])
        other = np.array([4.0, 5.0, 6.0])
        expected = input_data + other
        self._test_operation("add_array", lambda x: x + other, input_data, expected)

    def test_sub_scalar(self):
        """Test: x - 2"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data - 2
        self._test_operation("sub_scalar", lambda x: x - 2, input_data, expected)

    def test_mul_scalar(self):
        """Test: x * 3"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data * 3
        self._test_operation("mul_scalar", lambda x: x * 3, input_data, expected)

    def test_mul_np_float32(self):
        """Test: x * np.float32(2.0) — numpy scalars must become Python numbers."""
        input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = input_data * np.float32(2.0)
        self._test_operation(
            "mul_np_float32", lambda x: x * np.float32(2.0), input_data, expected
        )

    def test_mul_np_float64(self):
        """Test: x * np.float64(2.0) — numpy scalars must not leak into the FX graph."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data * np.float64(2.0)
        self._test_operation(
            "mul_np_float64", lambda x: x * np.float64(2.0), input_data, expected
        )

    def test_div_norm(self):
        """Test: normalization retains its runtime-dependent denominator."""
        input_data = np.array([3.0, 4.0], dtype=np.float32)
        expected = input_data / np.linalg.norm(input_data)
        self._test_operation(
            "div_norm", lambda x: x / np.linalg.norm(x), input_data, expected
        )

    def test_div_scalar(self):
        """Test: x / 2"""
        input_data = np.array([2.0, 4.0, 6.0])
        expected = input_data / 2
        self._test_operation("div_scalar", lambda x: x / 2, input_data, expected)

    def test_pow_scalar(self):
        """Test: x ** 2"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data**2
        self._test_operation("pow_scalar", lambda x: x**2, input_data, expected)

    def test_neg(self):
        """Test: -x"""
        input_data = np.array([1.0, -2.0, 3.0])
        expected = -input_data
        self._test_operation("neg", lambda x: -x, input_data, expected)

    # ==================== Reverse Arithmetic Operations ====================

    def test_radd(self):
        """Test: 1 + x"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = 1 + input_data
        self._test_operation("radd", lambda x: 1 + x, input_data, expected)

    def test_rsub(self):
        """Test: 10 - x"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = 10 - input_data
        self._test_operation("rsub", lambda x: 10 - x, input_data, expected)

    def test_rmul(self):
        """Test: 2 * x"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = 2 * input_data
        self._test_operation("rmul", lambda x: 2 * x, input_data, expected)

    def test_rdiv(self):
        """Test: 6 / x"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = 6 / input_data
        self._test_operation("rdiv", lambda x: 6 / x, input_data, expected)

    # ==================== Ufunc Operations ====================

    def test_np_sin(self):
        """Test: np.sin(x)"""
        input_data = np.array([0.0, 0.5, 1.0])
        expected = np.sin(input_data)
        self._test_operation("np_sin", lambda x: np.sin(x), input_data, expected)

    def test_np_cos(self):
        """Test: np.cos(x)"""
        input_data = np.array([0.0, 0.5, 1.0])
        expected = np.cos(input_data)
        self._test_operation("np_cos", lambda x: np.cos(x), input_data, expected)

    def test_np_exp(self):
        """Test: np.exp(x)"""
        input_data = np.array([0.0, 1.0, 2.0])
        expected = np.exp(input_data)
        self._test_operation("np_exp", lambda x: np.exp(x), input_data, expected)

    def test_np_log(self):
        """Test: np.log(x)"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.log(input_data)
        self._test_operation("np_log", lambda x: np.log(x), input_data, expected)

    def test_np_sqrt(self):
        """Test: np.sqrt(x)"""
        input_data = np.array([1.0, 4.0, 9.0])
        expected = np.sqrt(input_data)
        self._test_operation("np_sqrt", lambda x: np.sqrt(x), input_data, expected)

    def test_np_abs(self):
        """Test: np.abs(x)"""
        input_data = np.array([-1.0, 2.0, -3.0])
        expected = np.abs(input_data)
        self._test_operation("np_abs", lambda x: np.abs(x), input_data, expected)

    def test_np_tanh(self):
        """Test: np.tanh(x)"""
        input_data = np.array([0.0, 0.5, 1.0])
        expected = np.tanh(input_data)
        self._test_operation("np_tanh", lambda x: np.tanh(x), input_data, expected)

    # ==================== Reduction Operations ====================

    def test_np_sum_scalar(self):
        """Test: np.sum(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.asarray(np.sum(input_data))
        self._test_operation("np_sum_scalar", lambda x: np.sum(x), input_data, expected)

    def test_np_mean_scalar(self):
        """Test: np.mean(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.asarray(np.mean(input_data))
        self._test_operation(
            "np_mean_scalar", lambda x: np.mean(x), input_data, expected
        )

    def test_np_var_scalar(self):
        """Test: np.var(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.asarray(np.var(input_data))
        self._test_operation("np_var_scalar", lambda x: np.var(x), input_data, expected)

    def test_np_std_scalar(self):
        """Test: np.std(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.asarray(np.std(input_data))
        self._test_operation("np_std_scalar", lambda x: np.std(x), input_data, expected)

    def test_np_prod_scalar(self):
        """Test: np.prod(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.asarray(np.prod(input_data))
        self._test_operation(
            "np_prod_scalar", lambda x: np.prod(x), input_data, expected
        )

    def test_np_prod_axis(self):
        """Test: np.prod(x, axis=1) — axis must be renamed to dim for aten."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.prod(input_data, axis=1)
        self._test_operation(
            "np_prod_axis", lambda x: np.prod(x, axis=1), input_data, expected
        )

    def test_np_max_scalar(self):
        """Test: np.max(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 3.0, 2.0])
        expected = np.asarray(np.max(input_data))
        self._test_operation("np_max_scalar", lambda x: np.max(x), input_data, expected)

    def test_np_min_scalar(self):
        """Test: np.min(x) returning a scalar — must stay traced."""
        input_data = np.array([1.0, 3.0, 2.0])
        expected = np.asarray(np.min(input_data))
        self._test_operation("np_min_scalar", lambda x: np.min(x), input_data, expected)

    def test_np_sum_keepdims(self):
        """Test: np.sum(x, keepdims=True) - tests keepdims→keepdim conversion"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.sum(input_data, keepdims=True)
        self._test_operation(
            "np_sum_keepdims", lambda x: np.sum(x, keepdims=True), input_data, expected
        )

    def test_np_mean_keepdims(self):
        """Test: np.mean(x, keepdims=True) - tests keepdims→keepdim conversion"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.mean(input_data, keepdims=True)
        self._test_operation(
            "np_mean_keepdims",
            lambda x: np.mean(x, keepdims=True),
            input_data,
            expected,
        )

    def test_np_max_axis(self):
        """Test: np.max(x, axis=0)"""
        input_data = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]])
        expected = np.max(input_data, axis=0)
        self._test_operation(
            "np_max_axis", lambda x: np.max(x, axis=0), input_data, expected
        )

    def test_np_min_axis(self):
        """Test: np.min(x, axis=0)"""
        input_data = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]])
        expected = np.min(input_data, axis=0)
        self._test_operation(
            "np_min_axis", lambda x: np.min(x, axis=0), input_data, expected
        )

    def test_np_stack_axis(self):
        """Test: np.stack([x, x], axis=0) — axis must be renamed to dim for aten."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.stack([input_data, input_data], axis=0)
        self._test_operation(
            "np_stack_axis", lambda x: np.stack([x, x], axis=0), input_data, expected
        )

    def test_np_split_axis(self):
        """Test: np.split(x, 2, axis=0) — axis must be renamed to dim for aten."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.split(input_data, 2, axis=0)[0]
        self._test_operation(
            "np_split_axis", lambda x: np.split(x, 2, axis=0)[0], input_data, expected
        )

    def test_np_array_split_axis(self):
        """Test: np.array_split(x, 2, axis=0) — axis must be renamed to dim for aten."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array_split(input_data, 2, axis=0)[0]
        self._test_operation(
            "np_array_split_axis",
            lambda x: np.array_split(x, 2, axis=0)[0],
            input_data,
            expected,
        )

    def test_np_flip_axis(self):
        """Test: np.flip(x, axis=1) — axis becomes dims as a tuple."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.ascontiguousarray(np.flip(input_data, axis=1))
        self._test_operation(
            "np_flip_axis", lambda x: np.flip(x, axis=1), input_data, expected
        )

    def test_np_flip_positional(self):
        """Test: np.flip(x, 1) — integer dims must be wrapped for torch.flip."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.ascontiguousarray(np.flip(input_data, 1))
        self._test_operation(
            "np_flip_positional", lambda x: np.flip(x, 1), input_data, expected
        )

    def test_np_flip_all_axes(self):
        """Test: np.flip(x) — omitted axis reverses every dimension."""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.ascontiguousarray(np.flip(input_data))
        self._test_operation(
            "np_flip_all_axes", lambda x: np.flip(x), input_data, expected
        )

    def test_np_roll_axis(self):
        """Test: np.roll(x, 1, axis=1) — axis becomes dims, not dim."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.roll(input_data, 1, axis=1)
        self._test_operation(
            "np_roll_axis", lambda x: np.roll(x, 1, axis=1), input_data, expected
        )

    # ==================== Shape Operations ====================

    def test_reshape(self):
        """Test: np.reshape(x, (2, 3))"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = np.reshape(input_data, (2, 3))
        self._test_operation(
            "reshape", lambda x: np.reshape(x, (2, 3)), input_data, expected
        )

    def test_transpose_no_axes(self):
        """Test: np.transpose(x) - tests automatic axes reversal for torch.permute"""
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # No axes specified - numpy reverses all dims, framework should handle this
        expected = np.transpose(input_data)
        self._test_operation(
            "transpose_no_axes", lambda x: np.transpose(x), input_data, expected
        )

    def test_transpose_with_axes(self):
        """Test: np.transpose(x, axes) - explicit axes"""
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = np.transpose(input_data, (1, 0))
        self._test_operation(
            "transpose_with_axes",
            lambda x: np.transpose(x, (1, 0)),
            input_data,
            expected,
        )

    def test_squeeze(self):
        """Test: np.squeeze(x)"""
        input_data = np.array([[[1.0, 2.0, 3.0]]])
        expected = np.squeeze(input_data)
        self._test_operation("squeeze", lambda x: np.squeeze(x), input_data, expected)

    # ==================== Chained Operations ====================

    def test_chained_arithmetic(self):
        """Test: (x + 1) * 2 - 3"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = (input_data + 1) * 2 - 3
        self._test_operation(
            "chained_arithmetic", lambda x: (x + 1) * 2 - 3, input_data, expected
        )

    def test_chained_ufuncs(self):
        """Test: np.exp(np.sin(x))"""
        input_data = np.array([0.0, 0.5, 1.0])
        expected = np.exp(np.sin(input_data))
        self._test_operation(
            "chained_ufuncs", lambda x: np.exp(np.sin(x)), input_data, expected
        )

    def test_complex_computation(self):
        """Test: np.sqrt(x**2 + 1)"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.sqrt(input_data**2 + 1)
        self._test_operation(
            "complex_computation", lambda x: np.sqrt(x**2 + 1), input_data, expected
        )

    # ==================== Indexing Operations ====================

    def test_slice(self):
        """Test: x[1:4]"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = input_data[1:4]
        self._test_operation("slice", lambda x: x[1:4], input_data, expected)

    # ==================== Comparison Operations ====================

    def test_greater_than(self):
        """Test: x > 2"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = (input_data > 2).astype(np.float32)
        self._test_operation(
            "greater_than", lambda x: (x > 2).astype(np.float32), input_data, expected
        )

    # ==================== Clip Operation ====================

    def test_np_clip(self):
        """Test: np.clip(x, 1.5, 2.5)"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.clip(input_data, 1.5, 2.5)
        self._test_operation(
            "np_clip", lambda x: np.clip(x, 1.5, 2.5), input_data, expected
        )

    def test_np_maximum_scalar(self):
        """Test: np.maximum(x, 3.0) — torch.maximum requires tensor operands."""
        input_data = np.array([1.0, 4.0, 2.0])
        expected = np.maximum(input_data, 3.0)
        self._test_operation(
            "np_maximum_scalar", lambda x: np.maximum(x, 3.0), input_data, expected
        )

    def test_np_minimum_scalar(self):
        """Test: np.minimum(x, 3.0) — torch.minimum requires tensor operands."""
        input_data = np.array([1.0, 4.0, 2.0])
        expected = np.minimum(input_data, 3.0)
        self._test_operation(
            "np_minimum_scalar", lambda x: np.minimum(x, 3.0), input_data, expected
        )

    def test_np_round(self):
        """Test: np.round(x) — dispatches as a function, not a ufunc."""
        input_data = np.array([1.4, 2.6, -0.6])
        expected = np.round(input_data)
        self._test_operation("np_round", lambda x: np.round(x), input_data, expected)

    # ==================== Linear Algebra Operations ====================

    def test_np_matmul(self):
        """Test: np.matmul(x, x.T) — matmul is a ufunc and must stay traced."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.matmul(input_data, input_data.T)
        self._test_operation(
            "np_matmul", lambda x: np.matmul(x, x.T), input_data, expected
        )

    def test_matmul_operator(self):
        """Test: x @ x.T — the operator form routes through the same ufunc."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = input_data @ input_data.T
        self._test_operation("matmul_operator", lambda x: x @ x.T, input_data, expected)

    def test_np_tensordot_axes(self):
        """Test: np.tensordot(x, x, axes=...) — torch.tensordot takes dims."""
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.tensordot(input_data, input_data, axes=([1], [1]))
        self._test_operation(
            "np_tensordot_axes",
            lambda x: np.tensordot(x, x, axes=([1], [1])),
            input_data,
            expected,
        )

    # ==================== Sorting and Searching Operations ====================

    def test_np_sort(self):
        """Test: np.sort(x)"""
        input_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        expected = np.sort(input_data)
        self._test_operation("np_sort", lambda x: np.sort(x), input_data, expected)

    def test_np_sort_axis(self):
        for axis in (None, 1):
            x = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], dtype=np.float32)
            fn = lambda x: np.sort(x, axis=axis)
            self._test_operation(
                f"sort_{axis}", fn, x, fn(x), replay_inputs=(x + 2, -x)
            )

    def test_np_argsort(self):
        # Retain the default 1-D duplicate-value case and its integer output.
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        self._test_operation("argsort", np.argsort, x, np.argsort(x))

        x = np.array(
            [[3.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 5.0]], dtype=np.float32
        )
        fn = lambda x: np.argsort(x, axis=0)
        corner = x.copy()
        corner[0, 0] += 4
        self._test_operation(
            "argsort_axis", fn, x, fn(x), replay_inputs=(x + 2, corner, -x)
        )

    def test_np_where_compiled(self):
        """Test: np.where(x > 2, x, 0)"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.where(input_data > 2, input_data, 0.0)
        self._test_operation(
            "np_where", lambda x: np.where(x > 2, x, 0.0), input_data, expected
        )

    def test_np_nonzero_rows(self):
        """Test: np.nonzero(x)[0] — numpy returns one index array per dimension."""
        # The legacy ONNX Runtime NonZero kernel requires float32 here.
        input_data = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
        expected = np.nonzero(input_data)[0]
        self._test_operation(
            "np_nonzero_rows",
            lambda x: np.nonzero(x)[0],
            input_data,
            expected,
        )

    def test_np_nonzero_cols(self):
        """Test: np.nonzero(x)[1] — the second element indexes the last dimension."""
        # The legacy ONNX Runtime NonZero kernel requires float32 here.
        input_data = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
        expected = np.nonzero(input_data)[1]
        self._test_operation(
            "np_nonzero_cols",
            lambda x: np.nonzero(x)[1],
            input_data,
            expected,
        )

    # ==================== Setitem Operations ====================

    def test_setitem_single_index(self):
        """Test: x[0] = 10"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def setitem_func(x):
            x[0] = 10.0
            return x * 2

        expected = np.array([20.0, 4.0, 6.0, 8.0, 10.0])
        self._test_operation("setitem_single", setitem_func, input_data, expected)

    def test_setitem_slice(self):
        """Test: x[1:3] = [10, 20]"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def setitem_func(x):
            x[1:3] = np.array([10.0, 20.0])
            return x * 2

        expected = np.array([2.0, 20.0, 40.0, 8.0, 10.0])
        self._test_operation("setitem_slice", setitem_func, input_data, expected)

    def test_setitem_full_slice(self):
        """Test: x[:] = constant"""
        input_data = np.array([1.0, 2.0, 3.0])

        def setitem_func(x):
            x[:] = np.array([10.0, 20.0, 30.0])
            return x * 2

        expected = np.array([20.0, 40.0, 60.0])
        self._test_operation("setitem_full", setitem_func, input_data, expected)

    def test_setitem_step_slice(self):
        """Test: x[::2] = [10, 30, 50] with step"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def setitem_func(x):
            x[::2] = np.array([10.0, 30.0, 50.0])  # Assigns to indices 0, 2, 4
            return x * 2

        expected = np.array([20.0, 4.0, 60.0, 8.0, 100.0])
        self._test_operation("setitem_step", setitem_func, input_data, expected)

    def _validate_dynamic_setitem(self, graph_module, inputs, expected, test_name):
        """Validate export-first backends for traced NumPy index inputs."""
        actual = graph_module(*inputs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

        scripted = torch.jit.script(graph_module)
        scripted_actual = scripted(*inputs)
        self.assertTrue(torch.allclose(scripted_actual, expected, atol=1e-5))

        verify_onnx_dynamo(self, graph_module, inputs, expected, test_name=test_name)
        verify_exported_program(
            self, graph_module, inputs, expected, test_name=test_name
        )

    def test_setitem_traced_integer_index_augmented_assignment(self):
        """NumPy x[:, indices] += source uses shared getitem/setitem lowering."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(8, dtype=np.float32).reshape(2, 4), name="x")
        indices = ctx.create_input(np.array([0, 2], dtype=np.int64), name="indices")
        source = ctx.create_input(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            name="source",
        )

        x[:, indices] += source
        result = x * 2.0
        ctx.compile_trace({"result": result})
        self.assertNotIn("__setitem__", str(ctx.m.graph))

        input_x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        input_indices = torch.tensor([1, 3], dtype=torch.int64)
        input_source = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        expected = input_x.clone()
        expected[:, input_indices] += input_source
        expected *= 2.0
        self._validate_dynamic_setitem(
            ctx.m,
            (input_x, input_indices, input_source),
            expected,
            "numpy_traced_integer_augmented_assignment",
        )

    def test_setitem_traced_boolean_mask_augmented_assignment(self):
        """NumPy x[:, mask] *= scalar uses shared getitem/setitem lowering."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(8, dtype=np.float32).reshape(2, 4), name="x")
        mask = ctx.create_input(np.array([True, False, True, False]), name="mask")

        x[:, mask] *= 3.0
        result = x + 1.0
        ctx.compile_trace({"result": result})
        self.assertNotIn("__setitem__", str(ctx.m.graph))

        input_x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        input_mask = torch.tensor([False, True, False, True])
        expected = input_x.clone()
        expected[:, input_mask] *= 3.0
        expected += 1.0
        self._validate_dynamic_setitem(
            ctx.m,
            (input_x, input_mask),
            expected,
            "numpy_traced_mask_augmented_assignment",
        )

    def test_setitem_traced_index_plain_numpy_source(self):
        """A traced NumPy destination accepts a constant ndarray source."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.arange(8, dtype=np.float32).reshape(2, 4), name="x")
        indices = ctx.create_input(np.array([0, 2], dtype=np.int64), name="indices")
        source = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32)

        x[:, indices] = source
        result = x - 1.0
        ctx.compile_trace({"result": result})
        self.assertNotIn("__setitem__", str(ctx.m.graph))

        input_x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        input_indices = torch.tensor([1, 3], dtype=torch.int64)
        expected = input_x.clone()
        expected[:, input_indices] = torch.from_numpy(source)
        expected -= 1.0
        self._validate_dynamic_setitem(
            ctx.m,
            (input_x, input_indices),
            expected,
            "numpy_traced_index_plain_source",
        )


@pytest.mark.parametrize("equal_nan", [False, True])
def test_isclose_special_values_and_broadcasting(equal_nan):
    fn = lambda x, y: np.isclose(x, y, rtol=1e-4, atol=1e-6, equal_nan=equal_nan)
    x = np.array([[0.0, 1.00009, np.nan, np.inf, -np.inf]], dtype=np.float64)
    y = np.array([0.0, 1.0, np.nan, np.inf, np.inf], dtype=np.float64)
    model = _capture_numpy(fn, x, y)
    changed = np.array([[1e-5, 1.0002, 3.0, -np.inf, -np.inf]], dtype=np.float64)
    for a, b in ((x, y), (changed, y), (changed, changed[0])):
        torch.testing.assert_close(
            model(torch.from_numpy(a), torch.from_numpy(b)), torch.from_numpy(fn(a, b))
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cbrt_signed_zero_and_special_values(dtype):
    x = np.array([-np.inf, -8.0, -0.0, 0.0, 27.0, np.inf, np.nan], dtype=dtype)
    model = _capture_numpy(np.cbrt, x)
    for sample in (x, -x):
        actual = model(torch.from_numpy(sample))
        expected = torch.from_numpy(np.cbrt(sample))
        torch.testing.assert_close(actual, expected, equal_nan=True)
        torch.testing.assert_close(
            torch.signbit(actual[2:4]), torch.signbit(expected[2:4])
        )


@pytest.mark.parametrize("axis", [None, -1, (0, 1), ()])
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
def test_nanmean_nan_slices_and_explicit_dtype(axis):
    fn = lambda x: np.nanmean(x, axis=axis, dtype=np.float64, keepdims=True)
    x = np.array([[np.nan, np.nan], [1.0, np.nan]], dtype=np.float32)
    model = _capture_numpy(fn, x)
    changed = np.array([[2.0, 3.0], [np.nan, 4.0]], dtype=np.float32)
    for sample in (x, changed, changed + 2):
        torch.testing.assert_close(
            model(torch.from_numpy(sample)),
            torch.from_numpy(fn(sample)),
            equal_nan=True,
        )


def test_structured_outputs_and_constant_snapshot():
    constant = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fn = lambda x: (np.cross(x, constant), *np.gradient(x))
    x = np.arange(9, dtype=np.float32).reshape(3, 3)
    model = _capture_numpy(fn, x)
    expected = fn(x + 2)
    constant[:] = 100
    for actual, reference in zip(model(torch.from_numpy(x + 2)), expected):
        torch.testing.assert_close(actual, torch.from_numpy(reference))


@pytest.mark.parametrize("backend", ["fx", "jit", "pt2"])
@pytest.mark.parametrize(
    "fn, message",
    [
        (np.spacing, "no torch._numpy lowering"),
        (lambda x: np.allclose(x, x), "ATen capture failed"),
        (lambda x: np.sort(x, stable=True), "sorting"),
        (lambda x: np.argsort(x, kind="mergesort"), "sorting"),
        (lambda x: np.pad(x, 1, mode="reflect"), "pad"),
        (lambda x: np.pad(x, 1, constant_values=(1, 2)), "pad"),
        (lambda x: np.cbrt(x, order="C"), "unsupported signature"),
    ],
)
def test_failed_capture_on_unused_branch(fn, message, backend, monkeypatch):
    errors = []
    monkeypatch.setattr(_get_logger(), "error", errors.append)
    ctx = TracedTensorNode(name="unused_failure", node_index=0)
    sample = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    x = ctx.create_input(sample.copy(), name="x")
    unused = fn(x)
    np.testing.assert_equal(unused.data, fn(sample))
    # Consumers of a failed branch can also be discarded with that branch.
    unused + 1
    assert any(message in error and "unused_failure" in error for error in errors)
    ctx.compile_trace({"y": np.sin(x)})
    assert not any("leapp_capture_error" in node.meta for node in ctx.graph.nodes)
    model = ctx.m
    if backend == "jit":
        model = torch.jit.script(model)
    elif backend == "pt2":
        model = torch.export.export(
            model, (torch.from_numpy(sample),), strict=False
        ).module()
    for changed in (sample, sample + 2, -sample):
        _assert_numpy_output(model(torch.from_numpy(changed)), np.sin(changed))


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("fn", [np.spacing, lambda x: np.pad(x, 1, mode="reflect")])
def test_failed_capture_in_output_tree_is_rejected(fn, dependent, monkeypatch):
    errors = []
    monkeypatch.setattr(_get_logger(), "error", errors.append)
    ctx = TracedTensorNode(name="required_failure", node_index=0)
    x = ctx.create_input(np.array([3.0, 1.0, 2.0], dtype=np.float32), name="x")
    failed = fn(x)
    output = np.sin(failed) if dependent else failed
    assert errors
    with pytest.raises(
        NotImplementedError, match="declared output depends on a failed capture"
    ):
        ctx.compile_trace({"y": output})


def test_native_numpy_errors_still_propagate(monkeypatch):
    errors = []
    monkeypatch.setattr(_get_logger(), "error", errors.append)
    ctx = TracedTensorNode(name="invalid_numpy", node_index=0)
    x = ctx.create_input(np.ones(3, dtype=np.float32), name="x")
    before = tuple(ctx.graph.nodes)
    with pytest.raises(ValueError):
        np.reshape(x, (2, 2))
    assert not errors
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


@pytest.mark.parametrize("positional", [False, True])
def test_missing_ufunc_rejects_out_before_mutating_storage(positional):
    ctx = TracedTensorNode(name="missing_out", node_index=0)
    x = ctx.create_input(np.array([2.0, 1.0, 3.0]), name="x")
    out = np.zeros(3)
    with pytest.raises(NotImplementedError, match="mutation"):
        if positional:
            np.spacing(x, out)
        else:
            np.spacing(x, out=out)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_reject_overwrite_input_before_changing_storage():
    ctx = TracedTensorNode(name="overwrite", node_index=0)
    original = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    x = ctx.create_input(original, name="x")
    with pytest.raises(NotImplementedError, match="mutation"):
        np.nanmedian(x, overwrite_input=True)
    np.testing.assert_array_equal(original, [3.0, 1.0, 2.0])


def test_reject_mutating_call_before_changing_input():
    ctx = TracedTensorNode(name="mutation", node_index=0)
    original = np.array([np.nan, 1.0], dtype=np.float32)
    x = ctx.create_input(original, name="x")
    with pytest.raises(NotImplementedError, match="mutation"):
        np.nan_to_num(x, copy=False)
    np.testing.assert_array_equal(original, [np.nan, 1.0])


def test_successful_capture_executes_native_numpy_once(monkeypatch):
    calls = []

    def native_sin(x):
        calls.append(x)
        return np.sin(x)

    monkeypatch.setitem(
        aten_adapter.NUMPY_OVERRIDES,
        native_sin,
        aten_adapter._implementations()[np.sin],
    )
    sample = np.array([1.0, 2.0], dtype=np.float32)
    ctx = TracedTensorNode(name="single_native_call", node_index=0)
    x = ctx.create_input(sample, name="x")
    y = aten_adapter.capture_numpy_call(x, native_sin, (x,), {})
    assert len(calls) == 1
    assert np.shares_memory(calls[0], sample)
    ctx.compile_trace({"y": y})
    _assert_numpy_output(ctx.m(torch.from_numpy(sample + 1)), np.sin(sample + 1))


@pytest.mark.parametrize("implementation", [lambda x: 1, lambda x: (x, x)])
def test_invalid_capture_outputs_retain_failure_dependency(implementation, monkeypatch):
    monkeypatch.setitem(aten_adapter.NUMPY_OVERRIDES, np.sin, implementation)
    ctx = TracedTensorNode(name="invalid_structure", node_index=0)
    x = ctx.create_input(np.array([1.0, 2.0], dtype=np.float32), name="x")
    y = np.sin(x)
    np.testing.assert_array_equal(y.data, np.sin(x.data))
    with pytest.raises(NotImplementedError, match="declared output depends"):
        ctx.compile_trace({"y": y})


@pytest.mark.parametrize(
    "fn", [lambda x: np.copyto(x, 0), lambda x: np.fill_diagonal(x, 0)]
)
def test_reject_none_returning_mutator_before_changing_storage(fn):
    sample = np.ones((2, 2), dtype=np.float32)
    ctx = TracedTensorNode(name="native_mutator", node_index=0)
    x = ctx.create_input(sample, name="x")
    with pytest.raises(NotImplementedError, match="mutat"):
        fn(x)
    np.testing.assert_array_equal(sample, np.ones((2, 2), dtype=np.float32))


def test_eager_view_keeps_numpy_storage():
    ctx = TracedTensorNode(name="view", node_index=0)
    x = ctx.create_input(np.arange(3, dtype=np.float32), name="x")
    y = np.broadcast_to(x, (2, 3))
    assert np.shares_memory(x.data, y.data)
    assert not y.flags.writeable


if __name__ == "__main__":
    unittest.main()
