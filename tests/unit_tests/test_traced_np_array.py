"""Tests for TracedNpArray operations."""

import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import onnxruntime as ort
import torch

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedNpArray

Array = np.ndarray | TracedNpArray

warnings.filterwarnings(
    "ignore", message=".*legacy TorchScript-based ONNX export.*")
warnings.filterwarnings(
    "ignore", message=".*on empty non-base types in `__init__`.*")
warnings.filterwarnings(
    "ignore", message=".*feature will be removed.*")
warnings.filterwarnings(
    "ignore", message=".*Casting complex values to real.*")


class NpArrayArithmeticFunctions:
    """NumPy array arithmetic functions for testing."""

    @staticmethod
    def scalar_arithmetic(a: Array) -> Array:
        """Basic arithmetic operations."""
        b = a + 1
        c = b - 2
        d = c * 3
        e = d / 4
        f = e ** 5
        return f

    @staticmethod
    def scalar_arithmetic_reverse(a: Array) -> Array:
        """Reverse arithmetic where scalar comes first."""
        b = 1 + a
        c = 2 - b
        d = 3 * c
        e = 4 / d
        f = 5 ** e
        return f

    @staticmethod
    def array_arithmetic(a: Array) -> Array:
        """Array-array arithmetic operations."""
        b = np.array([4.0, 5.0, 6.0])
        c = np.array([7.0, 8.0, 9.0])
        d = a + 2 * b - 0.5 * c
        return d

    @staticmethod
    def array_arithmetic_reuse(a: Array) -> Array:
        """Array self arithmetic operations."""
        b = np.array([4.0, 5.0, 6.0])
        a = a + b
        a = a + 2
        a = a - 1
        a = a * 3
        a = a / 4
        a = a ** 5
        return a

    @staticmethod
    def matmul_operator(a: Array) -> Array:
        """Matrix multiplication operator on array."""
        b = np.array([4.0, 5.0, 6.0])
        c = a @ b
        return c

    @staticmethod
    def matmul_function(a: Array) -> Array:
        """np.matmul operation."""
        b = np.array([4.0, 5.0, 6.0])
        c = np.matmul(a, b)
        return c

    @staticmethod
    def sum_method(a: Array) -> Array:
        """Sum operation via method."""
        b = a.sum()
        return b

    @staticmethod
    def sum_function(a: Array) -> Array:
        """Sum operation via np.sum."""
        b = np.sum(a)
        return b

    @staticmethod
    def mean_method(a: Array) -> Array:
        """Mean operation via method."""
        b = a.mean()
        return b

    @staticmethod
    def mean_function(a: Array) -> Array:
        """Mean operation via np.mean."""
        b = np.mean(a)
        return b

    @staticmethod
    def max_method(a: Array) -> Array:
        """Max operation via method."""
        b = a.max()
        return b

    @staticmethod
    def max_function(a: Array) -> Array:
        """Max operation via np.max."""
        b = np.max(a)
        return b

    @staticmethod
    def min_method(a: Array) -> Array:
        """Min operation via method."""
        b = a.min()
        return b

    @staticmethod
    def min_function(a: Array) -> Array:
        """Min operation via np.min."""
        b = np.min(a)
        return b

    @staticmethod
    def argmax_method(a: Array) -> Array:
        """Argmax operation via method."""
        b = a.argmax()
        return b

    @staticmethod
    def argmax_function(a: Array) -> Array:
        """Argmax operation via np.argmax."""
        b = np.argmax(a)
        return b

    @staticmethod
    def argmin_method(a: Array) -> Array:
        """Argmin operation via method."""
        b = a.argmin()
        return b

    @staticmethod
    def argmin_function(a: Array) -> Array:
        """Argmin operation via np.argmin."""
        b = np.argmin(a)
        return b

    @staticmethod
    def reshape_method(a: Array) -> Array:
        """Reshape operation via method."""
        b = a.reshape(1, 3)
        c = b * 2
        d = c.reshape(3)
        return d

    @staticmethod
    def reshape_function(a: Array) -> Array:
        """Reshape operation via np.reshape."""
        b = np.reshape(a, (1, 3))
        c = b * 2
        d = np.reshape(c, (3,))
        return d

    @staticmethod
    def transpose_method(a: Array) -> Array:
        """array.transpose() method."""
        b = a.reshape(3, 1)
        c = b.transpose(1, 0)
        d = c * 2
        return d

    @staticmethod
    def transpose_function(a: Array) -> Array:
        """np.transpose operation."""
        b = a.reshape(3, 1)
        c = np.transpose(b, (1, 0))
        d = c * 2
        return d

    @staticmethod
    def squeeze_method(a: Array) -> Array:
        """array.squeeze() method."""
        b = a.reshape(1, 3, 1)
        c = b.squeeze()
        d = c * 2
        return d

    @staticmethod
    def squeeze_function(a: Array) -> Array:
        """np.squeeze operation."""
        b = a.reshape(1, 3, 1)
        c = np.squeeze(b)
        d = c * 2
        return d

    @staticmethod
    def broadcasting_operation(a: Array) -> Array:
        """Broadcasting operation."""
        b = a.reshape(3, 1)
        c = np.array([[4.0, 5.0]])
        d = b + c
        return d

    @staticmethod
    def indexing_single(a: Array) -> Array:
        """Single element indexing."""
        b = a[1]
        return b

    @staticmethod
    def indexing_slice(a: Array) -> Array:
        """Slice indexing."""
        b = a[0:2]
        c = b * 2
        return c

    @staticmethod
    def indexing_negative(a: Array) -> Array:
        """Negative indexing."""
        b = a[-1]
        return b

    @staticmethod
    def indexing_step(a: Array) -> Array:
        """Step indexing."""
        b = a[::2]
        c = b * 2
        return c


class TestTracedNpArray(unittest.TestCase):
    """Test TracedNpArray operations."""

    # ==================== Basic Property Tests ====================

    def test_shape(self):
        """Test shape property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4, 5), name="x")
        self.assertEqual(x.shape, (3, 4, 5))

    def test_ndim(self):
        """Test ndim property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4, 5), name="x")
        self.assertEqual(x.ndim, 3)

    def test_dtype(self):
        """Test dtype property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1, 2, 3], dtype=np.int32), name="x")
        self.assertEqual(x.dtype, np.int32)

    def test_len(self):
        """Test len() function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(10, 5), name="x")
        self.assertEqual(len(x), 10)

    def test_size(self):
        """Test size property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4, 5), name="x")
        self.assertEqual(x.size, 60)

    def test_T_property(self):
        """Test .T transpose property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4), name="x")
        result = x.T
        self.assertEqual(result.shape, (4, 3))
        self.assertIsInstance(result, TracedNpArray)

    # ==================== Arithmetic Operation Tests ====================

    def test_add(self):
        """Test addition."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x + 1
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 3.0, 4.0]))

    def test_radd(self):
        """Test reverse addition."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = 1 + x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 3.0, 4.0]))

    def test_sub(self):
        """Test subtraction."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x - 1
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([0.0, 1.0, 2.0]))

    def test_rsub(self):
        """Test reverse subtraction."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = 10 - x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([9.0, 8.0, 7.0]))

    def test_mul(self):
        """Test multiplication."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x * 2
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 4.0, 6.0]))

    def test_rmul(self):
        """Test reverse multiplication."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = 2 * x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([2.0, 4.0, 6.0]))

    def test_truediv(self):
        """Test division."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([2.0, 4.0, 6.0]), name="x")
        y = x / 2
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_rtruediv(self):
        """Test reverse division."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 4.0]), name="x")
        y = 8 / x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([8.0, 4.0, 2.0]))

    def test_pow(self):
        """Test power."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x ** 2
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 4.0, 9.0]))

    def test_neg(self):
        """Test negation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, -2.0, 3.0]), name="x")
        y = -x
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([-1.0, 2.0, -3.0]))

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
        x = ctx.create_input(np.array([0.0, np.pi/2, np.pi]), name="x")
        y = np.sin(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([0.0, 1.0, 0.0]), decimal=5)

    def test_np_cos(self):
        """Test np.cos ufunc."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([0.0, np.pi/2, np.pi]), name="x")
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

    def test_np_sum(self):
        """Test np.sum function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = np.sum(x)
        # Result might be scalar or 0-d array
        self.assertAlmostEqual(float(y), 6.0)

    def test_np_mean(self):
        """Test np.mean function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = np.mean(x)
        self.assertAlmostEqual(float(y), 2.0)

    def test_np_max(self):
        """Test np.max function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 3.0, 2.0]), name="x")
        y = np.max(x)
        self.assertAlmostEqual(float(y), 3.0)

    def test_np_min(self):
        """Test np.min function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 3.0, 2.0]), name="x")
        y = np.min(x)
        self.assertAlmostEqual(float(y), 1.0)

    def test_np_reshape(self):
        """Test np.reshape function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), name="x")
        y = np.reshape(x, (2, 3))
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (2, 3))

    def test_np_transpose(self):
        """Test np.transpose function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(3, 4), name="x")
        y = np.transpose(x)
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (4, 3))

    def test_np_squeeze(self):
        """Test np.squeeze function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.random.randn(1, 3, 1), name="x")
        y = np.squeeze(x)
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.shape, (3,))

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

    def test_np_sort(self):
        """Test np.sort function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([3.0, 1.0, 2.0]), name="x")
        y = np.sort(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_np_sort_axis(self):
        """Test np.sort with axis parameter."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([[3.0, 1.0], [2.0, 4.0]]), name="x")
        y = np.sort(x, axis=1)
        self.assertIsInstance(y, TracedNpArray)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_almost_equal(y, expected)

    def test_np_argsort(self):
        """Test np.argsort function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([3.0, 1.0, 2.0]), name="x")
        y = np.argsort(x)
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_equal(y, np.array([1, 2, 0]))

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
        np.testing.assert_array_almost_equal(y, x)

    # ==================== Graph Building Tests ====================

    def test_graph_basic_operations(self):
        """Test that basic operations create an FX graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x + 1
        z = y * 2
        
        # Compile and verify graph exists
        ctx.compile_trace({'output': z})
        self.assertIsNotNone(ctx.compiled_graph_module)
        self.assertIsNotNone(ctx.graph)

    def test_graph_has_nodes(self):
        """Test that operations are recorded in the graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x + 1
        z = y * 2
        
        ctx.compile_trace({'output': z})
        
        # Check graph has nodes
        nodes = list(ctx.graph.nodes)
        self.assertGreater(len(nodes), 0)

    def test_graph_multiple_inputs(self):
        """Test graph with multiple inputs."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = ctx.create_input(np.array([4.0, 5.0, 6.0]), name="y")
        z = x + y
        
        ctx.compile_trace({'output': z})
        self.assertIsNotNone(ctx.compiled_graph_module)

    # ==================== Operations After Compile Tests ====================

    def test_operations_after_compile_return_arrays(self):
        """Test that operations return raw arrays after compile_trace."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        y = x * 2
        
        ctx.compile_trace({'y': y})
        
        # After compilation, is_tracing should be False
        self.assertFalse(ctx.is_tracing)
        
        # Operations on traced array should now return regular arrays
        z = x + 1
        self.assertIsInstance(z, np.ndarray)
        self.assertNotIsInstance(z, TracedNpArray)

    def test_getitem_after_compile_returns_array(self):
        """Test that indexing returns raw array after compile."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        # Use slice to get array output (scalar indexing returns np.float64, not array)
        y = x[0:2]
        
        ctx.compile_trace({'y': y})
        
        # Indexing after compile should return regular array
        z = x[2:4]
        self.assertNotIsInstance(z, TracedNpArray)

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
        w = y ** 2 + z ** 2  # Should be ~1.0
        
        self.assertIsInstance(w, TracedNpArray)
        np.testing.assert_array_almost_equal(w, np.ones(3), decimal=5)


class TestArithmeticFunctions(unittest.TestCase):
    """Test all arithmetic functions can be traced and produce correct results."""

    def setUp(self):
        # Use [2.0, 3.0, 4.0] to avoid zeros in reverse arithmetic:
        # b = 1 + a = [3, 4, 5], c = 2 - b = [-1, -2, -3] (no zeros)
        self.test_data = np.array([2.0, 3.0, 4.0])

    def _test_function(self, func, name):
        """Helper to test a function produces correct results."""
        # Get expected result from regular numpy
        expected = func(self.test_data.copy())
        
        # Create traced version
        ctx = TracedTensorNode(name=f"test_{name}", node_index=0)
        traced_input = ctx.create_input(self.test_data.copy(), name="input")
        traced_result = func(traced_input)
        
        # Compare results
        if isinstance(traced_result, TracedNpArray):
            actual = traced_result
        else:
            actual = np.array(traced_result)
        
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_almost_equal(actual, expected, decimal=5)
        else:
            self.assertAlmostEqual(float(actual), float(expected), places=5)

    def test_scalar_arithmetic(self):
        """Test scalar arithmetic function."""
        self._test_function(NpArrayArithmeticFunctions.scalar_arithmetic, "scalar_arithmetic")

    def test_scalar_arithmetic_reverse(self):
        """Test reverse scalar arithmetic function."""
        self._test_function(NpArrayArithmeticFunctions.scalar_arithmetic_reverse, "scalar_arithmetic_reverse")

    def test_array_arithmetic(self):
        """Test array arithmetic function."""
        self._test_function(NpArrayArithmeticFunctions.array_arithmetic, "array_arithmetic")

    def test_sum_method(self):
        """Test sum method."""
        self._test_function(NpArrayArithmeticFunctions.sum_method, "sum_method")

    def test_sum_function(self):
        """Test sum function."""
        self._test_function(NpArrayArithmeticFunctions.sum_function, "sum_function")

    def test_mean_method(self):
        """Test mean method."""
        self._test_function(NpArrayArithmeticFunctions.mean_method, "mean_method")

    def test_mean_function(self):
        """Test mean function."""
        self._test_function(NpArrayArithmeticFunctions.mean_function, "mean_function")

    def test_reshape_method(self):
        """Test reshape method."""
        self._test_function(NpArrayArithmeticFunctions.reshape_method, "reshape_method")

    def test_reshape_function(self):
        """Test reshape function."""
        self._test_function(NpArrayArithmeticFunctions.reshape_function, "reshape_function")


class TestCompiledGraphExecution(unittest.TestCase):
    """Test that traced numpy operations compile and execute correctly.
    
    These tests verify that:
    1. The FX graph module produces correct output
    2. TorchScript export works and produces correct output
    3. ONNX export works and produces correct output
    """

    def _test_operation(self, op_name: str, np_func, input_data: np.ndarray, 
                        expected_output: np.ndarray, atol: float = 1e-5):
        """Helper to test a single operation through all export formats.
        
        Args:
            op_name: Name of the operation (for error messages)
            np_func: Function that takes TracedNpArray and returns result
            input_data: Input numpy array
            expected_output: Expected numpy output
            atol: Absolute tolerance for comparisons
        """
        # Create traced version
        ctx = TracedTensorNode(name=f"test_{op_name}", node_index=0)
        traced_input = ctx.create_input(input_data.copy(), name="input")
        traced_result = np_func(traced_input)
        
        # Handle scalar results (wrap in array for comparison)
        if not isinstance(traced_result, TracedNpArray):
            # Scalar result - skip graph tests as scalars aren't supported
            return
        
        # Compile the trace
        ctx.compile_trace({'output': traced_result})
        graph_module = ctx.compiled_graph_module
        
        # Convert input to torch tensor for execution
        input_tensor = torch.from_numpy(input_data.copy()).float()
        
        # Test 1: FX GraphModule execution
        try:
            graph_output = graph_module(input_tensor)
            graph_output_np = graph_output.detach().numpy()
            np.testing.assert_allclose(
                graph_output_np.flatten(), 
                expected_output.flatten(), 
                atol=atol,
                err_msg=f"{op_name}: FX GraphModule output doesn't match expected"
            )
        except Exception as e:
            self.fail(f"{op_name}: FX GraphModule execution failed: {e}")
        
        # Test 2: TorchScript export and execution
        try:
            scripted = torch.jit.script(graph_module)
            script_output = scripted(input_tensor)
            script_output_np = script_output.detach().numpy()
            np.testing.assert_allclose(
                script_output_np.flatten(), 
                expected_output.flatten(), 
                atol=atol,
                err_msg=f"{op_name}: TorchScript output doesn't match expected"
            )
        except Exception as e:
            self.fail(f"{op_name}: TorchScript export/execution failed: {e}")
        
        # Test 3: ONNX export and execution
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = pathlib.Path(tmpdir) / f"{op_name}.onnx"
                torch.onnx.export(
                    graph_module,
                    (input_tensor,),
                    onnx_path,
                    dynamo=False,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                )
                
                session = ort.InferenceSession(str(onnx_path))
                onnx_output = session.run(None, {"input": input_data.astype(np.float32)})[0]
                
                np.testing.assert_allclose(
                    onnx_output.flatten(), 
                    expected_output.flatten(), 
                    atol=atol,
                    err_msg=f"{op_name}: ONNX output doesn't match expected"
                )
        except Exception as e:
            self.fail(f"{op_name}: ONNX export/execution failed: {e}")

    # ==================== Arithmetic Operations ====================

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

    def test_div_scalar(self):
        """Test: x / 2"""
        input_data = np.array([2.0, 4.0, 6.0])
        expected = input_data / 2
        self._test_operation("div_scalar", lambda x: x / 2, input_data, expected)

    def test_pow_scalar(self):
        """Test: x ** 2"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = input_data ** 2
        self._test_operation("pow_scalar", lambda x: x ** 2, input_data, expected)

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

    def test_np_sum_keepdims(self):
        """Test: np.sum(x, keepdims=True) - tests keepdims→keepdim conversion"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.sum(input_data, keepdims=True)
        self._test_operation("np_sum_keepdims", lambda x: np.sum(x, keepdims=True), input_data, expected)

    def test_np_mean_keepdims(self):
        """Test: np.mean(x, keepdims=True) - tests keepdims→keepdim conversion"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.mean(input_data, keepdims=True)
        self._test_operation("np_mean_keepdims", lambda x: np.mean(x, keepdims=True), input_data, expected)

    def test_np_max_axis(self):
        """Test: np.max(x, axis=0)"""
        input_data = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]])
        expected = np.max(input_data, axis=0)
        self._test_operation("np_max_axis", lambda x: np.max(x, axis=0), input_data, expected)

    def test_np_min_axis(self):
        """Test: np.min(x, axis=0)"""
        input_data = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 5.0]])
        expected = np.min(input_data, axis=0)
        self._test_operation("np_min_axis", lambda x: np.min(x, axis=0), input_data, expected)

    # ==================== Shape Operations ====================

    def test_reshape(self):
        """Test: np.reshape(x, (2, 3))"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = np.reshape(input_data, (2, 3))
        self._test_operation("reshape", lambda x: np.reshape(x, (2, 3)), input_data, expected)

    def test_transpose_no_axes(self):
        """Test: np.transpose(x) - tests automatic axes reversal for torch.permute"""
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # No axes specified - numpy reverses all dims, framework should handle this
        expected = np.transpose(input_data)
        self._test_operation("transpose_no_axes", lambda x: np.transpose(x), input_data, expected)

    def test_transpose_with_axes(self):
        """Test: np.transpose(x, axes) - explicit axes"""
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = np.transpose(input_data, (1, 0))
        self._test_operation("transpose_with_axes", lambda x: np.transpose(x, (1, 0)), input_data, expected)

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
        self._test_operation("chained_arithmetic", lambda x: (x + 1) * 2 - 3, input_data, expected)

    def test_chained_ufuncs(self):
        """Test: np.exp(np.sin(x))"""
        input_data = np.array([0.0, 0.5, 1.0])
        expected = np.exp(np.sin(input_data))
        self._test_operation("chained_ufuncs", lambda x: np.exp(np.sin(x)), input_data, expected)

    def test_complex_computation(self):
        """Test: np.sqrt(x**2 + 1)"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.sqrt(input_data**2 + 1)
        self._test_operation("complex_computation", lambda x: np.sqrt(x**2 + 1), input_data, expected)

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
        self._test_operation("greater_than", lambda x: (x > 2).astype(np.float32), 
                            input_data, expected)

    # ==================== Clip Operation ====================

    def test_np_clip(self):
        """Test: np.clip(x, 1.5, 2.5)"""
        input_data = np.array([1.0, 2.0, 3.0])
        expected = np.clip(input_data, 1.5, 2.5)
        self._test_operation("np_clip", lambda x: np.clip(x, 1.5, 2.5), input_data, expected)

    # ==================== Sorting and Searching Operations ====================

    def test_np_sort(self):
        """Test: np.sort(x)"""
        input_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        expected = np.sort(input_data)
        self._test_operation("np_sort", lambda x: np.sort(x), input_data, expected)

    def test_np_sort_axis(self):
        """Test: np.sort(x, axis=1)"""
        input_data = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        expected = np.sort(input_data, axis=1)
        self._test_operation("np_sort_axis", lambda x: np.sort(x, axis=1), input_data, expected)

    def test_np_argsort(self):
        """Test: np.argsort(x)"""
        input_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        expected = np.argsort(input_data).astype(np.float32)  # Convert to float for comparison
        # Note: argsort returns indices as int64, but we compare as float for consistency
        self._test_operation("np_argsort", lambda x: np.argsort(x).astype(np.float32), 
                            input_data, expected)

    def test_np_where_compiled(self):
        """Test: np.where(x > 2, x, 0)"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.where(input_data > 2, input_data, 0.0)
        self._test_operation("np_where", lambda x: np.where(x > 2, x, 0.0), input_data, expected)

    # ==================== Setitem Operations ====================

    def _test_setitem(self, test_name: str, input_data: np.ndarray, 
                      setitem_func, expected_output: np.ndarray, atol: float = 1e-5):
        """Helper to test setitem operations through all export formats.
        
        Args:
            test_name: Name of the test (for error messages and ONNX filename)
            input_data: Input numpy array
            setitem_func: Function that takes TracedNpArray and does setitem + returns result
            expected_output: Expected numpy output
            atol: Absolute tolerance for comparisons
        """
        # Create traced version
        ctx = TracedTensorNode(name=f"test_{test_name}", node_index=0)
        traced_input = ctx.create_input(input_data.copy(), name="input")
        traced_result = setitem_func(traced_input)
        
        # Compile the trace
        ctx.compile_trace({'output': traced_result})
        graph_module = ctx.compiled_graph_module
        
        # Convert input to torch tensor for execution
        torch_input = torch.from_numpy(input_data.copy()).float()
        torch_expected = torch.from_numpy(expected_output).float()
        
        # Test 1: FX GraphModule execution
        try:
            output = graph_module(torch_input)
            self.assertTrue(
                torch.allclose(output, torch_expected, atol=atol),
                f"{test_name}: FX GraphModule execution failed - output doesn't match"
            )
        except Exception as e:
            self.fail(f"{test_name}: FX GraphModule execution failed: {e}")
        
        # Test 2: TorchScript export
        try:
            scripted = torch.jit.script(graph_module)
            output_ts = scripted(torch_input)
            self.assertTrue(
                torch.allclose(output_ts, torch_expected, atol=atol),
                f"{test_name}: TorchScript output doesn't match"
            )
        except Exception as e:
            self.fail(f"{test_name}: TorchScript export/execution failed: {e}")
        
        # Test 3: ONNX export
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = pathlib.Path(tmpdir) / f"{test_name}.onnx"
                torch.onnx.export(
                    graph_module,
                    (torch_input,),
                    onnx_path,
                    dynamo=False,
                    export_params=True,
                    opset_version=17,
                    input_names=['input'],
                    output_names=['output'],
                )
                session = ort.InferenceSession(str(onnx_path))
                output_onnx = session.run(None, {"input": torch_input.numpy()})[0]
                self.assertTrue(
                    np.allclose(output_onnx, expected_output, atol=atol),
                    f"{test_name}: ONNX output doesn't match"
                )
        except Exception as e:
            self.fail(f"{test_name}: ONNX export/execution failed: {e}")

    def test_setitem_single_index(self):
        """Test: x[0] = 10"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        def setitem_func(x):
            x[0] = 10.0
            return x * 2
        
        expected = np.array([20.0, 4.0, 6.0, 8.0, 10.0])
        self._test_setitem("setitem_single", input_data, setitem_func, expected)

    def test_setitem_slice(self):
        """Test: x[1:3] = [10, 20]"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        def setitem_func(x):
            x[1:3] = np.array([10.0, 20.0])
            return x * 2
        
        expected = np.array([2.0, 20.0, 40.0, 8.0, 10.0])
        self._test_setitem("setitem_slice", input_data, setitem_func, expected)

    def test_setitem_full_slice(self):
        """Test: x[:] = constant"""
        input_data = np.array([1.0, 2.0, 3.0])
        
        def setitem_func(x):
            x[:] = np.array([10.0, 20.0, 30.0])
            return x * 2
        
        expected = np.array([20.0, 40.0, 60.0])
        self._test_setitem("setitem_full", input_data, setitem_func, expected)

    def test_setitem_step_slice(self):
        """Test: x[::2] = [10, 30, 50] with step"""
        input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        def setitem_func(x):
            x[::2] = np.array([10.0, 30.0, 50.0])  # Assigns to indices 0, 2, 4
            return x * 2
        
        expected = np.array([20.0, 4.0, 60.0, 8.0, 100.0])
        self._test_setitem("setitem_step", input_data, setitem_func, expected)


if __name__ == "__main__":
    unittest.main()

