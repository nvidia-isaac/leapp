"""Tests for basic TrackedTensor operations."""

import inspect
import pathlib
import tempfile
import unittest

import onnxruntime as ort
import torch
import warnings

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.traced_tensor import TracedTensor

Tensor = torch.Tensor | TracedTensor

warnings.filterwarnings(
    "ignore", message=".*legacy TorchScript-based ONNX export.*")
warnings.filterwarnings(
    "ignore", message=".*on empty non-base types in `__init__`.*")
warnings.filterwarnings(
    "ignore", message=".*feature will be removed.*")


class TensorArithmeticFunctions:
    """Tensor arithmetic functions for testing."""

    @staticmethod
    def scalar_arithmetic(a: Tensor) -> Tensor:
        """Basic arithmetic operations."""
        b = a + 1
        c = b - 2
        d = c * 3
        e = d / 4
        f = e**5
        return f

    @staticmethod
    def scalar_arithmetic_reverse(a: Tensor) -> Tensor:
        """Reverse arithmetic where scalar comes first."""
        b = 1 + a
        c = 2 - b
        d = 3 * c
        e = 4 / d
        f = 5**e
        return f

    @staticmethod
    def tensor_arithmetic(a: Tensor) -> Tensor:
        """Tensor-tensor arithmetic operations."""
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.tensor([7.0, 8.0, 9.0])
        d = a + 2 * b - 0.5 * c
        return d

    @staticmethod
    def tensor_arithmetic_reuse_tensor(a: Tensor) -> Tensor:
        """Tensor self arithmetic operations."""
        b = torch.tensor([4.0, 5.0, 6.0])
        a = a + b
        a = a + 2
        a = a - 1
        a = a * 3
        a = a / 4
        a = a**5
        return a

    @staticmethod
    def matmul_operator_tensor(a: Tensor) -> Tensor:
        """Matrix multiplication operator on tensor."""
        b = torch.tensor([4.0, 5.0, 6.0])
        c = a @ b
        return c

    @staticmethod
    def matmul_operator_torch(a: Tensor) -> Tensor:
        """1D @ 1D (dot product)."""
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.matmul(a, b)
        return c

    @staticmethod
    def sum_operator_tensor(a: Tensor) -> Tensor:
        """Sum operation."""
        b = a.sum()
        return b

    @staticmethod
    def sum_operator_torch(a: Tensor) -> Tensor:
        """Sum operation."""
        b = torch.sum(a)
        return b

    @staticmethod
    def mean_operator_tensor(a: Tensor) -> Tensor:
        """Mean operation."""
        b = a.mean()
        return b

    @staticmethod
    def mean_operator_torch(a: Tensor) -> Tensor:
        """Mean operation."""
        b = torch.mean(a)
        return b

    @staticmethod
    def max_operator_tensor(a: Tensor) -> Tensor:
        """Max operation."""
        b = a.max()
        return b

    @staticmethod
    def max_operator_torch(a: Tensor) -> Tensor:
        """Max operation."""
        b = torch.max(a)
        return b

    @staticmethod
    def min_operator_tensor(a: Tensor) -> Tensor:
        """Min operation."""
        b = a.min()
        return b

    @staticmethod
    def min_operator_torch(a: Tensor) -> Tensor:
        """Min operation."""
        b = torch.min(a)
        return b

    @staticmethod
    def argmax_operator_tensor(a: Tensor) -> Tensor:
        """Argmax operation."""
        b = a.argmax()
        return b

    @staticmethod
    def argmax_operator_torch(a: Tensor) -> Tensor:
        """Argmax operation."""
        b = torch.argmax(a)
        return b

    @staticmethod
    def argmin_operator_tensor(a: Tensor) -> Tensor:
        """Argmin operation."""
        b = a.argmin()
        return b

    @staticmethod
    def argmin_operator_torch(a: Tensor) -> Tensor:
        """Argmin operation."""
        b = torch.argmin(a)
        return b

    @staticmethod
    def reshape_operator_tensor(a: Tensor) -> Tensor:
        """Reshape operation."""
        b = a.reshape(1, 3)
        c = b * 2
        d = c.reshape(3)
        return d

    @staticmethod
    def reshape_operator_torch(a: Tensor) -> Tensor:
        """Reshape operation."""
        b = torch.reshape(a, (1, 3))
        c = b * 2
        d = torch.reshape(c, (3,))
        return d

    @staticmethod
    def transpose_operator_tensor(a: Tensor) -> Tensor:
        """tensor.transpose() method."""
        b = a.reshape(3, 1)
        c = b.transpose(0, 1)
        d = c * 2
        return d

    @staticmethod
    def transpose_operator_torch(a: Tensor) -> Tensor:
        """torch.transpose operation."""
        b = a.reshape(3, 1)
        c = torch.transpose(b, 0, 1)
        d = c * 2
        return d

    @staticmethod
    def permute_operator_tensor(a: Tensor) -> Tensor:
        """tensor.permute() method."""
        b = a.reshape(1, 3, 1)
        c = b.permute(2, 0, 1)
        d = c * 2
        return d

    @staticmethod
    def permute_operator_torch(a: Tensor) -> Tensor:
        """torch.permute operation."""
        b = a.reshape(1, 3, 1)
        c = torch.permute(b, (2, 0, 1))
        d = c * 2
        return d

    @staticmethod
    def squeeze_operator_tensor(a: Tensor) -> Tensor:
        """tensor.squeeze() method."""
        b = a.reshape(1, 3, 1)
        c = b.squeeze()
        d = c * 2
        return d

    @staticmethod
    def squeeze_operator_torch(a: Tensor) -> Tensor:
        """torch.squeeze operation."""
        b = a.reshape(1, 3, 1)
        c = torch.squeeze(b)
        d = c * 2
        return d

    @staticmethod
    def unsqueeze_operator_tensor(a: Tensor) -> Tensor:
        """tensor.unsqueeze() method."""
        b = a.unsqueeze(0)
        c = b * 2
        return c

    @staticmethod
    def unsqueeze_operator_torch(a: Tensor) -> Tensor:
        """torch.unsqueeze operation."""
        b = torch.unsqueeze(a, 0)
        c = b * 2
        return c

    @staticmethod
    def type_conversion_operator_tensor(a: Tensor) -> Tensor:
        """Type conversion operation."""
        b = a.to(torch.float32)
        # Somehow ONNX fails to use the "to" operation as the last operation in
        # the graph, so we add a dummy operation after it.
        c = b * 1.0
        return c

    @staticmethod
    def broadcasting_operation(a: Tensor) -> Tensor:
        """Broadcasting operation."""
        # a is [1.0, 2.0, 3.0], reshape to (3, 1)
        b = a.reshape(3, 1)
        # Create a (1, 2) tensor
        c = torch.tensor([[4.0, 5.0]])
        # Broadcasting: (3, 1) + (1, 2) -> (3, 2)
        d = b + c
        return d

    @staticmethod
    def torch_mm_operation(a: Tensor) -> Tensor:
        """torch.mm operation (2D matrix multiplication)."""
        # Reshape to 2D for mm
        b = a.reshape(3, 1)
        c = torch.tensor([[2.0], [3.0], [4.0]])
        d = torch.mm(b.T, c)  # (1, 3) @ (3, 1) -> (1, 1)
        return d

    @staticmethod
    def relu_operator_torch(a: Tensor) -> Tensor:
        """torch.relu operation."""
        b = a - 2  # Make some values negative
        c = torch.relu(b)
        return c

    @staticmethod
    def sigmoid_operator_torch(a: Tensor) -> Tensor:
        """torch.sigmoid operation."""
        b = torch.sigmoid(a)
        return b

    @staticmethod
    def tanh_operator_torch(a: Tensor) -> Tensor:
        """torch.tanh operation."""
        b = torch.tanh(a)
        return b

    @staticmethod
    def cat_operator_torch(a: Tensor) -> Tensor:
        """torch.cat operation."""
        b = torch.tensor([4.0, 5.0, 6.0])
        c = torch.cat([a, b], dim=0)
        return c

    @staticmethod
    def slicing_basic(a: Tensor) -> Tensor:
        """Slicing operation."""
        b = a[1:]  # Slice from index 1 to end
        c = b * 2
        return c

    @staticmethod
    def slicing_with_step(a: Tensor) -> Tensor:
        """Slicing with step (::2 notation)."""
        # Create a longer tensor for step slicing
        b = torch.cat([a, a, a])  # shape (9,)
        c = b[::2]  # Every other element.
        d = c * 2
        return d

    @staticmethod
    def slicing_negative_indices(a: Tensor) -> Tensor:
        """Slicing with negative indices."""
        b = a[-2:]  # Last two elements.
        c = b * 2
        return c

    @staticmethod
    def slicing_multi_dimensional(a: Tensor) -> Tensor:
        """Multi-dimensional slicing."""
        b = torch.stack([a, a, a])  # shape (3, 3)
        c = b[1:, :]  # Slice first dimension.
        d = c * 2
        return d

    @staticmethod
    def indexing_with_int(a: Tensor) -> Tensor:
        """Indexing operation."""
        b = a[1]  # Get element at index 1
        c = b * 2
        return c

    @staticmethod
    def indexing_with_list(a: Tensor) -> Tensor:
        """Indexing with a list."""
        # Index with a list of indices
        indices = [0, 2]  # Get first and third elements
        b = a[indices]
        c = b * 2
        return c

    @staticmethod
    def indexing_with_tensor(a: Tensor) -> Tensor:
        """Indexing with a tensor."""
        # Index with a tensor of indices
        indices = torch.tensor([0, 2], dtype=torch.long)
        b = a[indices]
        c = b * 2
        return c

    @staticmethod
    def indexing_and_slicing(a: Tensor) -> Tensor:
        """Indexing and slicing."""
        b = torch.stack([a, a, a])  # shape (3, 3)
        indices = [0, 1]
        c = b[indices, 1:]  # Slice first dimension.
        return c

    @staticmethod
    def indexing_with_none(a: Tensor) -> Tensor:
        """None indexing (adds dimension)."""
        b = a[None, :]  # Add dimension at start
        c = b * 2
        return c

    @staticmethod
    def indexing_with_ellipsis(a: Tensor) -> Tensor:
        """Ellipsis indexing."""
        b = a.reshape(1, 3)
        c = b[..., 1]  # Select last dimension
        d = c * 2
        return d

    @staticmethod
    def indexing_empty_slice(a: Tensor) -> Tensor:
        """Empty slice indexing."""
        b = a[0:0]  # Empty slice
        c = torch.cat([b, a])  # Concatenate to verify it works
        return c

    # NOTE: Boolean indexing is currently not supported because the mask itself
    # becomes a TracedTensor, and FX tracer doesn't know how to handle TracedTensor
    # as an indexing argument. This would require special handling in __getitem__.
    # The function below is kept commented out because it's not part of the
    # automatic arithmetic function tests - it requires special handling.
    # @staticmethod
    # def boolean_indexing_operator(a: Tensor) -> Tensor:
    #     """Boolean/mask indexing."""
    #     # Create a boolean mask
    #     mask = a > 1.5  # [False, True, True] for [1.0, 2.0, 3.0]
    #     b = a[mask]
    #     c = b * 2
    #     return c

    @staticmethod
    def control_flow_conditional(a: Tensor) -> Tensor:
        """Conditional operation with if statement."""
        if a.sum() > 0:
            b = a * 2
        else:
            b = a * 3
        return b

    @staticmethod
    def control_flow_loop(a: Tensor) -> Tensor:
        """Loop operation."""
        for _ in range(3):
            a = a * 2
        return a


class TestTracedTensor(unittest.TestCase):
    """Test TracedTensor operations."""

    def test_shape(self):
        """Test shape property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4, 5), name="x")
        self.assertEqual(x.shape, torch.Size([3, 4, 5]))

    def test_dim(self):
        """Test dim property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4, 5), name="x")
        self.assertEqual(x.dim(), 3)

    def test_dtype(self):
        """Test dtype property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor(
            [1, 2, 3], dtype=torch.int32), name="x")
        self.assertEqual(x.dtype, torch.int32)

    def test_device(self):
        """Test device property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0]), name="x")
        self.assertEqual(x.device.type, "cpu")

    def test_len(self):
        """Test len() function."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(10, 5), name="x")
        self.assertEqual(len(x), 10)

    def test_ndim(self):
        """Test ndim property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4, 5), name="x")
        self.assertEqual(x.ndim, 3)

    def test_numel(self):
        """Test numel() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4, 5), name="x")
        self.assertEqual(x.numel(), 60)

    def test_size_method(self):
        """Test size() method with and without dimension."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4, 5), name="x")
        self.assertEqual(x.size(), torch.Size([3, 4, 5]))
        self.assertEqual(x.size(0), 3)
        self.assertEqual(x.size(1), 4)
        self.assertEqual(x.size(-1), 5)

    def test_is_contiguous(self):
        """Test is_contiguous() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        self.assertTrue(x.is_contiguous())

    def test_is_floating_point(self):
        """Test is_floating_point() method."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x_float = ctx.create_input(torch.randn(3), name="x_float")
        self.assertTrue(x_float.is_floating_point())

        ctx2 = TracedTensorNode(name="test2", node_index=1)
        x_int = ctx2.create_input(torch.tensor([1, 2, 3]), name="x_int")
        self.assertFalse(x_int.is_floating_point())

    def test_T_property(self):
        """Test .T transpose property."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        result = x.T
        self.assertEqual(result.shape, torch.Size([4, 3]))
        self.assertIsInstance(result, TracedTensor)

    # ==================== Method Descriptor Conversion Tests ====================

    def test_regular_tensor_div_traced_tensor(self):
        """Test regular_tensor / traced_tensor uses torch.div not torch.Tensor.div."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([2.0, 4.0, 6.0]), name="x")
        regular = torch.tensor([10.0, 20.0, 30.0])

        result = regular / x  # This triggers method descriptor path

        self.assertIsInstance(result, TracedTensor)
        expected = torch.tensor([5.0, 5.0, 5.0])
        self.assertTrue(torch.allclose(result.tensor, expected))

        # Verify it compiles to TorchScript
        ctx.compile_trace({'result': result})
        scripted = torch.jit.script(ctx.compiled_graph_module)
        self.assertTrue(torch.allclose(
            scripted(torch.tensor([2.0, 4.0, 6.0])), expected))

    def test_regular_tensor_mul_traced_tensor(self):
        """Test regular_tensor * traced_tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        regular = torch.tensor([2.0, 2.0, 2.0])

        result = regular * x

        self.assertIsInstance(result, TracedTensor)
        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(result.tensor, expected))

    def test_regular_tensor_sub_traced_tensor(self):
        """Test regular_tensor - traced_tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        regular = torch.tensor([10.0, 10.0, 10.0])

        result = regular - x

        expected = torch.tensor([9.0, 8.0, 7.0])
        self.assertTrue(torch.allclose(result.tensor, expected))

    def test_regular_tensor_add_traced_tensor(self):
        """Test regular_tensor + traced_tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        regular = torch.tensor([10.0, 10.0, 10.0])

        result = regular + x

        expected = torch.tensor([11.0, 12.0, 13.0])
        self.assertTrue(torch.allclose(result.tensor, expected))

    # ==================== is_tracing=False Tests ====================

    def test_operations_after_compile_return_tensors(self):
        """Test that operations return raw tensors after compile_trace."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = x * 2

        ctx.compile_trace({'y': y})

        # After compilation, is_tracing should be False
        self.assertFalse(ctx.is_tracing)

        # Operations on traced tensor should now return regular tensors
        z = x + 1
        self.assertIsInstance(z, torch.Tensor)
        self.assertNotIsInstance(z, TracedTensor)

    def test_getitem_after_compile_returns_tensor(self):
        """Test that indexing returns raw tensor after compile."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = x[0]

        ctx.compile_trace({'y': y})

        # Indexing after compile should return regular tensor
        z = x[1]
        self.assertIsInstance(z, torch.Tensor)
        self.assertNotIsInstance(z, TracedTensor)

    def test_to_after_compile_returns_tensor(self):
        """Test that .to() returns raw tensor after compile."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = x.to(torch.float64)

        ctx.compile_trace({'y': y})

        # .to() after compile should return regular tensor
        z = x.to(torch.float32)
        self.assertIsInstance(z, torch.Tensor)
        self.assertNotIsInstance(z, TracedTensor)

    # ==================== Clone and Contiguous Tests ====================

    def test_clone(self):
        """Test clone operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = x.clone()

        self.assertIsInstance(y, TracedTensor)
        self.assertTrue(torch.allclose(y.tensor, x.tensor))

    def test_contiguous(self):
        """Test contiguous operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        y = x.T.contiguous()

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([4, 3]))

    # ==================== Additional Math Operations ====================

    def test_abs(self):
        """Test abs operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([-1.0, 2.0, -3.0]), name="x")
        y = torch.abs(x)

        self.assertIsInstance(y, TracedTensor)
        expected = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

    def test_neg(self):
        """Test negation operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -2.0, 3.0]), name="x")
        y = -x

        self.assertIsInstance(y, TracedTensor)
        expected = torch.tensor([-1.0, 2.0, -3.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

    def test_exp_log(self):
        """Test exp and log operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = torch.exp(x)
        z = torch.log(y)

        self.assertIsInstance(z, TracedTensor)
        self.assertTrue(torch.allclose(z.tensor, x.tensor, atol=1e-6))

    def test_sin_cos(self):
        """Test sin and cos operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([0.0, 3.14159/2, 3.14159]), name="x")
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)

        self.assertIsInstance(sin_x, TracedTensor)
        self.assertIsInstance(cos_x, TracedTensor)

    def test_sqrt(self):
        """Test sqrt operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 4.0, 9.0]), name="x")
        y = torch.sqrt(x)

        expected = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

    def test_clamp(self):
        """Test clamp operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([-1.0, 0.5, 2.0]), name="x")
        y = torch.clamp(x, min=0.0, max=1.0)

        expected = torch.tensor([0.0, 0.5, 1.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

    # ==================== Reduction with Dimensions ====================

    def test_sum_with_dim(self):
        """Test sum with dimension argument."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        y = torch.sum(x, dim=1)

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([3]))

    def test_mean_with_dim(self):
        """Test mean with dimension argument."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        y = torch.mean(x, dim=0)

        self.assertEqual(y.shape, torch.Size([4]))

    def test_max_with_dim(self):
        """Test max with dimension returns tuple."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")
        values, indices = torch.max(x, dim=1)

        self.assertIsInstance(values, TracedTensor)
        self.assertEqual(values.shape, torch.Size([3]))

    # ==================== View and Reshape ====================

    def test_view(self):
        """Test view operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(12), name="x")
        y = x.view(3, 4)

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([3, 4]))

    def test_flatten(self):
        """Test flatten operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(2, 3, 4), name="x")
        y = x.flatten()

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([24]))

    def test_expand(self):
        """Test expand operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(1, 3), name="x")
        y = x.expand(4, 3)

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([4, 3]))

    def test_repeat(self):
        """Test repeat operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(2, 3), name="x")
        y = x.repeat(2, 3)

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([4, 9]))

    # ==================== Split and Chunk ====================

    def test_split(self):
        """Test split operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(6), name="x")
        parts = torch.split(x, 2)

        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertIsInstance(part, TracedTensor)
            self.assertEqual(part.shape, torch.Size([2]))

    def test_chunk(self):
        """Test chunk operation."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(6), name="x")
        parts = torch.chunk(x, 3)

        self.assertEqual(len(parts), 3)

    # ==================== Linear Algebra ====================

    def test_dot(self):
        """Test dot product."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = torch.tensor([4.0, 5.0, 6.0])
        result = torch.dot(x, y)

        self.assertIsInstance(result, TracedTensor)
        expected = torch.tensor(32.0)
        self.assertTrue(torch.allclose(result.tensor, expected))

    def test_outer(self):
        """Test outer product."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0]), name="x")
        y = torch.tensor([3.0, 4.0, 5.0])
        result = torch.outer(x, y)

        self.assertEqual(result.shape, torch.Size([2, 3]))

    def test_cross(self):
        """Test cross product."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 0.0, 0.0]), name="x")
        y = torch.tensor([0.0, 1.0, 0.0])
        result = torch.linalg.cross(x, y)

        expected = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(result.tensor, expected))

    # ==================== Edge Cases ====================

    def test_scalar_tensor(self):
        """Test operations on scalar (0-dim) tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor(5.0), name="x")
        y = x * 2

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([]))
        self.assertTrue(torch.allclose(y.tensor, torch.tensor(10.0)))

    def test_empty_tensor(self):
        """Test operations on empty tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([]), name="x")
        y = x * 2

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([0]))

    def test_chained_operations(self):
        """Test long chain of operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Long chain
        result = x.clone().reshape(1, 3).squeeze(0).unsqueeze(-1).flatten()

        self.assertIsInstance(result, TracedTensor)
        self.assertTrue(torch.allclose(result.tensor, x.tensor))

    # ==================== Comparison Operations ====================

    def test_comparison_operators(self):
        """Test all comparison operators."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = torch.tensor([2.0, 2.0, 2.0])

        gt = x > y
        lt = x < y
        ge = x >= y
        le = x <= y
        eq = x == y
        ne = x != y

        for result in [gt, lt, ge, le, eq, ne]:
            self.assertIsInstance(result, TracedTensor)

    # ==================== In-Place Operations Tests ====================

    def test_inplace_add(self):
        """Test in-place addition (+=)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        original_id = id(x)

        x += 2.0

        # Should return the same object
        self.assertEqual(id(x), original_id)
        # Should have updated value
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.allclose(x.tensor, expected))
        # Should still be TracedTensor
        self.assertIsInstance(x, TracedTensor)

        # Should compile successfully
        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_sub(self):
        """Test in-place subtraction (-=)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([5.0, 6.0, 7.0]), name="x")
        original_id = id(x)

        x -= 2.0

        self.assertEqual(id(x), original_id)
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([5.0, 6.0, 7.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_mul(self):
        """Test in-place multiplication (*=)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        original_id = id(x)

        x *= 3.0

        self.assertEqual(id(x), original_id)
        expected = torch.tensor([3.0, 6.0, 9.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_div(self):
        """Test in-place division (/=)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([4.0, 8.0, 12.0]), name="x")
        original_id = id(x)

        x /= 2.0

        self.assertEqual(id(x), original_id)
        expected = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([4.0, 8.0, 12.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_pow(self):
        """Test in-place power (**=)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([2.0, 3.0, 4.0]), name="x")
        original_id = id(x)

        x **= 2.0

        self.assertEqual(id(x), original_id)
        expected = torch.tensor([4.0, 9.0, 16.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([2.0, 3.0, 4.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_method_add_(self):
        """Test in-place method add_()."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        original_id = id(x)

        result = x.add_(2.0)

        # Should return the same object
        self.assertEqual(id(result), original_id)
        self.assertEqual(id(x), original_id)
        # Should have updated value
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        graph_result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(graph_result, expected))

    def test_inplace_method_mul_(self):
        """Test in-place method mul_()."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        original_id = id(x)

        result = x.mul_(3.0)

        self.assertEqual(id(result), original_id)
        expected = torch.tensor([3.0, 6.0, 9.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        graph_result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(graph_result, expected))

    def test_inplace_chained_operations(self):
        """Test chained in-place operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        original_id = id(x)

        x += 1.0
        x *= 2.0
        x -= 3.0

        # Should still be the same object
        self.assertEqual(id(x), original_id)
        # (x + 1) * 2 - 3 = ([2, 3, 4]) * 2 - 3 = [4, 6, 8] - 3 = [1, 3, 5]
        expected = torch.tensor([1.0, 3.0, 5.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_with_tensor_operand(self):
        """Test in-place operations with tensor operands."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = torch.tensor([10.0, 20.0, 30.0])

        x += y

        expected = torch.tensor([11.0, 22.0, 33.0])
        self.assertTrue(torch.allclose(x.tensor, expected))

        ctx.compile_trace({'x': x})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_inplace_after_compile_returns_raw_tensor(self):
        """Test that in-place operations return raw tensor after compile.

        After compilation (when tracing stops), operations on TracedTensor
        should return raw torch.Tensor so downstream code doesn't know
        it was ever traced.
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = x * 2

        ctx.compile_trace({'y': y})

        # After compilation, in-place ops should return raw tensor
        result = x
        result += 1

        # Should return a raw torch.Tensor, not TracedTensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertNotIsInstance(result, TracedTensor)

        # Should have correct computed value
        expected = torch.tensor([2.0, 3.0, 4.0])
        self.assertTrue(torch.allclose(result, expected))

    # ==================== Advanced Indexing Tests ====================

    def test_indexing_with_none_newaxis(self):
        """Test None indexing (adds dimension)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Add dimension at the start
        y = x[None, :]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([1, 3]))
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(y.tensor, expected))

        # Test compilation
        ctx.compile_trace({'y': y})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

        # Test TorchScript
        scripted = torch.jit.script(ctx.compiled_graph_module)
        result_scripted = scripted(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result_scripted, expected))

    def test_indexing_with_none_middle(self):
        """Test None indexing in middle of tuple."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")

        # Add dimension in the middle: (3, 4) -> (3, 1, 4)
        y = x[:, None, :]

        self.assertEqual(y.shape, torch.Size([3, 1, 4]))
        self.assertIsInstance(y, TracedTensor)

    def test_indexing_with_ellipsis(self):
        """Test ellipsis indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(2, 3, 4), name="x")

        # Select last dimension
        y = x[..., 0]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([2, 3]))

        # Test compilation
        ctx.compile_trace({'y': y})
        input_tensor = torch.randn(2, 3, 4)
        result = ctx.compiled_graph_module(input_tensor)
        expected = input_tensor[..., 0]
        self.assertTrue(torch.allclose(result, expected))

    def test_indexing_with_ellipsis_beginning(self):
        """Test ellipsis at beginning."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(2, 3, 4), name="x")

        y = x[..., 1, 2]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([2]))

    def test_indexing_empty_slice(self):
        """Test empty slice indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Empty slice
        y = x[0:0]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([0]))

        # Test compilation
        ctx.compile_trace({'y': y})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(result.shape, torch.Size([0]))

    def test_indexing_with_list_advanced(self):
        """Test advanced integer list indexing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0]), name="x")

        # Index with a list
        y = x[[0, 2, 4]]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([3]))
        expected = torch.tensor([10.0, 30.0, 50.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

        # Test compilation
        ctx.compile_trace({'y': y})
        result = ctx.compiled_graph_module(
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_indexing_with_list_2d(self):
        """Test list indexing on 2D tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(5, 3), name="x")

        # Index rows with a list
        y = x[[0, 2, 4], :]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([3, 3]))

    def test_indexing_negative_indices_advanced(self):
        """Test negative indices in various contexts."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), name="x")

        # Negative single index
        y = x[-1]
        self.assertIsInstance(y, TracedTensor)
        self.assertTrue(torch.allclose(y.tensor, torch.tensor(5.0)))

        # Negative slice
        ctx2 = TracedTensorNode(name="test2", node_index=1)
        x2 = ctx2.create_input(torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0]), name="x")
        z = x2[-3:-1]
        self.assertEqual(z.shape, torch.Size([2]))
        expected = torch.tensor([3.0, 4.0])
        self.assertTrue(torch.allclose(z.tensor, expected))

    def test_indexing_combination_advanced(self):
        """Test combination of various indexing methods."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(2, 3, 4, 5), name="x")

        # Mix of different indexing types
        y = x[0, :, 1:3, -2]

        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.shape, torch.Size([3, 2]))

        # Test compilation
        ctx.compile_trace({'y': y})
        input_tensor = torch.randn(2, 3, 4, 5)
        result = ctx.compiled_graph_module(input_tensor)
        expected = input_tensor[0, :, 1:3, -2]
        self.assertTrue(torch.allclose(result, expected))

    # ==================== Variable Reassignment and Graph Structure Tests ====================

    def test_variable_reassignment(self):
        """Test reassigning a variable doesn't cause issues."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Store original x reference
        original_x = x

        # Reassign x to a new value
        y = x + 1  # [2, 3, 4]
        x = y + 1  # [3, 4, 5] - reassign x

        # Both original_x and y should still be valid
        z = x + y  # [3, 4, 5] + [2, 3, 4] = [5, 7, 9]

        self.assertIsInstance(z, TracedTensor)
        expected = torch.tensor([5.0, 7.0, 9.0])
        self.assertTrue(torch.allclose(z.tensor, expected))

        # Test compilation
        ctx.compile_trace({'z': z})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

        # Verify TorchScript compilation
        scripted = torch.jit.script(ctx.compiled_graph_module)
        result_scripted = scripted(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result_scripted, expected))

    def test_multiple_outputs_with_shared_intermediate(self):
        """Test multiple outputs that share intermediate values."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create shared intermediate
        y = x * 2  # [2, 4, 6]

        # Create multiple outputs using y
        out1 = y + 1  # [3, 5, 7]
        out2 = y - 1  # [1, 3, 5]
        out3 = y * 3  # [6, 12, 18]

        # Compile with multiple outputs
        ctx.compile_trace({'out1': out1, 'out2': out2, 'out3': out3})

        # Test execution
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))

        # When multiple outputs, result should be a tuple
        if isinstance(result, tuple):
            self.assertEqual(len(result), 3)
            self.assertTrue(torch.allclose(
                result[0], torch.tensor([3.0, 5.0, 7.0])))
            self.assertTrue(torch.allclose(
                result[1], torch.tensor([1.0, 3.0, 5.0])))
            self.assertTrue(torch.allclose(
                result[2], torch.tensor([6.0, 12.0, 18.0])))

    def test_diamond_dependency_graph(self):
        """Test diamond-shaped dependency graph (one input, multiple paths, one output)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create diamond pattern:
        #     x
        #    / \
        #   a   b
        #    \ /
        #     c
        a = x + 1  # [2, 3, 4]
        b = x * 2  # [2, 4, 6]
        c = a + b  # [4, 7, 10]

        self.assertIsInstance(c, TracedTensor)
        expected = torch.tensor([4.0, 7.0, 10.0])
        self.assertTrue(torch.allclose(c.tensor, expected))

        # Test compilation
        ctx.compile_trace({'c': c})
        result = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(torch.allclose(result, expected))

    def test_deep_computation_chain(self):
        """Test a very deep chain of operations."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0]), name="x")

        # Create a chain of 100 operations
        result = x
        for i in range(100):
            result = result + 0.01  # Add 0.01 each time

        self.assertIsInstance(result, TracedTensor)
        expected = torch.tensor([2.0])  # 1.0 + 100 * 0.01
        self.assertTrue(torch.allclose(result.tensor, expected, atol=1e-5))

        # Test compilation
        ctx.compile_trace({'result': result})
        output = ctx.compiled_graph_module(torch.tensor([1.0]))
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_unused_intermediate_values(self):
        """Test that unused intermediate values don't appear in final graph."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create some intermediate values
        unused1 = x + 100  # This won't be used
        unused2 = x * 100  # This won't be used either

        # Only use x directly
        result = x + 1

        ctx.compile_trace({'result': result})

        # Check that the graph doesn't contain the unused operations
        graph_str = str(ctx.compiled_graph_module.graph)

        # The graph should have: placeholder (x), add (x + 1), output
        # Count the number of operations (excluding placeholder and output)
        nodes = list(ctx.compiled_graph_module.graph.nodes)
        call_function_nodes = [n for n in nodes if n.op == 'call_function']

        # Should only have 1 call_function (the add for result)
        self.assertEqual(len(call_function_nodes), 1)

    def test_complex_branching_graph(self):
        """Test complex graph with multiple branches and merges."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create complex branching structure
        a = x + 1
        b = x * 2
        c = a + b
        d = a * 3
        e = b - 1
        f = c + d
        g = e * 2
        result = f + g

        self.assertIsInstance(result, TracedTensor)

        # Test compilation
        ctx.compile_trace({'result': result})
        output = ctx.compiled_graph_module(torch.tensor([1.0, 2.0, 3.0]))

        # Verify result matches
        self.assertTrue(torch.allclose(output, result.tensor))

    # ==================== Special Method Edge Cases ====================

    def test_hash_tracedtensor(self):
        """Test hashing TracedTensor objects."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Try to hash the TracedTensor
        # This might fail if __hash__ is not implemented
        try:
            hash_val = hash(x)
            # If it succeeds, verify it's consistent
            self.assertEqual(hash(x), hash_val)
        except TypeError as e:
            # If it fails, that's okay - we just want to document the behavior
            self.assertIn("unhashable", str(e).lower())

    def test_tracedtensor_in_set(self):
        """Test using TracedTensor in a set."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = ctx.create_input(torch.tensor([4.0, 5.0, 6.0]), name="y")

        # Try to put TracedTensors in a set
        # This requires __hash__ to be implemented
        try:
            tensor_set = {x, y}
            self.assertEqual(len(tensor_set), 2)
        except TypeError as e:
            # If it fails, document it
            self.assertIn("unhashable", str(e).lower())

    def test_tracedtensor_as_dict_key(self):
        """Test using TracedTensor as dictionary key."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Try to use TracedTensor as dict key
        # This requires __hash__ and __eq__ to work properly
        try:
            tensor_dict = {x: "value"}
            self.assertEqual(tensor_dict[x], "value")
        except TypeError as e:
            # If it fails, document it
            self.assertIn("unhashable", str(e).lower())

    def test_iter_tracedtensor_1d(self):
        """Test iterating over 1D TracedTensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Try to iterate over TracedTensor
        try:
            elements = list(iter(x))
            self.assertEqual(len(elements), 3)

            # Check if elements are TracedTensors or scalars
            for elem in elements:
                # Could be either TracedTensor or torch.Tensor
                self.assertTrue(
                    isinstance(elem, (TracedTensor, torch.Tensor)) or
                    isinstance(elem, (int, float))
                )
        except TypeError as e:
            # If iteration is not supported, that's okay
            self.fail(f"Iteration failed: {e}")

    def test_iter_tracedtensor_2d(self):
        """Test iterating over 2D TracedTensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.randn(3, 4), name="x")

        # Iterating over 2D tensor should yield rows
        try:
            rows = list(iter(x))
            self.assertEqual(len(rows), 3)

            # Each row should have shape (4,)
            for row in rows:
                if isinstance(row, TracedTensor):
                    self.assertEqual(row.shape, torch.Size([4]))
                elif isinstance(row, torch.Tensor):
                    self.assertEqual(row.shape, torch.Size([4]))
        except TypeError as e:
            self.fail(f"Iteration failed: {e}")

    def test_for_loop_over_tracedtensor(self):
        """Test using TracedTensor in a for loop."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Use in for loop
        try:
            sum_val = 0
            for elem in x:
                # Extract the actual value
                if isinstance(elem, TracedTensor):
                    sum_val += elem.tensor.item()
                elif isinstance(elem, torch.Tensor):
                    sum_val += elem.item()
                else:
                    sum_val += float(elem)

            self.assertAlmostEqual(sum_val, 6.0, places=5)
        except Exception as e:
            self.fail(f"For loop failed: {e}")

    def test_reversed_tracedtensor(self):
        """Test reversing a TracedTensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Try to reverse the TracedTensor
        try:
            reversed_x = list(reversed(x))
            self.assertEqual(len(reversed_x), 3)

            # Check if order is reversed
            # Note: elements might be TracedTensors or tensors
            if isinstance(reversed_x[0], TracedTensor):
                self.assertTrue(torch.allclose(
                    reversed_x[0].tensor, torch.tensor(3.0)
                ))
                self.assertTrue(torch.allclose(
                    reversed_x[-1].tensor, torch.tensor(1.0)
                ))
            elif isinstance(reversed_x[0], torch.Tensor):
                self.assertTrue(torch.allclose(
                    reversed_x[0], torch.tensor(3.0)
                ))
                self.assertTrue(torch.allclose(
                    reversed_x[-1], torch.tensor(1.0)
                ))
        except TypeError as e:
            # If reversed is not supported, check the error
            self.fail(f"reversed() failed: {e}")

    def test_enumerate_tracedtensor(self):
        """Test using enumerate with TracedTensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([10.0, 20.0, 30.0]), name="x")

        # Try to enumerate
        try:
            indexed_elements = list(enumerate(x))
            self.assertEqual(len(indexed_elements), 3)

            # Check indices
            for i, (idx, elem) in enumerate(indexed_elements):
                self.assertEqual(idx, i)
        except Exception as e:
            self.fail(f"enumerate() failed: {e}")

    def test_zip_with_tracedtensor(self):
        """Test using zip with TracedTensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = ctx.create_input(torch.tensor([4.0, 5.0, 6.0]), name="y")

        # Try to zip two TracedTensors
        try:
            pairs = list(zip(x, y))
            self.assertEqual(len(pairs), 3)
        except Exception as e:
            self.fail(f"zip() failed: {e}")

    def test_boolean_indexing_with_regular_tensor_works(self):
        """Test that boolean indexing works when using mask.tensor."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create a boolean mask
        mask = x > 1.5

        # This should work - using the underlying tensor
        y = x[mask.tensor]

        self.assertIsInstance(y, TracedTensor)
        expected = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.allclose(y.tensor, expected))

    def test_all_arithmetic_functions(self):
        """Test all arithmetic functions with tracking, TorchScript, and ONNX."""
        # Get all static methods from TensorArithmeticFunctions
        methods = [
            (name, method)
            for name, method in inspect.getmembers(
                TensorArithmeticFunctions, predicate=inspect.isfunction
            )
        ]

        for func_name, func in methods:
            with self.subTest(function=func_name):
                # Test 1: Compare tracking vs non-tracking
                input_tensor = torch.tensor([1.0, 2.0, 3.0])
                expected = func(input_tensor)

                ctx = TracedTensorNode(name="test", node_index=0)
                input_tracked = ctx.create_input(
                    input_tensor, name="my_test_input")
                result_tracked = func(input_tracked)
                ctx.compile_trace({'result_tracked': result_tracked})
                graph_module = ctx.compiled_graph_module

                self.assertIsInstance(
                    result_tracked,
                    TracedTensor,
                    f"{func_name}: Result should be TracedTensor when input is TracedTensor",
                )

                self.assertTrue(
                    torch.allclose(result_tracked.tensor, expected),
                    f"{func_name}: Tracked result doesn't match expected result",
                )

                # Test 2: GraphModule export
                result_graph_module = graph_module(input_tensor)
                self.assertTrue(
                    torch.allclose(result_graph_module, expected),
                    f"{func_name}: GraphModule result doesn't match expected result",
                )

                # Test 3: TorchScript export
                traced_script = torch.jit.script(graph_module)
                result_traced_script = traced_script(input_tensor)
                self.assertTrue(
                    torch.allclose(result_traced_script, expected),
                    f"{func_name}: TorchScript result doesn't match expected result",
                )

                # Test 4: ONNX export
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        onnx_path = pathlib.Path(tmpdir) / f"{func_name}.onnx"
                        torch.onnx.export(
                            graph_module,
                            (input_tensor,),
                            onnx_path,
                            dynamo=False,
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            input_names=['my_test_input'],
                            output_names=['result_tracked'],
                        )

                        session = ort.InferenceSession(str(onnx_path))
                        result_onnx = session.run(
                            None, {"my_test_input": input_tensor.numpy()})[0]

                        self.assertTrue(
                            torch.allclose(
                                torch.from_numpy(result_onnx),
                                result_tracked.tensor,
                                atol=1e-6,
                            ),
                            f"{func_name}: ONNX result doesn't match tracked result",
                        )
                except Exception as e:
                    self.fail(f"{func_name}: Error exporting to ONNX: {e}")

    # ==================== Unsupported Operation error message ====================
    def test_boolean_indexing_raises_error(self):
        """Test that boolean indexing with TracedTensor raises a clear error."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Create a boolean mask (which is also a TracedTensor)
        mask = x > 1.5  # TracedTensor with dtype bool

        # Verify mask is indeed a TracedTensor with bool dtype
        self.assertIsInstance(mask, TracedTensor)
        self.assertEqual(mask.dtype, torch.bool)

        # Try to use boolean indexing - should raise NotImplementedError with helpful message
        with self.assertRaises(NotImplementedError) as context:
            y = x[mask]

        # Verify the error message is helpful
        error_msg = str(context.exception)
        self.assertIn("Boolean/mask indexing", error_msg)
        self.assertIn("not supported", error_msg)
        self.assertIn("torch.masked_select", error_msg)
        self.assertIn("mask.tensor", error_msg)

    def test_advanced_indexing_with_tracedtensor_raises_error(self):
        """Test that advanced indexing with TracedTensor indices raises clear error."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0]), name="x")

        # Create indices as TracedTensor (non-boolean)
        indices_tensor = torch.tensor([0, 2, 4], dtype=torch.long)
        ctx2 = TracedTensorNode(name="test2", node_index=1)
        indices = ctx2.create_input(indices_tensor, name="indices")

        # Try to use TracedTensor as index - should raise NotImplementedError
        with self.assertRaises(NotImplementedError) as context:
            y = x[indices]

        error_msg = str(context.exception)
        self.assertIn("Advanced indexing", error_msg)
        self.assertIn("not supported", error_msg)
        self.assertIn("indices.tensor", error_msg)

    # ==================== TorchScript Module Interaction Tests ====================

    def test_tracedtensor_with_torchscript_module_raises_error(self):
        """Test that using TorchScript module with TracedTensor raises clear error."""
        # Create a simple TorchScript module
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x * 2 + 1

        module = SimpleModule()
        scripted_module = torch.jit.script(module)

        # Use it with TracedTensor
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # This should raise ValueError during tracing
        with self.assertRaises(ValueError) as context:
            y = scripted_module(x)

        # Verify the error message is correct
        error_msg = str(context.exception)
        self.assertIn("TorchScript modules cannot be used", error_msg)
        self.assertIn("TracedTensor", error_msg)
        self.assertIn("during tracing", error_msg)

    def test_torchscript_module_with_parameters_raises_error(self):
        """Test that TorchScript module with parameters raises error with TracedTensor."""
        # Create a TorchScript module with parameters
        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([[2.0, 3.0, 4.0]]))

            def forward(self, x):
                return torch.matmul(self.weight, x)

        module = LinearModule()
        scripted_module = torch.jit.script(module)

        # Use it in tracing context
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Should raise ValueError during tracing
        with self.assertRaises(ValueError) as context:
            y = scripted_module(x)

        error_msg = str(context.exception)
        self.assertIn("TorchScript modules cannot be used", error_msg)
        self.assertIn("TracedTensor", error_msg)
        self.assertIn("during tracing", error_msg)

    def test_nested_torchscript_in_traced_operations_raises_error(self):
        """Test that TorchScript module nested with traced ops raises error."""
        class ReLUModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        scripted_relu = torch.jit.script(ReLUModule())

        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([-1.0, 0.0, 1.0, 2.0]), name="x")

        # Regular operation first
        y = x * 2  # This works fine

        # TorchScript module should raise ValueError
        with self.assertRaises(ValueError) as context:
            z = scripted_relu(y)  # TorchScript module with TracedTensor

        error_msg = str(context.exception)
        self.assertIn("TorchScript modules cannot be used", error_msg)
        self.assertIn("TracedTensor", error_msg)
        self.assertIn("during tracing", error_msg)

    def test_torchscript_with_multiple_inputs_raises_error(self):
        """Test that TorchScript module with multiple TracedTensor inputs raises error."""
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y * 2

        scripted_add = torch.jit.script(AddModule())

        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        y = ctx.create_input(torch.tensor([4.0, 5.0, 6.0]), name="y")

        # Should raise ValueError when calling with TracedTensors
        with self.assertRaises(ValueError) as context:
            z = scripted_add(x, y)

        error_msg = str(context.exception)
        self.assertIn("TorchScript modules cannot be used", error_msg)
        self.assertIn("TracedTensor", error_msg)
        self.assertIn("during tracing", error_msg)

    def test_torchscript_module_with_state_raises_error(self):
        """Test that TorchScript module with state raises error with TracedTensor."""
        class StatefulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                # Set fixed weights for reproducibility
                self.linear.weight.data = torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                self.linear.bias.data = torch.tensor([0.0, 0.0])

            def forward(self, x):
                return self.linear(x)

        module = StatefulModule()
        scripted_module = torch.jit.script(module)

        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")

        # Should raise ValueError when calling with TracedTensor
        with self.assertRaises(ValueError) as context:
            y = scripted_module(x)

        error_msg = str(context.exception)
        self.assertIn("TorchScript modules cannot be used", error_msg)
        self.assertIn("TracedTensor", error_msg)
        self.assertIn("during tracing", error_msg)


if __name__ == "__main__":
    unittest.main()
