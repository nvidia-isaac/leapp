#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for conversion between TracedTensor, TracedNpArray, and TracedWpArray.

These tests verify that:
1. Conversions produce the correct traced type
2. Data values are preserved correctly
3. FX graphs compile and execute correctly with torch tensor I/O
"""

import unittest
import numpy as np
import torch

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedTensor, TracedNpArray, TracedWpArray, wp
from leapp.export_manager import ExportManager


def _install_patches():
    ExportManager().patcher.install()


def _uninstall_patches():
    ExportManager().patcher.uninstall()


class TestTracedTensorToNumpy(unittest.TestCase):
    """Test conversions from TracedTensor to numpy/TracedNpArray."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_numpy_method_returns_traced_np_array(self):
        """Test TracedTensor.numpy() returns TracedNpArray when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        y = x.numpy()
        
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_numpy_method_returns_ndarray_after_compile(self):
        """Test TracedTensor.numpy() returns plain ndarray after compile."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        ctx.compile_trace({'output': x * 2})
        
        y = x.numpy()
        
        self.assertIsInstance(y, np.ndarray)
        self.assertNotIsInstance(y, TracedNpArray)

    def test_np_array_returns_traced_np_array(self):
        """Test np.array(TracedTensor) returns TracedNpArray when patched."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        y = np.array(x)
        
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_np_asarray_returns_traced_np_array(self):
        """Test np.asarray(TracedTensor) returns TracedNpArray when patched."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        y = np.asarray(x)
        
        self.assertIsInstance(y, TracedNpArray)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0]))

    def test_np_array_with_dtype(self):
        """Test np.array(TracedTensor, dtype=...) preserves tracing when patched."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), name="x")
        
        y = np.array(x, dtype=np.float32)
        
        self.assertIsInstance(y, TracedNpArray)
        self.assertEqual(y.dtype, np.float32)


class TestNumpyToTracedTensor(unittest.TestCase):
    """Test conversions from TracedNpArray to TracedTensor."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_torch_from_numpy_returns_traced_tensor(self):
        """Test torch.from_numpy(TracedNpArray) returns TracedTensor when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="x")
        
        y = torch.from_numpy(x)
        
        self.assertIsInstance(y, TracedTensor)
        self.assertTrue(torch.allclose(y.tensor, torch.tensor([1.0, 2.0, 3.0])))

    def test_torch_tensor_returns_traced_tensor(self):
        """Test torch.tensor(TracedNpArray) returns TracedTensor when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="x")
        
        y = torch.tensor(x)
        
        self.assertIsInstance(y, TracedTensor)
        self.assertTrue(torch.allclose(y.tensor, torch.tensor([1.0, 2.0, 3.0])))

    def test_torch_as_tensor_returns_traced_tensor(self):
        """Test torch.as_tensor(TracedNpArray) returns TracedTensor when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="x")
        
        y = torch.as_tensor(x)
        
        self.assertIsInstance(y, TracedTensor)
        self.assertTrue(torch.allclose(y.tensor, torch.tensor([1.0, 2.0, 3.0])))

    def test_torch_from_numpy_returns_tensor_after_compile(self):
        """Test torch.from_numpy returns plain tensor after compile."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="x")
        ctx.compile_trace({'output': x * 2})
        
        # After compile, x is no longer tracing
        y = torch.from_numpy(x.view(np.ndarray))
        
        self.assertIsInstance(y, torch.Tensor)
        self.assertNotIsInstance(y, TracedTensor)


class TestConversionFXGraphCompilation(unittest.TestCase):
    """Test that conversions work correctly in compiled FX graphs.
    
    All tests use torch.Tensor as input and output for the compiled graph.
    """

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_tensor_to_numpy_and_back(self):
        """Test: tensor → numpy → operations → tensor → output.
        
        Verifies that converting to numpy and back preserves tracing.
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        # Convert to numpy, do operation, convert back
        np_x = x.numpy()
        np_y = np_x * 2
        y = torch.from_numpy(np_y)
        
        # Compile
        ctx.compile_trace({'output': y})
        graph_module = ctx.m
        
        # Test with torch tensor input
        input_tensor = torch.tensor([4.0, 5.0, 6.0])
        expected = input_tensor * 2
        output = graph_module(input_tensor)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.allclose(output, expected))

    def test_numpy_operations_in_trace(self):
        """Test: tensor → numpy → numpy ops → tensor → output.
        
        Verifies numpy operations are recorded correctly.
        Note: Use keepdims=True for reductions to preserve TracedNpArray
        (scalar reductions return numpy scalars, losing tracing).
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0, 4.0]), name="x")
        
        # Convert to numpy and do numpy-specific operations
        np_x = x.numpy()
        np_y = np.sum(np_x, keepdims=True)  # Reduction with keepdims to preserve array
        y = torch.as_tensor(np_y)
        
        # Compile
        ctx.compile_trace({'output': y})
        graph_module = ctx.m
        
        # Test
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([10.0])  # sum with keepdims
        output = graph_module(input_tensor)
        
        self.assertTrue(torch.allclose(output, expected))

    def test_numpy_input_to_tensor_output(self):
        """Test: numpy input → operations → tensor output.
        
        Graph takes torch tensor, internally uses numpy ops.
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0]), name="x")
        
        # Do numpy operation then convert to tensor
        np_y = x + 10
        y = torch.from_numpy(np_y)
        
        # Compile
        ctx.compile_trace({'output': y})
        graph_module = ctx.m
        
        # Test with torch tensor input
        input_tensor = torch.tensor([4.0, 5.0, 6.0])
        expected = input_tensor + 10
        output = graph_module(input_tensor)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.allclose(output, expected))

    def test_chained_conversions(self):
        """Test: tensor → numpy → tensor → numpy → tensor.
        
        Multiple conversions should preserve tracing throughout.
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        # Chain: tensor → numpy → tensor → numpy → tensor
        np1 = x.numpy()
        t1 = torch.from_numpy(np1 * 2)
        np2 = t1.numpy()
        y = torch.as_tensor(np2 + 1)
        
        # Compile
        ctx.compile_trace({'output': y})
        graph_module = ctx.m
        
        # Test
        input_tensor = torch.tensor([4.0, 5.0, 6.0])
        expected = (input_tensor * 2) + 1
        output = graph_module(input_tensor)
        
        self.assertTrue(torch.allclose(output, expected))

    def test_np_array_conversion_in_trace(self):
        """Test np.array() conversion within a trace (with patching)."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0]), name="x")
        
        # Use np.array() for conversion (patched to preserve tracing)
        np_x = np.array(x)
        np_y = np_x * 3
        y = torch.from_numpy(np_y)
        
        # Compile
        ctx.compile_trace({'output': y})
        graph_module = ctx.m
        
        # Test
        input_tensor = torch.tensor([2.0, 3.0, 4.0])
        expected = input_tensor * 3
        output = graph_module(input_tensor)
        
        self.assertTrue(torch.allclose(output, expected))


class TestConversionWithDtype(unittest.TestCase):
    """Test conversions with dtype changes."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_numpy_with_dtype_change(self):
        """Test conversion with dtype change preserves tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), name="x")
        
        # Convert to float32 numpy using np.array with patching
        np_x = np.array(x, dtype=np.float32)
        
        self.assertIsInstance(np_x, TracedNpArray)
        self.assertEqual(np_x.dtype, np.float32)

    def test_torch_tensor_with_dtype(self):
        """Test torch.tensor with dtype preserves tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float64), name="x")
        
        # Convert to float32 tensor
        y = torch.tensor(x, dtype=torch.float32)
        
        self.assertIsInstance(y, TracedTensor)
        self.assertEqual(y.dtype, torch.float32)


class TestTracedTensorIdentityPreservation(unittest.TestCase):
    """Test that torch functions preserve TracedTensor identity/type correctly.
    
    Tests verify that:
    - torch.as_tensor returns same TracedTensor (no copy)
    - torch.tensor returns new TracedTensor with same proxy (copy semantics)
    """

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_torch_as_tensor_preserves_identity(self):
        """Test torch.as_tensor returns same TracedTensor object."""
        ctx = TracedTensorNode(name="test", node_index=0)
        traced = ctx.create_input(torch.randn(2, 3), name="x")
        
        result = torch.as_tensor(traced)
        
        self.assertIsInstance(result, TracedTensor)
        self.assertIs(result, traced)  # Same object (no copy)

    def test_torch_tensor_creates_new_traced_tensor(self):
        """Test torch.tensor returns new TracedTensor with same proxy.
        
        torch.tensor() always creates a copy, so we get a NEW TracedTensor
        (not the same object), but it preserves the proxy/context for tracing.
        """
        ctx = TracedTensorNode(name="test", node_index=0)
        traced = ctx.create_input(torch.randn(2, 3), name="x")
        
        result = torch.tensor(traced)
        
        self.assertIsInstance(result, TracedTensor)
        self.assertIsNot(result, traced)  # Different object (copy semantics)
        self.assertIs(result.proxy, traced.proxy)  # Same proxy (tracing preserved)
        self.assertTrue(torch.allclose(result.tensor, traced.tensor))


@unittest.skipIf(wp is None, "warp-lang is not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for warp conversion tests")
class TestTracedTensorToWarp(unittest.TestCase):
    """Test conversions from TracedTensor to TracedWpArray."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_wp_from_torch_returns_traced_wp_array(self):
        """Test wp.from_torch(TracedTensor) returns TracedWpArray when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], device="cuda"), name="x")

        y = wp.from_torch(x)

        self.assertIsInstance(y, TracedWpArray)
        self.assertIs(y.proxy, x.proxy)
        self.assertTrue(torch.allclose(wp.to_torch(y), x.tensor))

    def test_wp_array_returns_traced_wp_array(self):
        """Test wp.array(TracedTensor, dtype=...) returns TracedWpArray when patched."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], device="cuda"), name="x")

        y = wp.array(x, dtype=wp.float32)

        self.assertIsInstance(y, TracedWpArray)
        self.assertIs(y.proxy, x.proxy)
        self.assertTrue(torch.allclose(wp.to_torch(y), x.tensor))


@unittest.skipIf(wp is None, "warp-lang is not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for warp conversion tests")
class TestWarpToTracedTensor(unittest.TestCase):
    """Test conversions from TracedWpArray to TracedTensor."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_wp_to_torch_returns_traced_tensor(self):
        """Test wp.to_torch(TracedWpArray) returns TracedTensor when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x_torch = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], device="cuda"), name="x")
        x = wp.from_torch(x_torch)

        y = wp.to_torch(x)

        self.assertIsInstance(y, TracedTensor)
        self.assertIs(y.proxy, x.proxy)
        self.assertTrue(torch.allclose(y.tensor, x_torch.tensor))


@unittest.skipIf(wp is None, "warp-lang is not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for warp conversion tests")
class TestTracedNumpyToWarp(unittest.TestCase):
    """Test conversions from TracedNpArray to TracedWpArray."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_wp_from_numpy_returns_traced_wp_array(self):
        """Test wp.from_numpy(TracedNpArray) returns TracedWpArray when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), name="x")

        y = wp.from_numpy(x)

        self.assertIsInstance(y, TracedWpArray)
        self.assertIs(y.proxy, x.proxy)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, wp.float32)


@unittest.skipIf(wp is None, "warp-lang is not installed")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required for warp conversion tests")
class TestWarpToTracedNumpy(unittest.TestCase):
    """Test conversions from TracedWpArray to TracedNpArray."""

    def setUp(self):
        """Apply patches before each test."""
        _install_patches()

    def tearDown(self):
        """Remove patches after each test."""
        _uninstall_patches()

    def test_wp_array_numpy_returns_traced_np_array(self):
        """Test TracedWpArray.numpy() returns TracedNpArray when tracing."""
        ctx = TracedTensorNode(name="test", node_index=0)
        x_torch = ctx.create_input(torch.tensor([1.0, 2.0, 3.0], device="cuda"), name="x")
        x = wp.from_torch(x_torch)

        y = x.numpy()

        self.assertIsInstance(y, TracedNpArray)
        self.assertIs(y.proxy, x.proxy)
        np.testing.assert_array_almost_equal(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))


class TestPatchingBehavior(unittest.TestCase):
    """Test the patching mechanism itself."""

    def tearDown(self):
        """Ensure patches are removed after each test."""
        _uninstall_patches()

    def test_patches_can_be_toggled(self):
        """Test that patches can be applied and removed."""
        patcher = ExportManager().patcher

        # Start clean
        _uninstall_patches()
        self.assertFalse(patcher.installed)

        # Apply patches
        _install_patches()
        self.assertTrue(patcher.installed)

        # Remove patches
        _uninstall_patches()
        self.assertFalse(patcher.installed)


if __name__ == '__main__':
    unittest.main()

