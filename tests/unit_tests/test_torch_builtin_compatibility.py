#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for torch builtin function compatibility with TracedTensor.

These tests verify that common torch functions (especially torch.nn.functional.*)
can be traced and compiled correctly. This is important because:
1. Many torch functions are builtin_function_or_method (C++ extensions)
2. Some operations involve TorchScript modules that don't trace well
3. We need to ensure FX graphs compile and execute correctly

Each test:
1. Creates a TracedTensor input
2. Applies the torch function
3. Compiles the trace to an FX graph
4. Verifies the compiled graph produces correct output
"""

import unittest
from functools import partial

import torch
import torch.nn.functional as F

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedTensor


class TestFunctional(unittest.TestCase):
    """Test torch.nn.functional operations trace and compile correctly."""
    
    # Number of random inputs to test after compiling
    NUM_TEST_INPUTS = 5

    def _run_trace_and_compile(self, func, input_shape, test_name, **func_kwargs):
        """Helper to trace a function and verify compiled output matches.
        
        Compiles once, then tests with NUM_TEST_INPUTS random inputs.
        
        Args:
            func: Function to apply (e.g., F.relu, or partial(F.linear, weight=w))
            input_shape: Shape of input tensor
            test_name: Name for the trace context
            **func_kwargs: Additional kwargs to pass to func
        """
        input_tensor = torch.randn(*input_shape)
        
        ctx = TracedTensorNode(name=test_name, node_index=0)
        traced_input = ctx.create_input(input_tensor.clone(), name="x")
        
        output = func(traced_input, **func_kwargs)
        self.assertIsInstance(output, TracedTensor)
        
        ctx.compile_trace({'output': output})
        graph_module = ctx.compiled_graph_module
        
        # Test with multiple random inputs
        for i in range(self.NUM_TEST_INPUTS):
            test_input = torch.randn(*input_shape)
            expected = func(test_input, **func_kwargs)
            actual = graph_module(test_input)
            
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-5),
                f"{test_name} output mismatch on input {i+1}/{self.NUM_TEST_INPUTS}"
            )

    # =========================================================================
    # Linear
    # =========================================================================
    
    def test_linear(self):
        """Test F.linear with weight and bias."""
        weight = torch.randn(16, 8)
        bias = torch.randn(16)
        func = partial(F.linear, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (4, 8), "test_linear")

    def test_linear_no_bias(self):
        """Test F.linear without bias."""
        weight = torch.randn(16, 8)
        func = partial(F.linear, weight=weight, bias=None)
        self._run_trace_and_compile(func, (4, 8), "test_linear_no_bias")

    # =========================================================================
    # Activations
    # =========================================================================

    def test_relu(self):
        """Test F.relu activation."""
        self._run_trace_and_compile(F.relu, (4, 8), "test_relu")

    def test_relu_inplace(self):
        """Test F.relu with inplace=True."""
        # Note: inplace may behave differently, test non-inplace for tracing
        self._run_trace_and_compile(F.relu, (4, 8), "test_relu_inplace", inplace=False)

    def test_leaky_relu_default(self):
        """Test F.leaky_relu with default negative_slope=0.01."""
        self._run_trace_and_compile(F.leaky_relu, (4, 8), "test_leaky_relu")

    def test_leaky_relu_custom_slope(self):
        """Test F.leaky_relu with custom negative_slope."""
        self._run_trace_and_compile(
            F.leaky_relu, (4, 8), "test_leaky_relu_custom", 
            negative_slope=0.2
        )

    def test_gelu(self):
        """Test F.gelu activation."""
        self._run_trace_and_compile(F.gelu, (4, 8), "test_gelu")

    def test_tanh(self):
        """Test F.tanh activation."""
        self._run_trace_and_compile(torch.tanh, (4, 8), "test_tanh")

    def test_sigmoid(self):
        """Test F.sigmoid activation."""
        self._run_trace_and_compile(torch.sigmoid, (4, 8), "test_sigmoid")

    def test_silu(self):
        """Test F.silu (Swish) activation."""
        self._run_trace_and_compile(F.silu, (4, 8), "test_silu")

    def test_elu(self):
        """Test F.elu activation."""
        self._run_trace_and_compile(F.elu, (4, 8), "test_elu")

    # =========================================================================
    # Probability / Softmax
    # =========================================================================

    def test_softmax_dim0(self):
        """Test F.softmax along dim=0."""
        self._run_trace_and_compile(F.softmax, (4, 8), "test_softmax_dim0", dim=0)

    def test_softmax_dim1(self):
        """Test F.softmax along dim=1 (common for classification)."""
        self._run_trace_and_compile(F.softmax, (4, 8), "test_softmax_dim1", dim=1)

    def test_softmax_last_dim(self):
        """Test F.softmax along dim=-1."""
        self._run_trace_and_compile(F.softmax, (4, 8, 16), "test_softmax_last", dim=-1)

    def test_log_softmax(self):
        """Test F.log_softmax."""
        self._run_trace_and_compile(F.log_softmax, (4, 8), "test_log_softmax", dim=-1)

    # =========================================================================
    # Normalization
    # =========================================================================

    def test_normalize_l2_dim1(self):
        """Test F.normalize with L2 norm along dim=1."""
        self._run_trace_and_compile(
            F.normalize, (4, 8), "test_normalize_l2", 
            p=2.0, dim=1
        )

    def test_normalize_l1(self):
        """Test F.normalize with L1 norm."""
        self._run_trace_and_compile(
            F.normalize, (4, 8), "test_normalize_l1", 
            p=1.0, dim=1
        )

    def test_layer_norm(self):
        """Test F.layer_norm."""
        normalized_shape = [8]
        weight = torch.ones(8)
        bias = torch.zeros(8)
        func = partial(F.layer_norm, normalized_shape=normalized_shape, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (4, 8), "test_layer_norm")

    def test_layer_norm_2d(self):
        """Test F.layer_norm on 2D normalized shape (e.g., for images)."""
        normalized_shape = [8, 8]
        weight = torch.ones(8, 8)
        bias = torch.zeros(8, 8)
        func = partial(F.layer_norm, normalized_shape=normalized_shape, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (4, 8, 8), "test_layer_norm_2d")

    # =========================================================================
    # Dropout (eval mode - should be identity)
    # =========================================================================

    def test_dropout_eval(self):
        """Test F.dropout in eval mode (training=False) - should be identity."""
        self._run_trace_and_compile(
            F.dropout, (4, 8), "test_dropout_eval", 
            p=0.5, training=False
        )

    # =========================================================================
    # Convolutions
    # =========================================================================

    def test_conv1d(self):
        """Test F.conv1d."""
        weight = torch.randn(16, 3, 3)  # (out_channels, in_channels, kernel)
        bias = torch.randn(16)
        func = partial(F.conv1d, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 3, 32), "test_conv1d")

    def test_conv1d_with_padding(self):
        """Test F.conv1d with padding."""
        weight = torch.randn(16, 3, 3)
        bias = torch.randn(16)
        func = partial(F.conv1d, weight=weight, bias=bias, padding=1)
        self._run_trace_and_compile(func, (2, 3, 32), "test_conv1d_pad")

    def test_conv2d(self):
        """Test F.conv2d."""
        weight = torch.randn(16, 3, 3, 3)  # (out_ch, in_ch, kH, kW)
        bias = torch.randn(16)
        func = partial(F.conv2d, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_conv2d")

    def test_conv2d_with_stride_padding(self):
        """Test F.conv2d with stride and padding."""
        weight = torch.randn(16, 3, 3, 3)
        bias = torch.randn(16)
        func = partial(F.conv2d, weight=weight, bias=bias, stride=2, padding=1)
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_conv2d_stride")

    # =========================================================================
    # Pooling
    # =========================================================================

    def test_max_pool1d(self):
        """Test F.max_pool1d."""
        func = partial(F.max_pool1d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 32), "test_max_pool1d")

    def test_max_pool2d(self):
        """Test F.max_pool2d."""
        func = partial(F.max_pool2d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_max_pool2d")

    def test_max_pool2d_with_stride(self):
        """Test F.max_pool2d with custom stride."""
        func = partial(F.max_pool2d, kernel_size=3, stride=2, padding=1)
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_max_pool2d_stride")

    def test_avg_pool2d(self):
        """Test F.avg_pool2d."""
        func = partial(F.avg_pool2d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_avg_pool2d")

    def test_adaptive_avg_pool2d(self):
        """Test F.adaptive_avg_pool2d (common for variable input sizes)."""
        func = partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_adaptive_avg_pool2d")

    # =========================================================================
    # Interpolate / Resize
    # =========================================================================

    def test_interpolate_nearest(self):
        """Test F.interpolate with nearest neighbor mode."""
        func = partial(F.interpolate, size=(32, 32), mode='nearest')
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_interpolate_nearest")

    def test_interpolate_bilinear(self):
        """Test F.interpolate with bilinear mode."""
        func = partial(F.interpolate, size=(32, 32), mode='bilinear', align_corners=False)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_interpolate_bilinear")


# =============================================================================
# Recommended functions to test (in order of priority)
# =============================================================================
#
# HIGH PRIORITY - Common in RL/robotics:
# - F.relu, F.leaky_relu, F.gelu, F.tanh, F.sigmoid (activations)
# - F.softmax, F.log_softmax (probability distributions)
# - F.normalize (vector normalization)
# - F.batch_norm, F.layer_norm (normalization layers in eval mode)
# - F.dropout (in eval mode - should be identity)
# - F.conv1d, F.conv2d (convolutions)
# - F.max_pool1d, F.max_pool2d, F.avg_pool2d (pooling)
# - F.interpolate (resizing/upsampling)
#
# MEDIUM PRIORITY - Used in attention/transformers:
# - F.scaled_dot_product_attention (if available, PyTorch 2.0+)
# - F.multi_head_attention_forward
# - F.embedding
#
# LOWER PRIORITY - Less common but should work:
# - F.pad
# - F.unfold, F.fold
# - F.grid_sample
# - F.affine_grid
#
# TORCH OPERATIONS (not F.*) - NOT yet in test_traced_tensor.py:
# - torch.bmm (batched matrix multiplication)
# - torch.einsum (Einstein summation)
# - torch.where (conditional selection)
# =============================================================================


if __name__ == '__main__':
    unittest.main()

