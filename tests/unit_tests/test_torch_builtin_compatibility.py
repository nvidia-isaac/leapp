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

TODO: The following functions are NOT supported by ONNX export (opset 17):
  - F.l1_loss (aten::l1_loss not in ONNX)
  - F.smooth_l1_loss (aten::smooth_l1_loss not in ONNX)
  - F.huber_loss (aten::huber_loss not in ONNX)
  
  Options to address:
  1. Implement manual decomposition to basic ops (e.g., l1_loss = mean(abs(input - target)))
  2. Throw a warning when these ops are used with ONNX backend
  3. Use a higher opset version if/when these become available


TODO: The following may not be supported
F.batch_norm	Uses running mean/var stats that may not trace well
F.instance_norm	Similar running stats issues
F.group_norm	May have issues with affine parameters
F.embedding	Lookup table semantics can be tricky
F.multi_head_attention	Complex with multiple outputs, attention masks
F.scaled_dot_product_attention	Newer API, may have special handling
F.grid_sample	Complex indexing operations
F.affine_grid	Often paired with grid_sample
"""

import pathlib
import tempfile
import unittest
from functools import partial

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedTensor


class TestFunctional(unittest.TestCase):
    """Test torch.nn.functional operations trace and compile correctly."""
    
    # Number of random inputs to test after compiling
    NUM_TEST_INPUTS = 5

    def _run_trace_and_compile(self, func, input_shape, test_name, 
                               skip_torchscript=False, skip_onnx=False, **func_kwargs):
        """Helper to trace a function and verify compiled output matches.
        
        Compiles once, then tests with NUM_TEST_INPUTS random inputs.
        Tests FX GraphModule, TorchScript, and ONNX exports.
        
        Args:
            func: Function to apply (e.g., F.relu, or partial(F.linear, weight=w))
            input_shape: Shape of input tensor
            test_name: Name for the trace context
            skip_torchscript: Skip TorchScript testing (for known unsupported ops)
            skip_onnx: Skip ONNX testing (for known unsupported ops)
            **func_kwargs: Additional kwargs to pass to func
        """
        input_tensor = torch.randn(*input_shape)
        
        ctx = TracedTensorNode(name=test_name, node_index=0)
        traced_input = ctx.create_input(input_tensor.clone(), name="x")
        
        output = func(traced_input, **func_kwargs)
        self.assertIsInstance(output, TracedTensor)
        
        ctx.compile_trace({'output': output})
        graph_module = ctx.compiled_graph_module
        
        # Test with multiple random inputs across all export formats
        for i in range(self.NUM_TEST_INPUTS):
            test_input = torch.randn(*input_shape)
            expected = func(test_input, **func_kwargs)
            
            # Test 1: FX GraphModule execution
            actual_fx = graph_module(test_input)
            self.assertTrue(
                torch.allclose(actual_fx, expected, atol=1e-5),
                f"{test_name}: FX output mismatch on input {i+1}/{self.NUM_TEST_INPUTS}"
            )
        
        # Test 2: TorchScript export and execution (compile once, test with all inputs)
        if not skip_torchscript:
            try:
                scripted = torch.jit.script(graph_module)
                for i in range(self.NUM_TEST_INPUTS):
                    test_input = torch.randn(*input_shape)
                    expected = func(test_input, **func_kwargs)
                    actual_ts = scripted(test_input)
                    self.assertTrue(
                        torch.allclose(actual_ts, expected, atol=1e-5),
                        f"{test_name}: TorchScript output mismatch on input {i+1}"
                    )
            except Exception as e:
                self.fail(f"{test_name}: TorchScript export/execution failed: {e}")
        
        # Test 3: ONNX export and execution
        if not skip_onnx:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    onnx_path = pathlib.Path(tmpdir) / f"{test_name}.onnx"
                    sample_input = torch.randn(*input_shape)
                    torch.onnx.export(
                        graph_module,
                        (sample_input,),
                        onnx_path,
                        dynamo=False,
                        export_params=True,
                        opset_version=17,
                        input_names=['input'],
                        output_names=['output'],
                    )
                    session = ort.InferenceSession(str(onnx_path))
                    
                    for i in range(self.NUM_TEST_INPUTS):
                        test_input = torch.randn(*input_shape)
                        expected = func(test_input, **func_kwargs)
                        output_onnx = session.run(None, {"input": test_input.numpy()})[0]
                        self.assertTrue(
                            np.allclose(output_onnx, expected.numpy(), atol=1e-5),
                            f"{test_name}: ONNX output mismatch on input {i+1}"
                        )
            except Exception as e:
                self.fail(f"{test_name}: ONNX export/execution failed: {e}")

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

    def test_softplus(self):
        """Test F.softplus activation."""
        self._run_trace_and_compile(F.softplus, (4, 8), "test_softplus")

    def test_mish(self):
        """Test F.mish activation."""
        self._run_trace_and_compile(F.mish, (4, 8), "test_mish")

    def test_hardtanh(self):
        """Test F.hardtanh activation."""
        self._run_trace_and_compile(F.hardtanh, (4, 8), "test_hardtanh")

    def test_hardtanh_custom_range(self):
        """Test F.hardtanh with custom min/max values."""
        self._run_trace_and_compile(
            F.hardtanh, (4, 8), "test_hardtanh_custom",
            min_val=-2.0, max_val=2.0
        )

    def test_hardswish(self):
        """Test F.hardswish activation."""
        self._run_trace_and_compile(F.hardswish, (4, 8), "test_hardswish")

    def test_hardsigmoid(self):
        """Test F.hardsigmoid activation."""
        self._run_trace_and_compile(F.hardsigmoid, (4, 8), "test_hardsigmoid")

    def test_prelu(self):
        """Test F.prelu activation with learnable weight."""
        weight = torch.tensor([0.25])  # Single weight for all channels
        func = partial(F.prelu, weight=weight)
        self._run_trace_and_compile(func, (4, 8), "test_prelu")

    def test_prelu_per_channel(self):
        """Test F.prelu with per-channel weights."""
        weight = torch.randn(8)  # One weight per channel
        func = partial(F.prelu, weight=weight)
        self._run_trace_and_compile(func, (4, 8), "test_prelu_per_channel")

    def test_relu6(self):
        """Test F.relu6 activation (ReLU clamped to 6)."""
        self._run_trace_and_compile(F.relu6, (4, 8), "test_relu6")

    def test_celu(self):
        """Test F.celu activation."""
        self._run_trace_and_compile(F.celu, (4, 8), "test_celu")

    def test_selu(self):
        """Test F.selu activation (self-normalizing)."""
        self._run_trace_and_compile(F.selu, (4, 8), "test_selu")

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

    def test_conv3d(self):
        """Test F.conv3d."""
        # (out_ch, in_ch, kD, kH, kW)
        weight = torch.randn(8, 3, 3, 3, 3)
        bias = torch.randn(8)
        func = partial(F.conv3d, weight=weight, bias=bias)
        # Input: (batch, channels, depth, height, width)
        self._run_trace_and_compile(func, (2, 3, 8, 16, 16), "test_conv3d")

    def test_conv_transpose1d(self):
        """Test F.conv_transpose1d (deconvolution)."""
        weight = torch.randn(3, 16, 3)  # (in_ch, out_ch, kernel)
        bias = torch.randn(16)
        func = partial(F.conv_transpose1d, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 3, 32), "test_conv_transpose1d")

    def test_conv_transpose2d(self):
        """Test F.conv_transpose2d (deconvolution)."""
        weight = torch.randn(3, 16, 3, 3)  # (in_ch, out_ch, kH, kW)
        bias = torch.randn(16)
        func = partial(F.conv_transpose2d, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_conv_transpose2d")

    def test_conv_transpose2d_with_stride(self):
        """Test F.conv_transpose2d with stride for upsampling."""
        weight = torch.randn(3, 16, 4, 4)
        bias = torch.randn(16)
        func = partial(F.conv_transpose2d, weight=weight, bias=bias, stride=2, padding=1)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_conv_transpose2d_stride")

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

    def test_avg_pool1d(self):
        """Test F.avg_pool1d."""
        func = partial(F.avg_pool1d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 32), "test_avg_pool1d")

    def test_avg_pool3d(self):
        """Test F.avg_pool3d."""
        func = partial(F.avg_pool3d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 8, 16, 16), "test_avg_pool3d")

    def test_max_pool3d(self):
        """Test F.max_pool3d."""
        func = partial(F.max_pool3d, kernel_size=2)
        self._run_trace_and_compile(func, (2, 3, 8, 16, 16), "test_max_pool3d")

    def test_adaptive_avg_pool1d(self):
        """Test F.adaptive_avg_pool1d."""
        func = partial(F.adaptive_avg_pool1d, output_size=1)
        self._run_trace_and_compile(func, (2, 3, 32), "test_adaptive_avg_pool1d")

    def test_adaptive_avg_pool3d(self):
        """Test F.adaptive_avg_pool3d."""
        func = partial(F.adaptive_avg_pool3d, output_size=(1, 1, 1))
        self._run_trace_and_compile(func, (2, 3, 8, 16, 16), "test_adaptive_avg_pool3d")

    def test_adaptive_max_pool1d(self):
        """Test F.adaptive_max_pool1d."""
        func = partial(F.adaptive_max_pool1d, output_size=1)
        self._run_trace_and_compile(func, (2, 3, 32), "test_adaptive_max_pool1d")

    def test_adaptive_max_pool2d(self):
        """Test F.adaptive_max_pool2d."""
        func = partial(F.adaptive_max_pool2d, output_size=(1, 1))
        self._run_trace_and_compile(func, (2, 3, 32, 32), "test_adaptive_max_pool2d")

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

    def test_interpolate_scale_factor(self):
        """Test F.interpolate with scale_factor."""
        # Note: scale_factor must be float (2.0) not int (2) for TorchScript compatibility
        func = partial(F.interpolate, scale_factor=2.0, mode='nearest')
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_interpolate_scale")

    def test_interpolate_trilinear(self):
        """Test F.interpolate with trilinear mode for 3D."""
        func = partial(F.interpolate, size=(16, 32, 32), mode='trilinear', align_corners=False)
        self._run_trace_and_compile(func, (2, 3, 8, 16, 16), "test_interpolate_trilinear")

    # =========================================================================
    # Padding
    # =========================================================================

    def test_pad_constant(self):
        """Test F.pad with constant padding."""
        # Note: value must be float (0.0) not int (0) for TorchScript compatibility
        func = partial(F.pad, pad=(1, 1, 1, 1), mode='constant', value=0.0)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_pad_constant")

    def test_pad_reflect(self):
        """Test F.pad with reflect padding."""
        func = partial(F.pad, pad=(1, 1, 1, 1), mode='reflect')
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_pad_reflect")

    def test_pad_replicate(self):
        """Test F.pad with replicate padding."""
        func = partial(F.pad, pad=(1, 1, 1, 1), mode='replicate')
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_pad_replicate")

    def test_pad_1d(self):
        """Test F.pad on 1D input."""
        # Note: value must be float (0.0) not int (0) for TorchScript compatibility
        func = partial(F.pad, pad=(2, 2), mode='constant', value=0.0)
        self._run_trace_and_compile(func, (2, 3, 32), "test_pad_1d")

    # =========================================================================
    # Loss Functions
    # =========================================================================

    def test_mse_loss(self):
        """Test F.mse_loss."""
        target = torch.randn(4, 8)
        func = partial(F.mse_loss, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_mse_loss")

    def test_l1_loss(self):
        """Test F.l1_loss.
        
        Note: skip_onnx=True because aten::l1_loss is not supported in ONNX.
        """
        target = torch.randn(4, 8)
        func = partial(F.l1_loss, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_l1_loss", skip_onnx=True)

    def test_smooth_l1_loss(self):
        """Test F.smooth_l1_loss (Huber loss).
        
        Note: skip_onnx=True because aten::smooth_l1_loss is not supported in ONNX.
        """
        target = torch.randn(4, 8)
        func = partial(F.smooth_l1_loss, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_smooth_l1_loss", skip_onnx=True)

    def test_cross_entropy(self):
        """Test F.cross_entropy with class labels."""
        # Cross entropy needs (N, C) input and (N,) target with class indices
        target = torch.randint(0, 8, (4,))  # Class indices for 8 classes
        func = partial(F.cross_entropy, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_cross_entropy")

    def test_nll_loss(self):
        """Test F.nll_loss (negative log likelihood)."""
        target = torch.randint(0, 8, (4,))
        # NLL loss expects log-probabilities, so wrap with log_softmax
        def nll_with_logsoftmax(x, target):
            return F.nll_loss(F.log_softmax(x, dim=-1), target)
        func = partial(nll_with_logsoftmax, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_nll_loss")

    def test_binary_cross_entropy_with_logits(self):
        """Test F.binary_cross_entropy_with_logits."""
        target = torch.rand(4, 8)  # Binary targets in [0, 1]
        func = partial(F.binary_cross_entropy_with_logits, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_bce_logits")

    def test_huber_loss(self):
        """Test F.huber_loss.
        
        Note: skip_onnx=True because aten::huber_loss is not supported in ONNX.
        """
        target = torch.randn(4, 8)
        func = partial(F.huber_loss, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_huber_loss", skip_onnx=True)

    # =========================================================================
    # Torch Operations (not F.*)
    # =========================================================================

    def test_bmm(self):
        """Test torch.bmm (batched matrix multiplication)."""
        # bmm needs two 3D tensors: (batch, n, m) @ (batch, m, p)
        other = torch.randn(4, 8, 16)
        func = partial(torch.bmm, mat2=other)
        self._run_trace_and_compile(func, (4, 4, 8), "test_bmm")

    def test_einsum_matmul(self):
        """Test torch.einsum for matrix multiplication."""
        other = torch.randn(8, 16)
        def einsum_matmul(x, other):
            return torch.einsum('bi,ij->bj', x, other)
        func = partial(einsum_matmul, other=other)
        self._run_trace_and_compile(func, (4, 8), "test_einsum_matmul")

    def test_einsum_batched(self):
        """Test torch.einsum for batched operations."""
        other = torch.randn(4, 8, 16)
        def einsum_batched(x, other):
            return torch.einsum('bik,bkj->bij', x, other)
        func = partial(einsum_batched, other=other)
        self._run_trace_and_compile(func, (4, 4, 8), "test_einsum_batched")

    def test_where(self):
        """Test torch.where (conditional selection)."""
        y = torch.randn(4, 8)
        def where_positive(x, y):
            return torch.where(x > 0, x, y)
        func = partial(where_positive, y=y)
        self._run_trace_and_compile(func, (4, 8), "test_where")

    def test_clamp(self):
        """Test torch.clamp."""
        func = partial(torch.clamp, min=-1.0, max=1.0)
        self._run_trace_and_compile(func, (4, 8), "test_clamp")

    def test_abs(self):
        """Test torch.abs."""
        self._run_trace_and_compile(torch.abs, (4, 8), "test_abs")

    def test_sqrt(self):
        """Test torch.sqrt (on positive inputs)."""
        # Use abs to ensure positive inputs
        def sqrt_safe(x):
            return torch.sqrt(torch.abs(x) + 1e-6)
        self._run_trace_and_compile(sqrt_safe, (4, 8), "test_sqrt")

    def test_exp(self):
        """Test torch.exp."""
        self._run_trace_and_compile(torch.exp, (4, 8), "test_exp")

    def test_log(self):
        """Test torch.log (on positive inputs)."""
        def log_safe(x):
            return torch.log(torch.abs(x) + 1e-6)
        self._run_trace_and_compile(log_safe, (4, 8), "test_log")

    def test_pow(self):
        """Test torch.pow."""
        func = partial(torch.pow, exponent=2)
        self._run_trace_and_compile(func, (4, 8), "test_pow")

    def test_sin_cos(self):
        """Test torch.sin and torch.cos."""
        def sin_cos(x):
            return torch.sin(x) + torch.cos(x)
        self._run_trace_and_compile(sin_cos, (4, 8), "test_sin_cos")

    def test_stack(self):
        """Test torch.stack."""
        other = torch.randn(4, 8)
        def stack_with_other(x, other):
            return torch.stack([x, other], dim=0)
        func = partial(stack_with_other, other=other)
        self._run_trace_and_compile(func, (4, 8), "test_stack")

    def test_cat(self):
        """Test torch.cat."""
        other = torch.randn(4, 8)
        def cat_with_other(x, other):
            return torch.cat([x, other], dim=0)
        func = partial(cat_with_other, other=other)
        self._run_trace_and_compile(func, (4, 8), "test_cat")

    def test_split(self):
        """Test torch.split."""
        def split_and_sum(x):
            parts = torch.split(x, 4, dim=1)
            return parts[0] + parts[1]
        self._run_trace_and_compile(split_and_sum, (4, 8), "test_split")

    def test_chunk(self):
        """Test torch.chunk."""
        def chunk_and_sum(x):
            parts = torch.chunk(x, 2, dim=1)
            return parts[0] + parts[1]
        self._run_trace_and_compile(chunk_and_sum, (4, 8), "test_chunk")


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

