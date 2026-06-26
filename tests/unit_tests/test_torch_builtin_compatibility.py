#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pathlib
import tempfile
import unittest
from functools import partial

import pytest

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import TracedTensor

from tests.unit_tests.export_format_validation import verify_exported_program_on_random_inputs



@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
class TestFunctional(unittest.TestCase):
    """Test torch.nn.functional operations trace and compile correctly."""
    
    # Number of random inputs to test after compiling
    NUM_TEST_INPUTS = 5

    def _run_trace_and_compile(self, func, input_shape, test_name,
                               skip_torchscript=False, skip_onnx=False,
                               skip_exported_program=False,
                               onnx_opset=17, atol=1e-5, **func_kwargs):
        """Helper to trace a function and verify compiled output matches.
        
        Compiles once, then tests with NUM_TEST_INPUTS random inputs.
        Tests FX GraphModule, TorchScript, and ONNX exports.
        
        Args:
            func: Function to apply (e.g., F.relu, or partial(F.linear, weight=w))
            input_shape: Shape of input tensor
            test_name: Name for the trace context
            skip_torchscript: Skip TorchScript testing (for known unsupported ops)
            skip_onnx: Skip ONNX testing (for known unsupported ops)
            skip_exported_program: Skip torch.export testing (for known unsupported ops)
            **func_kwargs: Additional kwargs to pass to func
        """
        input_tensor = torch.randn(*input_shape)
        
        ctx = TracedTensorNode(name=test_name, node_index=0)
        traced_input = ctx.create_input(input_tensor.clone(), name="x")
        
        output = func(traced_input, **func_kwargs)
        self.assertIsInstance(output, TracedTensor)
        
        ctx.compile_trace({'output': output})
        graph_module = ctx.m
        
        # Test with multiple random inputs across all export formats
        for i in range(self.NUM_TEST_INPUTS):
            test_input = torch.randn(*input_shape)
            expected = func(test_input, **func_kwargs)
            
            # Test 1: FX GraphModule execution
            actual_fx = graph_module(test_input)
            self.assertTrue(
                torch.allclose(actual_fx, expected, atol=atol),
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
                        torch.allclose(actual_ts, expected, atol=atol),
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
                        opset_version=onnx_opset,
                        input_names=['input'],
                        output_names=['output'],
                    )
                    session = ort.InferenceSession(str(onnx_path))
                    
                    for i in range(self.NUM_TEST_INPUTS):
                        test_input = torch.randn(*input_shape)
                        expected = func(test_input, **func_kwargs)
                        output_onnx = session.run(None, {"input": test_input.numpy()})[0]
                        self.assertTrue(
                            np.allclose(output_onnx, expected.numpy(), atol=atol),
                            f"{test_name}: ONNX output mismatch on input {i+1}"
                        )
            except Exception as e:
                self.fail(f"{test_name}: ONNX export/execution failed: {e}")

        # Test 4: ExportedProgram export and execution
        if not skip_exported_program:
            verify_exported_program_on_random_inputs(
                self,
                graph_module,
                input_shape,
                func,
                test_name=test_name,
                num_inputs=self.NUM_TEST_INPUTS,
                atol=atol,
                func_kwargs=func_kwargs,
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

    def test_batch_norm(self):
        """Test F.batch_norm in eval mode (training=False).
        
        Uses fixed running_mean/running_var which are required when training=False.
        Running stats become constants captured in the trace.
        """
        num_features = 3
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)
        weight = torch.ones(num_features)
        bias = torch.zeros(num_features)
        func = partial(
            F.batch_norm, running_mean=running_mean, running_var=running_var,
            weight=weight, bias=bias, training=False
        )
        self._run_trace_and_compile(func, (2, 3, 8, 8), "test_batch_norm")

    def test_batch_norm_1d(self):
        """Test F.batch_norm on 1D input (e.g., after linear layer)."""
        num_features = 8
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)
        weight = torch.ones(num_features)
        bias = torch.zeros(num_features)
        func = partial(
            F.batch_norm, running_mean=running_mean, running_var=running_var,
            weight=weight, bias=bias, training=False
        )
        self._run_trace_and_compile(func, (4, 8), "test_batch_norm_1d")

    def test_instance_norm(self):
        """Test F.instance_norm.
        
        Uses default use_input_stats=True (computes stats from input).
        """
        num_features = 3
        weight = torch.ones(num_features)
        bias = torch.zeros(num_features)
        func = partial(F.instance_norm, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 3, 8, 8), "test_instance_norm")

    def test_instance_norm_with_running_stats(self):
        """Test F.instance_norm with running stats (use_input_stats=False)."""
        num_features = 3
        running_mean = torch.zeros(num_features)
        running_var = torch.ones(num_features)
        weight = torch.ones(num_features)
        bias = torch.zeros(num_features)
        func = partial(
            F.instance_norm, running_mean=running_mean, running_var=running_var,
            weight=weight, bias=bias, use_input_stats=False
        )
        self._run_trace_and_compile(func, (2, 3, 8, 8), "test_instance_norm_running")

    def test_group_norm(self):
        """Test F.group_norm with affine parameters."""
        num_channels = 8
        num_groups = 4
        weight = torch.ones(num_channels)
        bias = torch.zeros(num_channels)
        func = partial(F.group_norm, num_groups=num_groups, weight=weight, bias=bias)
        self._run_trace_and_compile(func, (2, 8, 16, 16), "test_group_norm")

    def test_group_norm_no_affine(self):
        """Test F.group_norm without affine parameters."""
        func = partial(F.group_norm, num_groups=4)
        self._run_trace_and_compile(func, (2, 8, 16, 16), "test_group_norm_no_affine")

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
    # Unfold / Fold
    # =========================================================================

    def test_unfold(self):
        """Test F.unfold (im2col) - extracts sliding local blocks from input."""
        # Input: (N, C, H, W), kernel_size=3 extracts 3x3 patches
        func = partial(F.unfold, kernel_size=3)
        self._run_trace_and_compile(func, (2, 3, 8, 8), "test_unfold")

    def test_unfold_with_params(self):
        """Test F.unfold with stride and padding."""
        func = partial(F.unfold, kernel_size=3, padding=1, stride=2)
        self._run_trace_and_compile(func, (2, 3, 8, 8), "test_unfold_params")

    def test_fold(self):
        """Test F.fold (col2im) - combines sliding local blocks into a tensor.
        
        Fold is the inverse of unfold. Input shape must match the output of
        an equivalent unfold operation.
        Note: aten::col2im requires ONNX opset 18+.
        """
        # Unfold (N=2, C=3, H=8, W=8) with kernel_size=3 produces
        # shape (2, 3*3*3, L) where L = (8-3+1)*(8-3+1) = 36
        func = partial(F.fold, output_size=(8, 8), kernel_size=3)
        self._run_trace_and_compile(func, (2, 27, 36), "test_fold", onnx_opset=18)

    def test_unfold_fold_roundtrip(self):
        """Test unfold → fold roundtrip (with divisor correction).
        
        Unfold then fold doesn't produce identity without normalizing by the
        overlap count. This test verifies the ops compose correctly.
        Note: aten::col2im (fold) requires ONNX opset 18+.
        """
        kernel_size = 3
        def unfold_fold(x):
            # x: (N, C, H, W)
            unfolded = F.unfold(x, kernel_size=kernel_size)
            folded = F.fold(unfolded, output_size=(8, 8), kernel_size=kernel_size)
            return folded
        self._run_trace_and_compile(unfold_fold, (2, 3, 8, 8), "test_unfold_fold", onnx_opset=18)

    # =========================================================================
    # Loss Functions
    # =========================================================================

    def test_mse_loss(self):
        """Test F.mse_loss."""
        target = torch.randn(4, 8)
        func = partial(F.mse_loss, target=target)
        self._run_trace_and_compile(func, (4, 8), "test_mse_loss")

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
    # =========================================================================
    # Embedding
    # =========================================================================

    def test_embedding(self):
        """Test F.embedding (lookup table).
        
        Note: Uses a custom test flow because embedding requires integer (LongTensor)
        inputs, not the float tensors generated by _run_trace_and_compile.
        """
        vocab_size = 100
        embed_dim = 16
        weight = torch.randn(vocab_size, embed_dim)

        input_indices = torch.randint(0, vocab_size, (4, 8))
        
        ctx = TracedTensorNode(name="test_embedding", node_index=0)
        traced_input = ctx.create_input(input_indices.clone(), name="x")
        
        output = F.embedding(traced_input, weight)
        self.assertIsInstance(output, TracedTensor)
        
        ctx.compile_trace({'output': output})
        graph_module = ctx.m
        
        for i in range(self.NUM_TEST_INPUTS):
            test_input = torch.randint(0, vocab_size, (4, 8))
            expected = F.embedding(test_input, weight)
            actual_fx = graph_module(test_input)
            self.assertTrue(
                torch.allclose(actual_fx, expected, atol=1e-5),
                f"test_embedding: FX output mismatch on input {i+1}/{self.NUM_TEST_INPUTS}"
            )

    # =========================================================================
    # Attention
    # =========================================================================

    def test_scaled_dot_product_attention(self):
        """Test F.scaled_dot_product_attention (PyTorch 2.0+).
        
        Only query is traced; key and value are fixed via partial.
        Note: skip_onnx=True because SDPA may not be supported in ONNX opset 17.
        """
        # (batch, num_heads, seq_len, head_dim)
        key = torch.randn(2, 4, 8, 16)
        value = torch.randn(2, 4, 8, 16)
        func = partial(F.scaled_dot_product_attention, key=key, value=value)
        self._run_trace_and_compile(
            func, (2, 4, 8, 16), "test_sdpa",
            skip_onnx=True
        )

    def test_multi_head_attention(self):
        """Test F.multi_head_attention_forward (self-attention).
        
        Wraps the forward function to return only the output tensor (not attention weights).
        Note: skip_onnx=True because MHA decomposition may not be supported in ONNX opset 17.
        """
        embed_dim = 32
        num_heads = 4
        
        in_proj_weight = torch.randn(3 * embed_dim, embed_dim)
        in_proj_bias = torch.randn(3 * embed_dim)
        out_proj_weight = torch.randn(embed_dim, embed_dim)
        out_proj_bias = torch.randn(embed_dim)
        
        def mha_self_attention(query):
            # Input shape: (seq_len, batch_size, embed_dim) — seq_len first
            output, _ = F.multi_head_attention_forward(
                query, query, query,  # self-attention: Q=K=V
                embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                None, None,  # bias_k, bias_v
                False,  # add_zero_attn
                0.0,  # dropout_p
                out_proj_weight, out_proj_bias,
                training=False,
                need_weights=False,
            )
            return output
        
        # Shape: (seq_len, batch_size, embed_dim)
        self._run_trace_and_compile(
            mha_self_attention, (8, 2, embed_dim), "test_mha",
            skip_onnx=True
        )

    # =========================================================================
    # Spatial Transforms
    # =========================================================================

    def test_grid_sample(self):
        """Test F.grid_sample.
        
        Grid values are in [-1, 1] range for normalized coordinates.
        """
        # grid shape: (N, H_out, W_out, 2)
        grid = torch.rand(2, 8, 8, 2) * 2 - 1  # Random grid in [-1, 1]
        func = partial(F.grid_sample, grid=grid, align_corners=True)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_grid_sample")

    def test_grid_sample_bilinear(self):
        """Test F.grid_sample with bilinear interpolation (default)."""
        grid = torch.rand(2, 8, 8, 2) * 2 - 1
        func = partial(F.grid_sample, grid=grid, mode='bilinear', 
                       padding_mode='zeros', align_corners=False)
        self._run_trace_and_compile(func, (2, 3, 16, 16), "test_grid_sample_bilinear")

    def test_affine_grid(self):
        """Test F.affine_grid.
        
        Generates a sampling grid from an affine transformation matrix (theta).
        Input theta shape: (N, 2, 3) for 2D spatial transforms.
        Note: aten::affine_grid_generator requires ONNX opset 20+.
        """
        func = partial(F.affine_grid, size=(2, 3, 16, 16), align_corners=True)
        self._run_trace_and_compile(func, (2, 2, 3), "test_affine_grid", onnx_opset=20)

    def test_affine_grid_and_grid_sample(self):
        """Test F.affine_grid → F.grid_sample pipeline (common in STN).
        
        Note: aten::affine_grid_generator requires ONNX opset 20+.
        """
        source = torch.randn(2, 3, 16, 16)
        
        def stn_transform(theta):
            grid = F.affine_grid(theta, source.size(), align_corners=True)
            return F.grid_sample(source, grid, align_corners=True)
        
        self._run_trace_and_compile(stn_transform, (2, 2, 3), "test_affine_grid_sample", onnx_opset=20, atol=1e-4)

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

