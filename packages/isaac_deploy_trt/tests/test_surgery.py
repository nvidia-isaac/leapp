#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""GraphSurgeon rewrites that prepare ONNX graphs for TensorRT."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from isaac_deploy_trt import surgery as ogs
from isaac_deploy_trt.prepare import needs_tensorrt_rewrite, rewrite_onnx_file
from isaac_deploy_trt.unsupported_surgery import rewrite_unsupported_onnx_ops


def _chain_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Abs", ["x"], ["a"], name="abs0"),
        helper.make_node("Relu", ["a"], ["y"], name="relu0"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="chain",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        value_info=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _model(nodes, inputs, outputs, initializers=None):
    graph = helper.make_graph(
        nodes=nodes,
        name="g",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers or [],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class _GraphSurgeonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ogs._gs()
        except ImportError:
            raise unittest.SkipTest("onnx-graphsurgeon not installed")


class TestSurgery(_GraphSurgeonTest):
    def test_rewrite_uint8_intermediate_casts(self):
        nodes = [
            helper.make_node("Cast", ["x"], ["u"], name="to_u8", to=2),
            helper.make_node("Relu", ["u"], ["u2"], name="relu_u"),
            helper.make_node("Cast", ["u2"], ["y"], name="to_f32", to=1),
        ]
        graph = helper.make_graph(
            nodes=nodes,
            name="u8",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            value_info=[
                helper.make_tensor_value_info("u", TensorProto.UINT8, [2]),
                helper.make_tensor_value_info("u2", TensorProto.UINT8, [2]),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        gs_graph = ogs.load_graph(model)
        rewritten = ogs.rewrite_uint8_intermediate_casts(gs_graph)
        self.assertEqual(rewritten, ["to_u8"])

    def test_disable_resize_antialias(self):
        resize = helper.make_node(
            "Resize",
            ["x", "", "", "scales"],
            ["y"],
            name="resize0",
            mode="linear",
            antialias=1,
        )
        scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])
        graph = helper.make_graph(
            nodes=[resize],
            name="rz",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])],
            initializer=[scales],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        gs_graph = ogs.load_graph(model)
        self.assertEqual(ogs.disable_resize_antialias(gs_graph), ["resize0"])

    def test_reduce_reshape_rank_for_tensorrt(self):
        target = helper.make_tensor(
            "shape9", TensorProto.INT64, [9], [1, 2, 3, 8, 2, 16, 11, 2, 16]
        )
        final = helper.make_tensor("shape2", TensorProto.INT64, [2], [352, 1536])
        nodes = [
            helper.make_node("Reshape", ["x", "shape9"], ["v"], name="reshape9"),
            helper.make_node(
                "Transpose", ["v"], ["p"], name="perm9", perm=[0, 3, 6, 4, 7, 2, 1, 5, 8]
            ),
            helper.make_node("Reshape", ["p", "shape2"], ["y"], name="reshape2"),
        ]
        graph = helper.make_graph(
            nodes=nodes,
            name="shuffle",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 256, 352])
            ],
            outputs=[
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [352, 1536])
            ],
            initializer=[target, final],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        gs_graph = ogs.load_graph(model)
        self.assertEqual(ogs.reduce_reshape_rank_for_tensorrt(gs_graph), ["reshape9"])

    def test_rewrite_onnx_file_preserves_io(self):
        resize = helper.make_node(
            "Resize",
            ["x", "", "", "scales"],
            ["y"],
            name="resize0",
            mode="linear",
            antialias=1,
        )
        scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])
        graph = helper.make_graph(
            nodes=[resize],
            name="rz",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])],
            initializer=[scales],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.onnx")
            dst = os.path.join(tmp, "out.onnx")
            onnx.save(model, src)
            result = rewrite_onnx_file(src, dst)
            self.assertTrue(result.rewritten)
            out = onnx.load(dst)
            self.assertEqual([i.name for i in out.graph.input], ["x"])
            self.assertEqual([o.name for o in out.graph.output], ["y"])


class UnsupportedSurgeryTests(_GraphSurgeonTest):
    def test_bitwise_not_int32_becomes_sub(self):
        model = _model(
            [helper.make_node("BitwiseNot", ["x"], ["y"], name="bn")],
            [helper.make_tensor_value_info("x", TensorProto.INT32, [2])],
            [helper.make_tensor_value_info("y", TensorProto.INT32, [2])],
        )
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), ["bn"])
        skipped = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(skipped, ops=frozenset()), [])
        self.assertEqual(skipped.nodes[0].op, "BitwiseNot")
        ops = [node.op for node in graph.nodes]
        self.assertNotIn("BitwiseNot", ops)
        sub = next(node for node in graph.nodes if node.op == "Sub")
        self.assertEqual(int(sub.inputs[0].values), -1)
        self.assertEqual(sub.inputs[1].name, "x")
        self.assertEqual(sub.outputs[0].name, "y")

    def test_bool_bitwise_and_becomes_and(self):
        model = _model(
            [helper.make_node("BitwiseAnd", ["a", "b"], ["y"], name="band")],
            [
                helper.make_tensor_value_info("a", TensorProto.BOOL, [2]),
                helper.make_tensor_value_info("b", TensorProto.BOOL, [2]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.BOOL, [2])],
        )
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), ["band"])
        self.assertEqual([node.op for node in graph.nodes], ["And"])

    def test_integer_bitwise_and_stays(self):
        model = _model(
            [helper.make_node("BitwiseAnd", ["a", "b"], ["y"], name="band")],
            [
                helper.make_tensor_value_info("a", TensorProto.INT32, [2]),
                helper.make_tensor_value_info("b", TensorProto.INT32, [2]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.INT32, [2])],
        )
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), [])
        self.assertEqual(graph.nodes[0].op, "BitwiseAnd")

    def test_qlinear_conv_becomes_conv(self):
        weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), np.int8), "w")
        model = _model(
            [
                helper.make_node(
                    "QLinearConv",
                    ["x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp"],
                    ["y"],
                    name="qlc",
                )
            ],
            [helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 1, 1])],
            [helper.make_tensor_value_info("y", TensorProto.INT8, [1, 1, 1, 1])],
            [
                numpy_helper.from_array(np.array(0.5, np.float32), "x_scale"),
                numpy_helper.from_array(np.array(0, np.int8), "x_zp"),
                weight,
                numpy_helper.from_array(np.array(0.25, np.float32), "w_scale"),
                numpy_helper.from_array(np.array(0, np.int8), "w_zp"),
                numpy_helper.from_array(np.array(0.1, np.float32), "y_scale"),
                numpy_helper.from_array(np.array(0, np.int8), "y_zp"),
            ],
        )
        self.assertTrue(needs_tensorrt_rewrite(model))
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), ["qlc"])
        ops = {node.op for node in graph.nodes}
        self.assertIn("Conv", ops)
        self.assertNotIn("QLinearConv", ops)
        self.assertNotIn("QuantizeLinear", ops)
        conv = next(node for node in graph.nodes if node.op == "Conv")
        self.assertEqual(conv.outputs[0].name, "qlc_conv")
        self.assertEqual(graph.outputs[0].name, "y")

    def test_center_crop_pad_becomes_slice(self):
        target = numpy_helper.from_array(np.array([2, 2], np.int64), "shape")
        model = _model(
            [helper.make_node("CenterCropPad", ["x", "shape"], ["y"], name="crop", axes=[2, 3])],
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
            [target],
        )
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), ["crop"])
        slice_node = next(node for node in graph.nodes if node.op == "Slice")
        starts = [int(v) for v in np.asarray(slice_node.inputs[1].values).reshape(-1)]
        ends = [int(v) for v in np.asarray(slice_node.inputs[2].values).reshape(-1)]
        self.assertEqual(starts, [0, 0, 1, 1])
        self.assertEqual(ends, [1, 1, 3, 3])
        self.assertEqual(slice_node.outputs[0].name, "y")

    def test_dynamic_quantize_linear_expands(self):
        model = _model(
            [helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zp"], name="dq")],
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
            [
                helper.make_tensor_value_info("y", TensorProto.UINT8, [4]),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("zp", TensorProto.UINT8, []),
            ],
        )
        graph = ogs.load_graph(model)
        self.assertEqual(rewrite_unsupported_onnx_ops(graph), ["dq"])
        ops = {node.op for node in graph.nodes}
        self.assertNotIn("DynamicQuantizeLinear", ops)
        self.assertIn("ReduceMin", ops)
        self.assertIn("ReduceMax", ops)
        self.assertEqual(graph.outputs[1].name, "scale")


if __name__ == "__main__":
    unittest.main()
