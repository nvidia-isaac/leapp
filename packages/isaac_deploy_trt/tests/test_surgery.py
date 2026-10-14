#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from isaac_deploy_trt import surgery as ogs
from isaac_deploy_trt.prepare import rewrite_onnx_file


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


class TestSurgery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ogs._gs()
        except ImportError:
            raise unittest.SkipTest("onnx-graphsurgeon not installed")

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


if __name__ == "__main__":
    unittest.main()
