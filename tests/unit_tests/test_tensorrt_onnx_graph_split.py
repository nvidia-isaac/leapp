#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for TensorRT ONNX graph split helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

from leapp.backends.tensorrt_onnx_graph_split import (
    extract_onnx_subgraph,
    is_op_tensorrt_supported,
    partition_nodes_by_tensorrt_support,
    replace_unsupported_nodes_with_identity,
    split_onnx_by_tensorrt_support,
)


def _make_mixed_model() -> onnx.ModelProto:
    """Build Abs -> Compress -> Relu (supported, unsupported, supported)."""
    nodes = [
        helper.make_node("Abs", ["x"], ["a"], name="abs0"),
        helper.make_node("Compress", ["a", "cond"], ["c"], name="compress0"),
        helper.make_node("Relu", ["c"], ["y"], name="relu0"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="mixed",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        value_info=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, None),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestTensorRTOnnxGraphSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import isaac_deploy_trt.tensorrt_onnx_ops  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

    def test_is_op_tensorrt_supported(self):
        self.assertTrue(is_op_tensorrt_supported("Abs"))
        self.assertFalse(is_op_tensorrt_supported("Compress"))
        self.assertFalse(is_op_tensorrt_supported("TotallyFakeOp"))
        self.assertTrue(
            is_op_tensorrt_supported("TotallyFakeOp", unknown_as_supported=True)
        )

    def test_partition_nodes_by_tensorrt_support(self):
        model = _make_mixed_model()
        segments = partition_nodes_by_tensorrt_support(model.graph.node)
        self.assertEqual(len(segments), 3)
        self.assertEqual((segments[0].start, segments[0].end, segments[0].supported), (0, 1, True))
        self.assertEqual((segments[1].start, segments[1].end, segments[1].supported), (1, 2, False))
        self.assertEqual((segments[2].start, segments[2].end, segments[2].supported), (2, 3, True))
        self.assertEqual(segments[1].ops, ("Compress",))

    def test_extract_onnx_subgraph(self):
        model = _make_mixed_model()
        mid = extract_onnx_subgraph(model, 1, 2, graph_name="unsupported_mid")
        self.assertEqual(len(mid.graph.node), 1)
        self.assertEqual(mid.graph.node[0].op_type, "Compress")
        self.assertEqual([i.name for i in mid.graph.input], ["a", "cond"])
        self.assertEqual([o.name for o in mid.graph.output], ["c"])

        first = extract_onnx_subgraph(model, 0, 1)
        self.assertEqual([i.name for i in first.graph.input], ["x"])
        self.assertEqual([o.name for o in first.graph.output], ["a"])

    def test_split_onnx_by_tensorrt_support_writes_files(self):
        model = _make_mixed_model()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "mixed.onnx")
            out_dir = os.path.join(tmp, "split")
            onnx.save(model, onnx_path)

            result = split_onnx_by_tensorrt_support(onnx_path, output_dir=out_dir)
            self.assertEqual(len(result.segments), 3)
            self.assertEqual(len(result.supported_paths), 2)
            self.assertEqual(len(result.unsupported_paths), 1)
            self.assertTrue(all(os.path.isfile(p) for p in result.supported_paths))
            self.assertTrue(os.path.isfile(result.unsupported_paths[0]))

            unsupported = onnx.load(result.unsupported_paths[0])
            self.assertEqual([n.op_type for n in unsupported.graph.node], ["Compress"])

    def test_replace_unsupported_nodes_with_identity(self):
        # Abs -> SoftmaxCrossEntropyLoss is awkward; use BitShift-like 1:1 via custom
        # unsupported catalog entry simulation with Compress replaced only when 1:1.
        # Compress is not 1:1, so leave a Relu->Unique->Relu style with Unique unsupported
        # but Unique can be multi-output — use AffineGrid? Also multi-in.
        # Det is unsupported and is 1-in / 1-out.
        nodes = [
            helper.make_node("Abs", ["x"], ["a"], name="abs0"),
            helper.make_node("Det", ["a"], ["d"], name="det0"),
            helper.make_node("Relu", ["d"], ["y"], name="relu0"),
        ]
        graph = helper.make_graph(
            nodes=nodes,
            name="det_mid",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        replaced = replace_unsupported_nodes_with_identity(model)
        self.assertEqual([n.op_type for n in replaced.graph.node], ["Abs", "Identity", "Relu"])
        # Original model unchanged.
        self.assertEqual(model.graph.node[1].op_type, "Det")


if __name__ == "__main__":
    unittest.main()
