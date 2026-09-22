#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for TensorRT plugin replacement helpers."""

from __future__ import annotations

import os
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

try:
    from isaac_deploy_trt import surgery as ogs
except ImportError:
    ogs = None  # type: ignore[assignment]

from leapp.backends.tensorrt_plugin_replace import (
    TrtPluginSpec,
    replace_layer_with_trt_plugin,
    replace_layers_by_op_with_trt_plugin,
    replace_onnx_layer_with_trt_plugin,
    replace_subgraph_with_trt_plugin,
)


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


def _dual_relu_model() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Relu", ["x"], ["a"], name="relu0"),
        helper.make_node("Relu", ["a"], ["y"], name="relu1"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name="dual_relu",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestTrtPluginReplace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            if ogs is None:
                raise unittest.SkipTest("isaac_deploy_trt is not installed")
            ogs._gs()  # type: ignore[union-attr]
        except ImportError:
            raise unittest.SkipTest("onnx-graphsurgeon not installed")

    def test_plugin_spec_attrs(self):
        spec = TrtPluginSpec(
            name="BatchedNMS_TRT",
            attrs={"numClasses": 80},
            version="1",
            namespace="",
        )
        attrs = spec.onnx_attrs()
        self.assertEqual(attrs["numClasses"], 80)
        self.assertEqual(attrs["plugin_version"], "1")
        self.assertEqual(attrs["plugin_namespace"], "")

    def test_replace_layer_with_trt_plugin(self):
        graph = ogs.load_graph(_chain_model())
        spec = TrtPluginSpec(name="AbsPlugin_TRT", attrs={"axis": 0}, version="1")
        replace_layer_with_trt_plugin(graph, "abs0", spec)
        node = ogs.get_node(graph, "abs0")
        self.assertEqual(node.op, "AbsPlugin_TRT")
        self.assertEqual(node.attrs["axis"], 0)
        self.assertEqual(node.attrs["plugin_version"], "1")

    def test_replace_layers_by_op(self):
        graph = ogs.load_graph(_dual_relu_model())
        replaced = replace_layers_by_op_with_trt_plugin(
            graph, "Relu", "ReluPlugin_TRT"
        )
        self.assertEqual(len(replaced), 2)
        model = ogs.export_model(graph)
        self.assertEqual(
            [n.op_type for n in model.graph.node],
            ["ReluPlugin_TRT", "ReluPlugin_TRT"],
        )

    def test_replace_subgraph_with_trt_plugin(self):
        graph = ogs.load_graph(_chain_model())
        replace_subgraph_with_trt_plugin(
            graph,
            inputs=["x"],
            outputs=["y"],
            plugin=TrtPluginSpec(
                name="FusedAbsRelu_TRT",
                node_name="fused0",
                attrs={"fused": 1},
            ),
        )
        model = ogs.export_model(graph)
        self.assertEqual(len(model.graph.node), 1)
        self.assertEqual(model.graph.node[0].op_type, "FusedAbsRelu_TRT")
        self.assertEqual(model.graph.node[0].name, "fused0")

    def test_replace_onnx_layer_file_api(self):
        model = _chain_model()
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.onnx")
            dst = os.path.join(tmp, "out.onnx")
            onnx.save(model, src)
            out = replace_onnx_layer_with_trt_plugin(
                src,
                plugin=TrtPluginSpec(name="ReluPlugin_TRT", version="2"),
                layer_name="relu0",
                output_path=dst,
            )
            self.assertTrue(os.path.isfile(dst))
            relu = next(n for n in out.graph.node if n.name == "relu0")
            self.assertEqual(relu.op_type, "ReluPlugin_TRT")
            attr_map = {a.name: a for a in relu.attribute}
            self.assertIn("plugin_version", attr_map)

    def test_replace_onnx_layer_requires_single_mode(self):
        with self.assertRaises(ValueError):
            replace_onnx_layer_with_trt_plugin(
                _chain_model(),
                plugin="X",
                layer_name="abs0",
                onnx_op="Relu",
            )


if __name__ == "__main__":
    unittest.main()
