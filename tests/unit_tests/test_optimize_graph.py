#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for ``leapp.optimize_graph``."""

from __future__ import annotations

import os
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper
import yaml

from leapp.export_manager import ExportManager
from leapp.leapp import _MANAGER, optimize_graph


def _write_resize_onnx(path: str) -> None:
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
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )


class TestOptimizeGraph(unittest.TestCase):
    def test_requires_compiled_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("missing_graph")
            ExportManager.set_interpret_graph(False)
            with self.assertRaises(Exception):
                optimize_graph(trt_compatible=True)

    def test_no_options_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("opt_graph")
            ExportManager.set_interpret_graph(False)
            yaml_path = os.path.join(tmp, "opt_graph.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"models": {}}, handle)
            result = optimize_graph()
            self.assertEqual(result, {})

    def test_trt_compatible_rewrites_onnx_in_graph_dir(self):
        try:
            import isaac_deploy_trt.surgery  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("opt_graph")
            ExportManager.set_interpret_graph(False)
            onnx_path = os.path.join(tmp, "preprocess.onnx")
            _write_resize_onnx(onnx_path)
            spec = {
                "models": {
                    "preprocess": {
                        "parameters": {
                            "model_path": "preprocess.onnx",
                            "backend": "onnx",
                            "md5sum": "old",
                            "sha256sum": "old",
                        }
                    }
                }
            }
            with open(os.path.join(tmp, "opt_graph.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            applied = optimize_graph(trt_compatible=True)
            self.assertIn("trt_compatible", applied)
            self.assertTrue(applied["trt_compatible"].models[0].rewritten)

            model = onnx.load(onnx_path)
            antialias = next(
                a.i for a in model.graph.node[0].attribute if a.name == "antialias"
            )
            self.assertEqual(int(antialias), 0)


if __name__ == "__main__":
    unittest.main()
