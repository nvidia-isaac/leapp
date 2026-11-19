#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for ``leapp.optimize_graph``."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

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

            node = type("Node", (), {})()
            node.name = "preprocess"
            node.model_path = onnx_path
            node.md5sum = "old"
            node.sha256sum = "old"
            _MANAGER.nodes["preprocess"] = node
            _MANAGER.detected_nodes = {
                "preprocess": {
                    "parameters": {"md5sum": "old", "sha256sum": "old"}
                }
            }
            try:
                with patch.object(_MANAGER, "validate_optimized_onnx_models") as validate:
                    applied = optimize_graph(
                        trt_compatible=True,
                        rtol=2e-3,
                        atol=3e-5,
                    )
                detected_sha = _MANAGER.detected_nodes["preprocess"]["parameters"]["sha256sum"]
            finally:
                _MANAGER.nodes.pop("preprocess", None)
                _MANAGER.detected_nodes = {}
            self.assertEqual(applied["trt_compatible"], ["minimal"])
            validate.assert_called_once()
            self.assertEqual(
                [os.path.basename(path) for path in validate.call_args.args[0]],
                ["preprocess.onnx"],
            )
            self.assertEqual(validate.call_args.kwargs["rtol"], 2e-3)
            self.assertEqual(validate.call_args.kwargs["atol"], 3e-5)
            self.assertTrue(validate.call_args.kwargs["strict"])
            self.assertFalse(validate.call_args.kwargs["raise_on_failure"])
            self.assertNotEqual(node.md5sum, "old")
            self.assertNotEqual(node.sha256sum, "old")
            self.assertEqual(detected_sha, node.sha256sum)
            with open(os.path.join(tmp, "opt_graph.yaml"), encoding="utf-8") as handle:
                updated = yaml.safe_load(handle)
            self.assertEqual(
                updated["models"]["preprocess"]["parameters"]["sha256sum"],
                node.sha256sum,
            )

            model = onnx.load(onnx_path)
            antialias = next(
                a.i for a in model.graph.node[0].attribute if a.name == "antialias"
            )
            self.assertEqual(int(antialias), 0)

    def test_trt_version_selects_catalog(self):
        try:
            import isaac_deploy_trt.surgery  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("opt_graph")
            ExportManager.set_interpret_graph(False)
            _write_resize_onnx(os.path.join(tmp, "preprocess.onnx"))
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

            with patch.object(_MANAGER, "validate_optimized_onnx_models"):
                applied = optimize_graph(trt_compatible=True, trt_version=10.14)

            self.assertEqual(applied["trt_compatible"], ["10.14"])
            with open(os.path.join(tmp, "opt_graph.yaml"), encoding="utf-8") as handle:
                updated = yaml.safe_load(handle)
            parameters = updated["models"]["preprocess"]["parameters"]
            self.assertNotIn("optimization", parameters)
            self.assertNotIn("tensorrt_version", parameters)

    def test_unsupported_model_omits_trt_compatible(self):
        try:
            import isaac_deploy_trt.surgery  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("opt_graph")
            ExportManager.set_interpret_graph(False)
            graph = helper.make_graph(
                nodes=[helper.make_node("Det", ["x"], ["y"], name="det0")],
                name="det",
                inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
                outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
            )
            onnx.save(
                helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
                os.path.join(tmp, "head.onnx"),
            )
            spec = {
                "models": {
                    "head": {
                        "parameters": {
                            "model_path": "head.onnx",
                            "backend": "onnx",
                            "md5sum": "old",
                            "sha256sum": "old",
                        }
                    }
                }
            }
            with open(os.path.join(tmp, "opt_graph.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            applied = optimize_graph(trt_compatible=True, validate=False)
            self.assertNotIn("trt_compatible", applied)

    def test_numerical_mismatch_restores_original_onnx(self):
        try:
            import isaac_deploy_trt.surgery  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            _MANAGER.set_save_path(tmp)
            _MANAGER.set_graph_name("opt_graph")
            ExportManager.set_interpret_graph(False)
            good_path = os.path.join(tmp, "good.onnx")
            bad_path = os.path.join(tmp, "bad.onnx")
            _write_resize_onnx(good_path)
            _write_resize_onnx(bad_path)
            with open(bad_path, "rb") as handle:
                original_bad = handle.read()
            spec = {
                "models": {
                    "good": {
                        "parameters": {
                            "model_path": "good.onnx",
                            "backend": "onnx",
                            "md5sum": "old-good",
                            "sha256sum": "old-good",
                        }
                    },
                    "bad": {
                        "parameters": {
                            "model_path": "bad.onnx",
                            "backend": "onnx",
                            "md5sum": "old-bad",
                            "sha256sum": "old-bad",
                        }
                    },
                }
            }
            yaml_path = os.path.join(tmp, "opt_graph.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            class _Backend:
                compiled_model = None

            class _Node:
                def __init__(self, name, path, fail):
                    self.name = name
                    self.model_path = path
                    self.md5sum = "old"
                    self.sha256sum = "old"
                    self.export_backend = _Backend()
                    self._fail = fail

                def validate_compiled_model(self, rtol, atol):
                    if self._fail:
                        return False, "numerical mismatch"
                    return True, None

            good = _Node("good", good_path, fail=False)
            bad = _Node("bad", bad_path, fail=True)
            _MANAGER.nodes["good"] = good
            _MANAGER.nodes["bad"] = bad
            _MANAGER.detected_nodes = {
                "good": {"parameters": {"md5sum": "old", "sha256sum": "old"}},
                "bad": {"parameters": {"md5sum": "old", "sha256sum": "old"}},
            }
            try:
                with self.assertRaises(Exception) as raised:
                    optimize_graph(trt_compatible=True)
            finally:
                _MANAGER.nodes.pop("good", None)
                _MANAGER.nodes.pop("bad", None)
                _MANAGER.detected_nodes = {}

            self.assertIn("bad.onnx", str(raised.exception))
            self.assertIn("original ONNX file was restored", str(raised.exception))
            with open(bad_path, "rb") as handle:
                self.assertEqual(handle.read(), original_bad)
            good_model = onnx.load(good_path)
            antialias = next(
                attribute.i
                for attribute in good_model.graph.node[0].attribute
                if attribute.name == "antialias"
            )
            self.assertEqual(int(antialias), 0)
            with open(yaml_path, encoding="utf-8") as handle:
                updated = yaml.safe_load(handle)
            self.assertEqual(
                updated["models"]["bad"]["parameters"]["sha256sum"],
                "old-bad",
            )
            self.assertNotEqual(
                updated["models"]["good"]["parameters"]["sha256sum"],
                "old-good",
            )
            self.assertEqual(bad.sha256sum, "old")
            self.assertNotEqual(good.sha256sum, "old")


if __name__ == "__main__":
    unittest.main()
