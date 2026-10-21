#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for TensorRT-aligned LEAPP export bundles."""

from __future__ import annotations

import os
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper
import yaml

try:
    from isaac_deploy_trt.surgery import graph_io_signature
    from leapp.backends.tensorrt_leapp_bundle import (
        align_leapp_export_for_tensorrt,
        rewrite_leapp_export_for_tensorrt,
    )
except ImportError:
    graph_io_signature = None  # type: ignore[assignment]
    align_leapp_export_for_tensorrt = None  # type: ignore[assignment]
    rewrite_leapp_export_for_tensorrt = None  # type: ignore[assignment]

from leapp.backends.onnx_export_backend import ONNXExportBackend


def _write_abs_onnx(path: str) -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Abs", ["x"], ["y"], name="abs0")],
        name="abs",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        path,
    )


def _write_det_onnx(path: str) -> None:
    graph = helper.make_graph(
        nodes=[helper.make_node("Det", ["x"], ["y"], name="det0")],
        name="det",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        path,
    )


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


class TestTensorrtLeappBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if graph_io_signature is None:
            raise unittest.SkipTest("isaac_deploy_trt is not installed")

    def test_aligns_filenames_io_and_yaml_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            dst = os.path.join(tmp, "dst")
            os.makedirs(src)
            abs_path = os.path.join(src, "backbone.onnx")
            video_path = os.path.join(src, "preprocess_video.onnx")
            _write_abs_onnx(abs_path)
            _write_resize_onnx(video_path)
            sidecar = os.path.join(src, "preprocess_video_trt.onnx")
            _write_resize_onnx(sidecar)

            spec = {
                "models": {
                    "backbone": {
                        "parameters": {
                            "model_path": "backbone.onnx",
                            "backend": "onnx",
                            "md5sum": "old",
                            "sha256sum": "old",
                        }
                    },
                    "preprocess_video": {
                        "parameters": {
                            "model_path": "preprocess_video.onnx",
                            "backend": "onnx",
                            "md5sum": "old",
                            "sha256sum": "old",
                        }
                    },
                }
            }
            yaml_path = os.path.join(src, "exported_leapp.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            result = align_leapp_export_for_tensorrt(src, dst)
            by_name = {model.filename: model for model in result.models}
            self.assertFalse(by_name["backbone.onnx"].rewritten)
            self.assertTrue(by_name["preprocess_video.onnx"].rewritten)
            self.assertTrue(os.path.isfile(os.path.join(dst, "preprocess_video.onnx")))
            self.assertFalse(os.path.isfile(os.path.join(dst, "preprocess_video_trt.onnx")))

            source_io = graph_io_signature(onnx.load(video_path))
            aligned_io = graph_io_signature(
                onnx.load(os.path.join(dst, "preprocess_video.onnx"))
            )
            self.assertEqual(source_io, aligned_io)

            with open(os.path.join(dst, "exported_leapp.yaml"), encoding="utf-8") as handle:
                out_spec = yaml.safe_load(handle)
            video_params = out_spec["models"]["preprocess_video"]["parameters"]
            self.assertEqual(video_params["model_path"], "preprocess_video.onnx")
            self.assertEqual(
                video_params["sha256sum"], by_name["preprocess_video.onnx"].sha256sum
            )
            self.assertNotEqual(video_params["sha256sum"], "old")
            self.assertEqual(
                out_spec["models"]["backbone"]["parameters"]["backend"], "onnx"
            )
            self.assertTrue(
                out_spec["models"]["backbone"]["parameters"]["tensorrt_compatible"]
            )
            self.assertTrue(
                video_params["tensorrt_compatible"]
            )

    def test_yaml_marks_unsupported_ops_not_tensorrt_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            dst = os.path.join(tmp, "dst")
            os.makedirs(src)
            det_path = os.path.join(src, "head.onnx")
            _write_det_onnx(det_path)
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
            with open(os.path.join(src, "exported_leapp.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            align_leapp_export_for_tensorrt(src, dst)
            with open(os.path.join(dst, "exported_leapp.yaml"), encoding="utf-8") as handle:
                out_spec = yaml.safe_load(handle)
            self.assertFalse(
                out_spec["models"]["head"]["parameters"]["tensorrt_compatible"]
            )

    def test_rewrite_leapp_export_for_tensorrt_inplace(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "preprocess_video.onnx")
            _write_resize_onnx(video_path)
            spec = {
                "models": {
                    "preprocess_video": {
                        "parameters": {
                            "model_path": "preprocess_video.onnx",
                            "backend": "onnx",
                            "md5sum": "old",
                            "sha256sum": "old",
                        }
                    }
                }
            }
            yaml_path = os.path.join(tmp, "graph.yaml")
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(spec, handle)

            before = onnx.load(video_path)
            antialias = next(
                a.i for a in before.graph.node[0].attribute if a.name == "antialias"
            )
            self.assertEqual(int(antialias), 1)

            result = rewrite_leapp_export_for_tensorrt(tmp)
            self.assertEqual(result.output_dir, os.path.abspath(tmp))
            self.assertTrue(result.models[0].rewritten)

            after = onnx.load(video_path)
            antialias = next(
                a.i for a in after.graph.node[0].attribute if a.name == "antialias"
            )
            self.assertEqual(int(antialias), 0)
            with open(yaml_path, encoding="utf-8") as handle:
                out_spec = yaml.safe_load(handle)
            self.assertNotEqual(
                out_spec["models"]["preprocess_video"]["parameters"]["md5sum"],
                "old",
            )

    def test_onnx_export_backend_annotates_tensorrt_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "abs.onnx")
            _write_abs_onnx(onnx_path)
            backend = ONNXExportBackend.__new__(ONNXExportBackend)
            backend.backend_params = {}
            backend._annotate_tensorrt_compatible(onnx_path)
            self.assertTrue(backend.tensorrt_compatible)
            self.assertTrue(backend.get_backend_metadata()["tensorrt_compatible"])

            det_path = os.path.join(tmp, "det.onnx")
            _write_det_onnx(det_path)
            backend.tensorrt_compatible = None
            backend._annotate_tensorrt_compatible(det_path)
            self.assertFalse(backend.tensorrt_compatible)

            backend.backend_params = {"annotate_tensorrt_compatible": False}
            backend.tensorrt_compatible = None
            backend._annotate_tensorrt_compatible(onnx_path)
            self.assertIsNone(getattr(backend, "tensorrt_compatible", None))
            self.assertNotIn("tensorrt_compatible", backend.get_backend_metadata())


if __name__ == "__main__":
    unittest.main()
