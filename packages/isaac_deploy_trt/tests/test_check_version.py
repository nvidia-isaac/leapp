#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for selecting a TensorRT release in ``isaac_deploy_trt check``."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from isaac_deploy_trt.cli import main
from isaac_deploy_trt.prepare import surgeon_ops_for_target
from isaac_deploy_trt.tensorrt_onnx_ops import SURGEON_UNSUPPORTED_OPS
from isaac_deploy_trt.validate import (
    check_onnx_ops_against_tensorrt,
    resolve_tensorrt_catalog,
)


def _write_model(path: Path, op_type: str) -> None:
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"])
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([node], "g", [value], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)


class CheckTensorRTVersionTests(unittest.TestCase):
    def test_resolve_10_16_orin_catalog(self):
        path = resolve_tensorrt_catalog("10.16", platform="orin")
        self.assertEqual(path.parent.name, "v10.16")
        self.assertIn("orin", path.name)
        self.assertEqual(
            resolve_tensorrt_catalog("v10.16.1", platform="orin"),
            path,
        )

    def test_check_uses_10_16_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "add.onnx"
            _write_model(model, "Add")
            result = check_onnx_ops_against_tensorrt(
                str(model), tensorrt="10.16", platform="orin"
            )
            default = check_onnx_ops_against_tensorrt(str(model))
            platform_only = check_onnx_ops_against_tensorrt(str(model), platform="orin")
        self.assertTrue(result.ok)
        self.assertIn("/v10.16/", result.catalog_path.replace("\\", "/"))
        self.assertIn("minimal/tensorrt_layers.json", default.catalog_path.replace("\\", "/"))
        self.assertEqual(default.tensorrt_version, "minimal")
        self.assertEqual(result.tensorrt_version, "10.16")
        self.assertTrue(default.ok)
        self.assertIn(
            "minimal/tensorrt_layers.json",
            platform_only.catalog_path.replace("\\", "/"),
        )

    def test_unknown_release_is_rejected(self):
        with self.assertRaises(KeyError):
            resolve_tensorrt_catalog("9.9")

    def test_cli_check_tensorrt_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "add.onnx"
            _write_model(model, "Add")
            code = main([
                "check",
                "--onnx",
                str(model),
                "--tensorrt",
                "10.16",
                "--platform",
                "orin",
                "--strict",
            ])
            unknown = main([
                "check",
                "--onnx",
                str(model),
                "--tensorrt",
                "9.9",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(unknown, 2)

    def test_default_rewrite_uses_minimal_catalog(self):
        ops, catalog = surgeon_ops_for_target()
        self.assertIn("minimal/tensorrt_layers.json", catalog.replace("\\", "/"))
        self.assertTrue(set(ops) <= SURGEON_UNSUPPORTED_OPS)

    def test_rewrite_uses_10_16_catalog(self):
        ops, catalog = surgeon_ops_for_target(tensorrt="10.16", platform="orin")
        self.assertIsNotNone(catalog)
        self.assertIn("v10.16", catalog)
        self.assertTrue(set(ops) <= SURGEON_UNSUPPORTED_OPS)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "add.onnx"
            out = Path(tmp) / "add_trt.onnx"
            _write_model(model, "Add")
            code = main([
                "optimize",
                str(model),
                "-o",
                str(out),
                "--tensorrt",
                "10.16",
                "--platform",
                "orin",
            ])
            unknown = main([
                "optimize",
                str(model),
                "-o",
                str(Path(tmp) / "bad.onnx"),
                "--tensorrt",
                "9.9",
            ])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
        self.assertEqual(unknown, 2)


if __name__ == "__main__":
    unittest.main()
