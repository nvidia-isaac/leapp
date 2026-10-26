#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for listing ONNX operators the TensorRT catalog marks unsupported."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from isaac_deploy_trt.tensorrt_onnx_ops import (
    format_unsupported_onnx_op_report,
    load_unsupported_onnx_op_report,
    unsupported_onnx_op_report,
)

_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "tensorrt_catalogs"
    / "v11.2"
    / "tensorrt_layers_cuda132_x86.json"
)


class UnsupportedOpsTests(unittest.TestCase):
    def test_report_includes_known_unsupported_op(self):
        report = unsupported_onnx_op_report()
        names = [row["name"] for row in report["ops"]]
        self.assertIn("AffineGrid", names)
        self.assertNotIn("Add", names)
        dynamic = next(row for row in report["ops"] if row["name"] == "DynamicQuantizeLinear")
        self.assertIn("Not supported", dynamic["restrictions"])
        self.assertTrue(dynamic["surgeon"])
        affine = next(row for row in report["ops"] if row["name"] == "AffineGrid")
        self.assertFalse(affine["surgeon"])

    def test_text_report_lists_unsupported_ops(self):
        text = format_unsupported_onnx_op_report(unsupported_onnx_op_report())
        self.assertIn("AffineGrid", text)
        self.assertIn("marked unsupported", text)

    def test_catalog_json(self):
        report = load_unsupported_onnx_op_report(str(_CATALOG))
        names = [row["name"] for row in report["ops"]]
        self.assertIn("Unique", names)
        self.assertEqual(report["platform"], "x86")
        encoded = json.dumps(report)
        self.assertIn("DynamicQuantizeLinear", encoded)


if __name__ == "__main__":
    unittest.main()
