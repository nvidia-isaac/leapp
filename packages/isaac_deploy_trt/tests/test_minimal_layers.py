#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the intersection of checked-in TensorRT layer catalogs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from isaac_deploy_trt.cli import main
from isaac_deploy_trt.minimal_layers import minimal_supported_layers

_CHECKED_IN = (
    Path(__file__).resolve().parents[1]
    / "tensorrt_catalogs"
    / "minimal"
    / "tensorrt_layers.json"
)


class MinimalLayersTests(unittest.TestCase):
    def test_intersection_drops_newer_and_jetpack_layers(self):
        report = minimal_supported_layers()
        names = set(report["layers"])
        excluded = {item["name"] for item in report["excluded"]}
        self.assertIn("CONVOLUTION", names)
        self.assertNotIn("ATTENTION", names)
        self.assertIn("ATTENTION", excluded)
        self.assertIn("MOE", excluded)
        self.assertGreater(report["layer_count"], 0)
        self.assertEqual(report["layer_count"], len(names))

    def test_checked_in_catalog_matches_intersection(self):
        report = minimal_supported_layers()
        checked = json.loads(_CHECKED_IN.read_text(encoding="utf-8"))
        self.assertEqual(checked["layers"], report["layers"])
        self.assertEqual(checked["onnx_operators"], report["onnx_operators"])
        self.assertEqual(
            [item["name"] for item in checked["excluded"]],
            [item["name"] for item in report["excluded"]],
        )

    def test_cli_writes_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.json"
            code = main(["minimal-layers", "-o", str(path)])
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(payload["kind"], "minimal_supported_layers")
        self.assertIn("CONVOLUTION", payload["layers"])


if __name__ == "__main__":
    unittest.main()
