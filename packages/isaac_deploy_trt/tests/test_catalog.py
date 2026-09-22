#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from isaac_deploy_trt.catalog import build_catalog
from isaac_deploy_trt.headers import parse_tensorrt_headers

SAMPLE_OPERATORS_MD = """
# Supported ONNX Operators

TensorRT 11.2 supports operators in the inclusive range of opset 9 to opset 24.

## Operator Support Matrix

| Operator | Supported | Supported Types | Restrictions |
|---------------------------|------------|-----------------|------------------------------------------------------------------------------------------------------------------------|
| Abs | Y | FP32, FP16, BF16, INT32, INT64 |
| AffineGrid | N |
"""

SAMPLE_HEADERS = """
enum class DataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT32 = 3,
};
enum class LayerType : int32_t
{
    kCONVOLUTION = 0,
};
enum class NetworkDefinitionCreationFlag : int32_t
{
    kSTRONGLY_TYPED TRT_DEPRECATED_ENUM = 0,
};
"""


class TestCatalogHeaders(unittest.TestCase):
    def test_parse_headers(self):
        catalog = parse_tensorrt_headers([SAMPLE_HEADERS], sources=["fixture.h"])
        self.assertEqual(catalog.data_types, ["kFLOAT", "kHALF", "kINT32"])
        self.assertEqual(catalog.layer_type_names, ["CONVOLUTION"])
        self.assertTrue(catalog.strongly_typed)

    def test_join_operators_md_with_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            include = Path(tmp) / "include"
            include.mkdir()
            (include / "NvInfer.h").write_text(SAMPLE_HEADERS, encoding="utf-8")
            catalog = build_catalog(
                operators_md=str(md_path),
                include_dir=str(include),
            )
        self.assertEqual(
            catalog["onnx_operators"]["Abs"]["nvinfer_datatypes"],
            ["kFLOAT", "kHALF", "kINT32"],
        )
        self.assertNotIn("kBF16", catalog["onnx_operators"]["Abs"]["nvinfer_datatypes"])
        self.assertTrue(catalog["nvinfer"]["strongly_typed"])
        json.dumps(catalog)
        self.assertEqual(catalog["supported_releases"], ["v11.2"])

    def test_shared_releases(self):
        headers = parse_tensorrt_headers([SAMPLE_HEADERS], sources=["fixture.h"])
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            catalog = build_catalog(
                operators_md=str(md_path),
                releases=["v11.1", "11.2"],
                headers_by_release={"v11.1": headers, "v11.2": headers},
            )
        self.assertEqual(catalog["supported_releases"], ["v11.1", "v11.2"])
        self.assertNotIn("databases", catalog)


if __name__ == "__main__":
    unittest.main()
