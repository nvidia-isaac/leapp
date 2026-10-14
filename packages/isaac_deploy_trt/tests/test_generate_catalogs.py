#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for CUDA 13.2 TensorRT layer catalog generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isaac_deploy_trt.headers import parse_tensorrt_headers
from isaac_deploy_trt.platform_catalog import (
    DEFAULT_PLATFORMS,
    PLATFORMS,
    generate,
    parse_operators_md,
    resolve_platform_name,
)


SAMPLE_OPERATORS_MD = """
# Supported ONNX Operators

TensorRT 11.2 supports operators in the inclusive range of opset 9 to opset 24.

## Operator Support Matrix

| Operator | Supported | Supported Types | Restrictions |
|---------------------------|------------|-----------------|------------------------------------------------------------------------------------------------------------------------|
| Abs | Y | FP32, FP16, BF16, INT32, INT64 |
| Attention | Y | FP32, FP16, BF16, INT8, FP8 | Q, K, V must be 4D |
| AffineGrid | N |
| Conv | Y | FP32, FP16, BF16 |
| DequantizeLinear | Y | INT8, FP8, FP4, INT4 | x_zero_point must be zero |
| DynamicQuantizeLinear | N | Not supported. TensorRT IDynamicQuantize can be composed from ONNX local functions.
| Relu | Y | FP32, FP16, BF16 |
| Xor | Y | BOOL
"""


class TestGenerateTensorrtSupportedLayers(unittest.TestCase):
    def test_parse_operators_md(self):
        doc = parse_operators_md(SAMPLE_OPERATORS_MD, source="sample")
        self.assertEqual(doc.tensorrt_version, "11.2")
        self.assertEqual(doc.opset_min, 9)
        self.assertEqual(doc.opset_max, 24)
        by_name = {op.name: op for op in doc.ops}
        self.assertTrue(by_name["Abs"].supported)
        self.assertIn("BF16", by_name["Abs"].types)
        self.assertFalse(by_name["AffineGrid"].supported)
        self.assertTrue(by_name["Xor"].supported)
        self.assertEqual(by_name["Xor"].types, ["BOOL"])
        self.assertIn("Not supported", by_name["DynamicQuantizeLinear"].restrictions)

    def test_orin_drops_blackwell_precisions_and_keeps_dla(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            result = generate(
                platforms=["orin", "thor", "x86", "dgx"],
                operators_md=str(md_path),
                output_dir=Path(tmp) / "out",
            )
        orin = next(c for c in result["catalogs"] if c["platform"] == "orin")
        thor = next(c for c in result["catalogs"] if c["platform"] == "thor")
        x86 = next(c for c in result["catalogs"] if c["platform"] == "x86")
        abs_types = orin["onnx_operators"]["Abs"]["types"]
        self.assertNotIn("BF16", abs_types)
        self.assertIn("FP16", abs_types)
        self.assertIn("INT8", orin["onnx_operators"]["Attention"]["types"])
        self.assertNotIn("FP8", orin["onnx_operators"]["Attention"]["types"])
        self.assertEqual(
            orin["onnx_operators"]["DequantizeLinear"]["types"], ["INT8"]
        )
        self.assertTrue(orin["dla"])
        self.assertTrue(orin["tensorrt_layers"]["CONVOLUTION"]["dla"])
        self.assertFalse(orin["tensorrt_layers"]["MATRIX_MULTIPLY"]["dla"])
        self.assertFalse(orin["tensorrt_layers"]["ATTENTION"]["gpu"])

        self.assertIn("FP8", thor["onnx_operators"]["Attention"]["types"])
        self.assertIn("BF16", thor["onnx_operators"]["Abs"]["types"])
        self.assertFalse(thor["dla"])
        self.assertTrue(x86["tensorrt_11"])
        self.assertTrue(x86["tensorrt_layers"]["ATTENTION"]["gpu"])
        self.assertEqual(x86["supported_releases"], ["v11.2"])
        self.assertEqual(
            x86["onnx_operators"]["Abs"]["nvinfer_datatypes"],
            ["kFLOAT", "kHALF", "kBF16", "kINT32", "kINT64"],
        )
        self.assertIn("kFLOAT", orin["onnx_operators"]["Abs"]["nvinfer_datatypes"])
        self.assertNotIn("kBF16", orin["onnx_operators"]["Abs"]["nvinfer_datatypes"])

    def test_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            out = Path(tmp) / "out"
            generate(
                platforms=["orin"],
                operators_md=str(md_path),
                output_dir=out,
            )
            json_path = out / "tensorrt_layers_cuda132_orin.json"
            md_out = out / "tensorrt_layers_cuda132_orin.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_out.is_file())
            text = md_out.read_text(encoding="utf-8")
            self.assertIn("Jetson Orin", text)
            self.assertIn("| Conv |", text)

    def test_platform_keys(self):
        self.assertEqual(
            set(PLATFORMS),
            {
                "x86",
                "orin",
                "thor",
                "dgx",
                "noble",
                "noble-fastos",
                "noble-jetpack",
            },
        )
        self.assertEqual(DEFAULT_PLATFORMS, ("noble", "noble-fastos", "noble-jetpack"))
        self.assertEqual(resolve_platform_name("amd64"), "noble")
        self.assertEqual(resolve_platform_name("spark"), "noble-fastos")
        self.assertEqual(resolve_platform_name("jetson"), "noble-jetpack")
        self.assertEqual(PLATFORMS["noble"].isaac_debian_dist, "noble")
        self.assertEqual(PLATFORMS["noble-fastos"].isaac_debian_dist, "noble-fastos")
        self.assertEqual(PLATFORMS["noble-jetpack"].isaac_debian_dist, "noble-jetpack")
        self.assertFalse(PLATFORMS["noble-fastos"].tensorrt_11)
        self.assertFalse(PLATFORMS["noble-jetpack"].tensorrt_11)
        self.assertTrue(PLATFORMS["noble-jetpack"].dla)

    def test_isaac_debian_platforms_write_hyphenated_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            out = Path(tmp) / "out"
            result = generate(
                platforms=["jetson", "spark", "amd64"],
                operators_md=str(md_path),
                output_dir=out,
            )
            names = {c["platform"] for c in result["catalogs"]}
            self.assertEqual(names, {"noble", "noble-fastos", "noble-jetpack"})
            self.assertTrue((out / "tensorrt_layers_cuda132_noble.json").is_file())
            self.assertTrue(
                (out / "tensorrt_layers_cuda132_noble-fastos.json").is_file()
            )
            self.assertTrue(
                (out / "tensorrt_layers_cuda132_noble-jetpack.json").is_file()
            )
            jetpack = next(
                c for c in result["catalogs"] if c["platform"] == "noble-jetpack"
            )
            self.assertEqual(jetpack["isaac_debian_dist"], "noble-jetpack")
            self.assertFalse(jetpack["tensorrt_11"])
            self.assertFalse(jetpack["tensorrt_layers"]["ATTENTION"]["gpu"])

    def test_headers_join_nvinfer_types_and_layer_enum(self):
        sample_headers = """
enum class DataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kINT64 = 8,
};
enum class LayerType : int32_t
{
    kCONVOLUTION = 0,
    kMOE = 55,
};
enum class NetworkDefinitionCreationFlag : int32_t
{
    kSTRONGLY_TYPED TRT_DEPRECATED_ENUM = 0,
};
"""
        headers = parse_tensorrt_headers([sample_headers], sources=["fixture.h"])
        self.assertIn("kFLOAT", headers.data_types)
        self.assertNotIn("kBF16", headers.data_types)
        self.assertTrue(headers.strongly_typed)
        self.assertIn("MOE", headers.layer_type_names)

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            include = Path(tmp) / "include"
            include.mkdir()
            (include / "NvInfer.h").write_text(sample_headers, encoding="utf-8")
            result = generate(
                platforms=["x86", "orin"],
                operators_md=str(md_path),
                include_dir=str(include),
            )
        x86 = next(c for c in result["catalogs"] if c["platform"] == "x86")
        self.assertEqual(
            x86["onnx_operators"]["Abs"]["nvinfer_datatypes"],
            ["kFLOAT", "kHALF", "kINT32", "kINT64"],
        )
        self.assertNotIn("kBF16", x86["onnx_operators"]["Abs"]["nvinfer_datatypes"])
        self.assertTrue(x86["nvinfer"]["strongly_typed"])
        self.assertIn("MOE", x86["tensorrt_layers"])
        self.assertTrue(x86["tensorrt_layers"]["MOE"]["gpu"])
        orin = next(c for c in result["catalogs"] if c["platform"] == "orin")
        self.assertFalse(orin["tensorrt_layers"]["MOE"]["gpu"])

    def test_identical_release_headers_share_one_database(self):
        headers = parse_tensorrt_headers(
            [
                """
enum class DataType : int32_t { kFLOAT = 0, kHALF = 1, };
enum class LayerType : int32_t { kCONVOLUTION = 0, };
"""
            ],
            sources=["same.h"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            result = generate(
                platforms=["x86"],
                operators_md=str(md_path),
                releases=["v11.1", "v11.2"],
                headers_by_release={"v11.1": headers, "v11.2": headers},
            )
        self.assertEqual(len(result["databases"]), 1)
        self.assertEqual(
            result["catalogs"][0]["supported_releases"], ["v11.1", "v11.2"]
        )
        self.assertEqual(result["databases"][0]["id"], "v11.2")

    def test_different_release_headers_make_two_databases(self):
        h10 = parse_tensorrt_headers(
            ["enum class DataType : int32_t { kFLOAT = 0, };"],
            sources=["t10.h"],
        )
        h11 = parse_tensorrt_headers(
            ["enum class DataType : int32_t { kFLOAT = 0, kBF16 = 7, };"],
            sources=["t11.h"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "operators.md"
            md_path.write_text(SAMPLE_OPERATORS_MD, encoding="utf-8")
            out = Path(tmp) / "out"
            result = generate(
                platforms=["x86"],
                operators_md=str(md_path),
                output_dir=out,
                releases=["v10.16", "v11.2"],
                headers_by_release={"v10.16": h10, "v11.2": h11},
            )
            self.assertEqual(len(result["databases"]), 2)
            by_id = {db["id"]: db for db in result["databases"]}
            self.assertEqual(by_id["v10.16"]["supported_releases"], ["v10.16"])
            self.assertEqual(by_id["v11.2"]["supported_releases"], ["v11.2"])
            self.assertTrue((out / "v10.16" / "tensorrt_layers_cuda132_x86.json").is_file())
            self.assertTrue((out / "v11.2" / "tensorrt_layers_cuda132_x86.json").is_file())
            index = (out / "index.json").read_text(encoding="utf-8")
            self.assertIn("release_to_database", index)


if __name__ == "__main__":
    unittest.main()
