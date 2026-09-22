#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Rewrite ONNX graphs so TensorRT can parse them, without LEAPP."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "apply_tensorrt_rewrites",
    "check_onnx_ops_against_tensorrt",
    "merge_leapp_onnx_pipeline",
    "rewrite_directory",
    "rewrite_onnx_file",
    "rewrite_onnx_for_tensorrt",
    "save_merged_pipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"rewrite_directory", "rewrite_onnx_file"}:
        from isaac_deploy_trt.prepare import rewrite_directory, rewrite_onnx_file

        mapping = {
            "rewrite_directory": rewrite_directory,
            "rewrite_onnx_file": rewrite_onnx_file,
        }
        return mapping[name]
    if name in {
        "apply_tensorrt_rewrites",
        "rewrite_onnx_for_tensorrt",
        "surgery",
    }:
        import importlib

        surgery_mod = importlib.import_module("isaac_deploy_trt.surgery")
        mapping = {
            "apply_tensorrt_rewrites": surgery_mod.apply_tensorrt_rewrites,
            "rewrite_onnx_for_tensorrt": surgery_mod.rewrite_onnx_for_tensorrt,
            "surgery": surgery_mod,
        }
        return mapping[name]
    if name == "check_onnx_ops_against_tensorrt":
        from isaac_deploy_trt.validate import check_onnx_ops_against_tensorrt

        return check_onnx_ops_against_tensorrt
    if name in {"merge_leapp_onnx_pipeline", "save_merged_pipeline"}:
        from isaac_deploy_trt.pipeline_merge import (
            merge_leapp_onnx_pipeline,
            save_merged_pipeline,
        )

        mapping = {
            "merge_leapp_onnx_pipeline": merge_leapp_onnx_pipeline,
            "save_merged_pipeline": save_merged_pipeline,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
