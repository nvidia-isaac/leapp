#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared schema and encode/decode helpers for ``leapp::warp_runner``."""

from __future__ import annotations

import torch

NAMESPACE = "leapp"
OP_NAME = "warp_runner"
QUALIFIED_NAME = f"{NAMESPACE}::{OP_NAME}"

_SCHEMA = f"{OP_NAME}(Tensor[] inputs, str runtime_metadata, Tensor bundle) -> Tensor[]"

ONNX_WRP_DOMAIN = "com.nvidia.warp"
ONNX_WRP_OPSET = 1
ONNX_WRP_OP_TYPE = "WrpRunner"


def get_op() -> "torch._ops.OpOverloadPacket":
    return getattr(getattr(torch.ops, NAMESPACE), OP_NAME)


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{QUALIFIED_NAME}: unknown output dtype name '{name}'")
    return dtype


def _format_output_shape_attr(output_shapes: list[list[int]]) -> str:
    parts = []
    for shape in output_shapes:
        if not shape:
            parts.append("0")
            continue
        parts.append(",".join(str(int(dim)) for dim in shape))
    return ";".join(parts)


def encode_output_shapes(output_shapes: list[list[int]]) -> str:
    return _format_output_shape_attr(output_shapes)


def decode_output_shapes(encoded: str) -> list[list[int]]:
    if not encoded:
        return []
    shapes: list[list[int]] = []
    for part in encoded.split(";"):
        part = part.strip()
        if not part or part == "0":
            shapes.append([])
            continue
        shapes.append([int(dim) for dim in part.split(",") if dim.strip()])
    return shapes


def encode_output_dtypes(output_dtypes: list[str]) -> str:
    return ",".join(output_dtypes)


def decode_output_dtypes(encoded: str) -> list[str]:
    if not encoded:
        return []
    return [part.strip() for part in encoded.split(",") if part.strip()]


def encode_output_mask(output_mask: list[bool]) -> str:
    return ",".join("1" if flag else "0" for flag in output_mask)


def decode_output_mask(encoded: str) -> list[bool]:
    if not encoded:
        return []
    return [part.strip() == "1" for part in encoded.split(",") if part.strip()]
