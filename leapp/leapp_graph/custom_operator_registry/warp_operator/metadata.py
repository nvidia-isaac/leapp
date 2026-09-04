#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Runtime metadata for the embedded Warp APIC runner.

The metadata is intentionally backend-neutral. ONNX stores it as a string
attribute on ``com.nvidia.warp::WrpRunner``; PyTorch/TorchScript/ExportedProgram
pass the same string to ``leapp::warp_runner``. The WRPB bytes stay in a uint8
tensor input so ONNX external-data handling still applies to large bundles.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from leapp.utils.logging import _get_logger

from .schema import QUALIFIED_NAME


def encode_runtime_metadata(metadata: dict[str, Any]) -> str:
    """Serialize runtime metadata deterministically."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def decode_runtime_metadata(encoded: str) -> dict[str, Any]:
    """Decode and minimally validate runtime metadata."""
    if not encoded:
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: runtime_metadata is empty",
            error_type=ValueError,
        )
    try:
        metadata = json.loads(encoded)
    except json.JSONDecodeError as exc:
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: invalid runtime_metadata JSON: {exc}",
            error_type=ValueError,
            cause=exc,
        )
    if not isinstance(metadata, dict):
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: runtime_metadata must decode to an object",
            error_type=ValueError,
        )
    return metadata


def _shape_to_list(shape: Any) -> list[int]:
    if shape is None:
        return []
    return [int(dim) for dim in shape]


def _numel(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


def _normalize_dtype_name(dtype: str) -> str:
    if dtype.startswith("wp."):
        dtype = dtype[3:]
    for name in sorted(_DTYPE_BYTES, key=len, reverse=True):
        if dtype == name or dtype.endswith(f".{name}") or name in dtype:
            return name
    return dtype


def _num_bytes(shape: list[int], dtype: str) -> int:
    if not shape:
        return 0
    itemsize = _DTYPE_BYTES.get(dtype)
    if itemsize is None:
        return 0
    return _numel(shape) * itemsize


def _tensor_spec(
    index: int,
    ref: Any,
    shape: Any,
    dtype: str,
    *,
    mask: bool | None = None,
) -> dict[str, Any]:
    shape = _shape_to_list(shape)
    dtype = _normalize_dtype_name(dtype or "")
    spec = {
        "logical_index": int(index),
        "param_name": ref.name if mask is not False else None,
        "dtype": dtype,
        "shape": shape,
        "num_bytes": _num_bytes(shape, dtype),
        "warp_dtype": ref.dtype,
        "warp_shape": _shape_to_list(ref.shape),
        "component_shape": _shape_to_list(ref.component_shape),
    }
    if mask is not None:
        spec["mask"] = bool(mask)
    return spec


def build_runtime_metadata(
    *,
    segment: Any,
    input_refs: list[Any],
    output_refs: list[Any],
    output_shapes: list[list[int]],
    output_dtypes: list[str],
    output_mask: list[bool],
) -> dict[str, Any]:
    """Build the metadata payload consumed by all runtime adapters."""
    if len(output_refs) != len(output_shapes):
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_shapes ({len(output_shapes)}) must have equal length",
            error_type=ValueError,
        )
    if len(output_refs) != len(output_dtypes):
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_dtypes ({len(output_dtypes)}) must have equal length",
            error_type=ValueError,
        )
    if len(output_refs) != len(output_mask):
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_mask ({len(output_mask)}) must have equal length",
            error_type=ValueError,
        )

    inputs = []
    for index, ref in enumerate(input_refs):
        inputs.append(_tensor_spec(index, ref, ref.storage_shape, ref.storage_dtype))

    outputs = []
    for index, (ref, shape, dtype, keep) in enumerate(
        zip(output_refs, output_shapes, output_dtypes, output_mask)
    ):
        outputs.append(_tensor_spec(index, ref, shape, dtype, mask=keep))

    return {
        "inputs": inputs,
        "outputs": outputs,
    }


def output_specs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = metadata.get("outputs", [])
    if not isinstance(outputs, list):
        _get_logger().fatal(
            f"{QUALIFIED_NAME}: runtime_metadata.outputs must be a list",
            error_type=ValueError,
        )
    return outputs


def output_shapes(metadata: dict[str, Any]) -> list[list[int]]:
    return [list(spec.get("shape", [])) for spec in output_specs(metadata)]


def output_dtypes(metadata: dict[str, Any]) -> list[str]:
    return [str(spec.get("dtype", "")) for spec in output_specs(metadata)]


def output_mask(metadata: dict[str, Any]) -> list[bool]:
    return [bool(spec.get("mask", True)) for spec in output_specs(metadata)]
