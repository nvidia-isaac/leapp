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

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from .schema import QUALIFIED_NAME

RUNTIME_METADATA_VERSION = 1


def encode_runtime_metadata(metadata: dict[str, Any]) -> str:
    """Serialize runtime metadata deterministically."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def decode_runtime_metadata(encoded: str) -> dict[str, Any]:
    """Decode and minimally validate runtime metadata."""
    if not encoded:
        raise ValueError(f"{QUALIFIED_NAME}: runtime_metadata is empty")
    try:
        metadata = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{QUALIFIED_NAME}: invalid runtime_metadata JSON: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{QUALIFIED_NAME}: runtime_metadata must decode to an object")
    version = metadata.get("schema_version")
    if version != RUNTIME_METADATA_VERSION:
        raise ValueError(
            f"{QUALIFIED_NAME}: unsupported runtime_metadata schema_version "
            f"{version!r}; expected {RUNTIME_METADATA_VERSION}"
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


def _ref_device(ref: Any) -> dict[str, Any]:
    device = ref.device
    text = str(device) if device is not None else ""
    if text.startswith("cuda"):
        parts = text.split(":", 1)
        return {
            "device_kind": "cuda",
            "device_index": int(parts[1]) if len(parts) == 2 and parts[1] else 0,
            "capture_device": text,
        }
    if text:
        return {"device_kind": text, "device_index": 0, "capture_device": text}
    return {"device_kind": "unknown", "device_index": 0, "capture_device": ""}


def _output_spec(
    index: int,
    ref: Any,
    shape: list[int],
    dtype: str,
    mask: bool,
) -> dict[str, Any]:
    return {
        "logical_index": int(index),
        "mask": bool(mask),
        "param_name": ref.name if mask else None,
        "dtype": dtype,
        "shape": shape,
        "num_bytes": _num_bytes(shape, dtype),
        "constant_fill": None if mask else 0,
    }


def build_runtime_metadata(
    *,
    segment: Any,
    input_refs: list[Any],
    output_refs: list[Any],
    output_shapes: list[list[int]],
    output_dtypes: list[str],
    output_mask: list[bool],
    wrp_name: str | None = None,
    bundle_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Build the metadata payload consumed by all runtime adapters."""
    if len(output_refs) != len(output_shapes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_shapes ({len(output_shapes)}) must have equal length"
        )
    if len(output_refs) != len(output_dtypes):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_dtypes ({len(output_dtypes)}) must have equal length"
        )
    if len(output_refs) != len(output_mask):
        raise ValueError(
            f"{QUALIFIED_NAME}: output_refs ({len(output_refs)}) and "
            f"output_mask ({len(output_mask)}) must have equal length"
        )

    inputs = []
    for index, ref in enumerate(input_refs):
        shape = _shape_to_list(ref.shape)
        dtype = _normalize_dtype_name(ref.dtype or "")
        inputs.append(
            {
                "logical_index": int(index),
                "param_name": ref.name,
                "dtype": dtype,
                "shape": shape,
                "num_bytes": _num_bytes(shape, dtype),
                "layout": "contiguous",
            }
        )

    outputs = [
        _output_spec(index, ref, shape, dtype, keep)
        for index, (ref, shape, dtype, keep) in enumerate(
            zip(output_refs, output_shapes, output_dtypes, output_mask)
        )
    ]

    runtime_target = {"device_kind": "unknown", "device_index": 0, "capture_device": ""}
    for ref in [*input_refs, *output_refs]:
        runtime_target = _ref_device(ref)
        if runtime_target["device_kind"] != "unknown":
            break

    bundle = {"format": "WRPB", "version": 1}
    if bundle_bytes is not None:
        bundle["num_bytes"] = len(bundle_bytes)
        bundle["sha256"] = hashlib.sha256(bundle_bytes).hexdigest()

    return {
        "schema_version": RUNTIME_METADATA_VERSION,
        "op_type": "leapp.warp_runner",
        "wrp_name": wrp_name or getattr(segment, "wrp_name", None) or "",
        "runtime_target": runtime_target,
        "inputs": inputs,
        "outputs": outputs,
        "apic_params": [
            {
                "name": spec["param_name"],
                "direction": "input",
                "num_bytes": spec["num_bytes"],
                "alias_group": None,
            }
            for spec in inputs
        ]
        + [
            {
                "name": spec["param_name"],
                "direction": "output",
                "num_bytes": spec["num_bytes"],
                "alias_group": None,
            }
            for spec in outputs
            if spec["mask"] and spec["param_name"]
        ],
        "bundle": bundle,
    }


def output_specs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = metadata.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError(f"{QUALIFIED_NAME}: runtime_metadata.outputs must be a list")
    return outputs


def output_shapes(metadata: dict[str, Any]) -> list[list[int]]:
    return [list(spec.get("shape", [])) for spec in output_specs(metadata)]


def output_dtypes(metadata: dict[str, Any]) -> list[str]:
    return [str(spec.get("dtype", "")) for spec in output_specs(metadata)]


def output_mask(metadata: dict[str, Any]) -> list[bool]:
    return [bool(spec.get("mask", True)) for spec in output_specs(metadata)]
