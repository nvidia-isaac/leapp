#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Join onnx-tensorrt operators.md with TensorRT header enumerations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from isaac_deploy_trt.headers import (
    TensorRTHeaderCatalog,
    load_tensorrt_headers,
    map_onnx_dtypes_to_nvinfer,
    normalize_release_tag,
)

DEFAULT_OPERATORS_MD_URL = (
    "https://raw.githubusercontent.com/onnx/onnx-tensorrt/main/docs/operators.md"
)

TYPE_ALIASES = {
    "FLOAT32": "FP32",
    "FLOAT": "FP32",
    "FLOAT16": "FP16",
    "BFLOAT16": "BF16",
    "DOUBLE": "FP64",
}
KNOWN_DTYPES = {
    "FP32", "FP16", "BF16", "FP8", "FP4", "FP64",
    "INT8", "INT4", "INT32", "INT64", "UINT8", "BOOL", "TF32",
}


@dataclass
class OnnxOpRow:
    name: str
    supported: bool
    types: List[str]
    restrictions: str


def parse_operators_md(text: str, *, source: str = "operators.md") -> Dict[str, object]:
    version = "unknown"
    opset_min, opset_max = 9, 24
    header = re.search(
        r"TensorRT\s+([0-9.]+)\s+supports operators in the inclusive range of opset\s+(\d+)\s+to opset\s+(\d+)",
        text,
        re.IGNORECASE,
    )
    if header:
        version = header.group(1)
        opset_min = int(header.group(2))
        opset_max = int(header.group(3))

    ops: List[OnnxOpRow] = []
    in_table = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## ") and "Operator Support Matrix" in line:
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or cells[0] in {"Operator", ""}:
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        name = cells[0]
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            continue
        supported_cell = cells[1] if len(cells) > 1 else "N"
        supported = supported_cell.upper().startswith("Y")
        types_cell = cells[2].strip() if len(cells) > 2 else ""
        parts = [
            TYPE_ALIASES.get(part.strip().upper().replace(" ", ""), part.strip().upper().replace(" ", ""))
            for part in types_cell.split(",")
            if part.strip()
        ]
        types = [part for part in parts if part in KNOWN_DTYPES]
        restrictions = cells[3] if len(cells) > 3 else ""
        if not restrictions and types_cell and not types:
            restrictions = types_cell
        ops.append(
            OnnxOpRow(
                name=name,
                supported=supported,
                types=types,
                restrictions=restrictions.strip(),
            )
        )
    if not ops:
        raise ValueError(f"No operator rows parsed from {source}")
    return {
        "tensorrt_version": version,
        "opset_min": opset_min,
        "opset_max": opset_max,
        "source": source,
        "ops": ops,
    }


def catalog_fingerprint(
    onnx_operators: Dict[str, object],
    headers: Optional[TensorRTHeaderCatalog],
) -> tuple:
    ops = tuple(
        (
            name,
            info.get("supported"),
            tuple(info.get("types") or []),
            info.get("restrictions"),
        )
        for name, info in sorted(onnx_operators.items())
    )
    if headers is None:
        header_key = None
    else:
        header_key = (
            tuple(headers.data_types),
            tuple(headers.layer_types),
            tuple(headers.builder_flags),
            tuple(headers.network_definition_creation_flags),
        )
    return (ops, header_key)


def build_catalog(
    *,
    operators_md: Optional[str] = None,
    include_dir: Optional[str] = None,
    fetch_headers: bool = False,
    headers: Optional[TensorRTHeaderCatalog] = None,
    releases: Optional[List[str]] = None,
    headers_by_release: Optional[Dict[str, TensorRTHeaderCatalog]] = None,
) -> Dict[str, object]:
    tags = [normalize_release_tag(tag) for tag in (releases or [])] or [None]
    groups: Dict[tuple, Dict[str, object]] = {}
    order: List[tuple] = []
    for tag in tags:
        if headers_by_release is not None and tag is not None:
            header_catalog = headers_by_release.get(tag)
        elif headers is not None:
            header_catalog = headers
        elif include_dir:
            header_catalog = load_tensorrt_headers(include_dir=include_dir, fetch=False)
        elif tag is not None:
            header_catalog = load_tensorrt_headers(ref=tag, fetch=True)
        elif fetch_headers:
            header_catalog = load_tensorrt_headers(fetch=True)
        else:
            header_catalog = None

        available = header_catalog.data_types if header_catalog is not None else None
        onnx_operators: Dict[str, object] = {}
        doc: Dict[str, object] = {}
        if operators_md:
            text = Path(operators_md).read_text(encoding="utf-8")
            parsed = parse_operators_md(text, source=operators_md)
            doc = {
                "tensorrt_version": parsed["tensorrt_version"],
                "opset_min": parsed["opset_min"],
                "opset_max": parsed["opset_max"],
                "source": parsed["source"],
            }
            for op in parsed["ops"]:
                onnx_operators[op.name] = {
                    "supported": op.supported,
                    "types": list(op.types),
                    "nvinfer_datatypes": map_onnx_dtypes_to_nvinfer(
                        op.types, available=available
                    ),
                    "restrictions": op.restrictions,
                }
        fingerprint = catalog_fingerprint(onnx_operators, header_catalog)
        if fingerprint not in groups:
            groups[fingerprint] = {
                "operators_md": doc,
                "nvinfer": None if header_catalog is None else header_catalog.as_dict(),
                "onnx_operators": onnx_operators,
                "releases": [],
            }
            order.append(fingerprint)
        if tag is not None:
            groups[fingerprint]["releases"].append(tag)

    databases = []
    for fingerprint in order:
        group = groups[fingerprint]
        supported = sorted(set(group["releases"]))
        if not supported and group["operators_md"].get("tensorrt_version"):
            version = group["operators_md"]["tensorrt_version"]
            if version and version != "unknown":
                supported = [normalize_release_tag(version)]
        payload = {
            "supported_releases": supported,
            "operators_md": group["operators_md"],
            "nvinfer": group["nvinfer"],
            "onnx_operators": group["onnx_operators"],
        }
        databases.append(payload)

    if len(databases) == 1:
        return databases[0]
    return {"databases": databases}


def write_catalog(catalog: Dict[str, object], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return output
