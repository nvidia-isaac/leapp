#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Parse TensorRT public headers for DataType, LayerType, and strongly-typed flags.

ONNX operator Y/N coverage still comes from onnx-tensorrt ``operators.md``.
Headers add nvinfer1 enumerations so catalogs can annotate ONNX dtypes with
``kFLOAT`` / ``kHALF`` / … and record ``NetworkDefinitionCreationFlag::kSTRONGLY_TYPED``.

Default sources (override with a local TensorRT ``include/``)::

    https://github.com/NVIDIA/TensorRT/tree/main/include
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_TENSORRT_REPO = "NVIDIA/TensorRT"
DEFAULT_HEADER_REF = "main"
DEFAULT_HEADER_BASE = (
    f"https://raw.githubusercontent.com/{DEFAULT_TENSORRT_REPO}/{DEFAULT_HEADER_REF}/include"
)
DEFAULT_HEADER_FILES = (
    "NvInferRuntimeBase.h",
    "NvInfer.h",
    "NvInferRuntime.h",
)

# operators.md tokens → nvinfer1::DataType enumerators in the headers.
ONNX_TYPE_TO_NVINFER: Dict[str, str] = {
    "FP32": "kFLOAT",
    "FLOAT32": "kFLOAT",
    "FLOAT": "kFLOAT",
    "FP16": "kHALF",
    "FLOAT16": "kHALF",
    "BF16": "kBF16",
    "BFLOAT16": "kBF16",
    "FP8": "kFP8",
    "FP4": "kFP4",
    "FP64": "kDOUBLE",
    "DOUBLE": "kDOUBLE",
    "INT8": "kINT8",
    "INT4": "kINT4",
    "INT32": "kINT32",
    "INT64": "kINT64",
    "UINT8": "kUINT8",
    "BOOL": "kBOOL",
    "TF32": "kFLOAT",  # TF32 is a compute mode on kFLOAT, not a DataType
}

STRONGLY_TYPED_FLAG = "NetworkDefinitionCreationFlag::kSTRONGLY_TYPED"
STRONGLY_TYPED_NOTE = (
    "NetworkDefinitionCreationFlag::kSTRONGLY_TYPED (TensorRT 11: always on; "
    "the flag is retained for API compatibility). Tensor DataType is "
    "authoritative; TensorRT will not implicitly cast. Listed "
    "nvinfer_datatypes must match the ONNX/TRT tensor types."
)

_ENUM_NAMES = (
    "DataType",
    "LayerType",
    "BuilderFlag",
    "NetworkDefinitionCreationFlag",
)
_ENUM_BODY = re.compile(
    r"enum\s+class\s+("
    + "|".join(_ENUM_NAMES)
    + r")\b[^{]*\{(.*?)\};",
    re.DOTALL,
)
_ENUM_MEMBER = re.compile(r"(?:^|[,{])\s*k([A-Z][A-Z0-9_]*)\b")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_SKIP_MEMBERS = {"kMAX", "kVALUE"}


@dataclass
class TensorRTHeaderCatalog:
    """Enumerations extracted from TensorRT headers."""

    source: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    layer_types: List[str] = field(default_factory=list)
    builder_flags: List[str] = field(default_factory=list)
    network_definition_creation_flags: List[str] = field(default_factory=list)

    @property
    def strongly_typed(self) -> bool:
        return "kSTRONGLY_TYPED" in self.network_definition_creation_flags or (
            "kSTRONGLY_TYPED" in self.builder_flags
        )

    @property
    def layer_type_names(self) -> List[str]:
        """LayerType enumerators without the ``k`` prefix (CONVOLUTION, …)."""
        names: List[str] = []
        for member in self.layer_types:
            if member in _SKIP_MEMBERS:
                continue
            names.append(member[1:] if member.startswith("k") else member)
        return names

    def map_onnx_dtypes(self, onnx_types: Sequence[str]) -> List[str]:
        available = self.data_types or None
        return map_onnx_dtypes_to_nvinfer(onnx_types, available=available)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": list(self.source),
            "data_types": list(self.data_types),
            "layer_types": list(self.layer_types),
            "layer_type_names": self.layer_type_names,
            "builder_flags": list(self.builder_flags),
            "network_definition_creation_flags": list(
                self.network_definition_creation_flags
            ),
            "strongly_typed": self.strongly_typed,
            "strongly_typed_flag": STRONGLY_TYPED_FLAG if self.strongly_typed else None,
            "strongly_typed_note": STRONGLY_TYPED_NOTE if self.strongly_typed else None,
            "onnx_type_to_nvinfer": dict(ONNX_TYPE_TO_NVINFER),
        }


def map_onnx_dtypes_to_nvinfer(
    onnx_types: Sequence[str],
    *,
    available: Optional[Sequence[str]] = None,
) -> List[str]:
    """Map operators.md type tokens to nvinfer1::DataType enumerators."""
    allowed = set(available) if available else None
    mapped: List[str] = []
    seen = set()
    for token in onnx_types:
        enumerator = ONNX_TYPE_TO_NVINFER.get(token.upper().replace(" ", ""))
        if enumerator is None:
            continue
        if allowed is not None and enumerator not in allowed:
            continue
        if enumerator not in seen:
            seen.add(enumerator)
            mapped.append(enumerator)
    return mapped


def _strip_comments(text: str) -> str:
    text = _COMMENT_BLOCK.sub("", text)
    return _LINE_COMMENT.sub("", text)


def _unique_members(body: str) -> List[str]:
    members = [f"k{name}" for name in _ENUM_MEMBER.findall(body)]
    unique: List[str] = []
    seen = set()
    for member in members:
        if member in _SKIP_MEMBERS or member in seen:
            continue
        seen.add(member)
        unique.append(member)
    return unique


def parse_tensorrt_headers(
    texts: Iterable[str],
    *,
    sources: Optional[Sequence[str]] = None,
) -> TensorRTHeaderCatalog:
    """Parse concatenated or multiple NvInfer*.h texts."""
    catalog = TensorRTHeaderCatalog(source=list(sources or []))
    cleaned = _strip_comments("\n".join(texts))
    found = {name: [] for name in _ENUM_NAMES}
    for match in _ENUM_BODY.finditer(cleaned):
        found[match.group(1)] = _unique_members(match.group(2))
    catalog.data_types = found["DataType"]
    catalog.layer_types = found["LayerType"]
    catalog.builder_flags = found["BuilderFlag"]
    catalog.network_definition_creation_flags = found["NetworkDefinitionCreationFlag"]
    return catalog


def _read_url(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def default_include_dir() -> Optional[str]:
    env = os.environ.get("TENSORRT_INCLUDE_DIR") or os.environ.get("TRT_INCLUDE_DIR")
    if env and Path(env).is_dir():
        return env
    return None


def normalize_release_tag(tag: str) -> str:
    """Normalize a TensorRT GitHub release ref (``11.2`` → ``v11.2``)."""
    cleaned = str(tag).strip()
    if not cleaned:
        raise ValueError("Empty TensorRT release tag")
    if cleaned in {"main", "master", "HEAD"} or re.fullmatch(r"[0-9a-fA-F]{7,40}", cleaned):
        return cleaned
    if cleaned[0] in "vV" and len(cleaned) > 1 and cleaned[1].isdigit():
        return "v" + cleaned[1:]
    if cleaned[0].isdigit():
        return "v" + cleaned
    return cleaned


def header_base_url_for_ref(
    ref: str = DEFAULT_HEADER_REF,
    *,
    repo: str = DEFAULT_TENSORRT_REPO,
) -> str:
    """Raw GitHub include/ URL for a TensorRT tag, branch, or commit."""
    return f"https://raw.githubusercontent.com/{repo}/{normalize_release_tag(ref)}/include"


def load_tensorrt_headers(
    *,
    include_dir: Optional[str] = None,
    header_files: Sequence[str] = DEFAULT_HEADER_FILES,
    header_base_url: Optional[str] = None,
    ref: str = DEFAULT_HEADER_REF,
    fetch: bool = True,
) -> TensorRTHeaderCatalog:
    """Load headers from a local TensorRT include dir or a GitHub ref."""
    texts: List[str] = []
    sources: List[str] = []
    include_dir = include_dir or (None if fetch else default_include_dir())
    base = header_base_url or header_base_url_for_ref(ref)
    if include_dir:
        root = Path(include_dir)
        for name in header_files:
            path = root / name
            if not path.is_file():
                nested = root / "NvInfer" / name
                path = nested if nested.is_file() else path
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
                sources.append(str(path))
        if not texts:
            raise FileNotFoundError(
                f"No TensorRT headers {list(header_files)} under {include_dir}"
            )
    elif fetch:
        for name in header_files:
            url = f"{base.rstrip('/')}/{name}"
            try:
                texts.append(_read_url(url))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            sources.append(url)
        if not texts:
            raise FileNotFoundError(
                f"No TensorRT headers {list(header_files)} at {base}"
            )
    else:
        raise ValueError("Pass include_dir or fetch=True")
    catalog = parse_tensorrt_headers(texts, sources=sources)
    catalog.source = list(catalog.source)
    return catalog
