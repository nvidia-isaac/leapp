#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Generate TensorRT supported-layer catalogs for CUDA 13.2 platforms.

ONNX operator coverage comes from the public ONNX-TensorRT matrix (same parser
across platforms). Hardware overlays then drop dtypes and native layers that
NVIDIA's support matrix / DLA docs do not list for that SKU.

Native ``DataType`` / ``LayerType`` / ``NetworkDefinitionCreationFlag`` values
are parsed from TensorRT headers (local ``include/`` or GitHub) and joined onto
each ONNX row as ``nvinfer_datatypes``. Headers do not replace operators.md.

This is not a live engine-build guarantee.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from isaac_deploy_trt.headers import (
    ONNX_TYPE_TO_NVINFER,
    STRONGLY_TYPED_FLAG,
    STRONGLY_TYPED_NOTE,
    TensorRTHeaderCatalog,
    load_tensorrt_headers,
    map_onnx_dtypes_to_nvinfer,
    normalize_release_tag,
)

DEFAULT_OPERATORS_MD_URL = (
    "https://raw.githubusercontent.com/onnx/onnx-tensorrt/main/docs/operators.md"
)

# CUDA 13.2 is the toolkit the user asked to target. TensorRT *packages* may
# still be labeled cuda-13.3 while remaining compatible with CUDA 13.x.
TARGET_CUDA = "13.2"

TYPE_ALIASES = {
    "FLOAT32": "FP32",
    "FLOAT": "FP32",
    "FLOAT16": "FP16",
    "BFLOAT16": "BF16",
    "DOUBLE": "FP64",
}

# TensorRT ILayerType names that exist in TRT 10.13–11.2. Live installs can
# replace this via probe_installed_tensorrt().
TENSORRT_LAYER_TYPES = [
    "ACTIVATION",
    "ASSERTION",
    "ATTENTION",
    "CAST",
    "CONCATENATION",
    "CONDITION",
    "CONSTANT",
    "CONVOLUTION",
    "CUMULATIVE",
    "DECONVOLUTION",
    "DEQUANTIZE",
    "DYNAMIC_QUANTIZE",
    "EINSUM",
    "ELEMENTWISE",
    "FILL",
    "GATHER",
    "GRID_SAMPLE",
    "IDENTITY",
    "LOOP",
    "LOOP_OUTPUT",
    "LRN",
    "MATRIX_MULTIPLY",
    "NMS",
    "NON_ZERO",
    "NORMALIZATION",
    "ONE_HOT",
    "PADDING",
    "PARAMETRIC_RELU",
    "PLUGIN",
    "PLUGIN_V2",
    "PLUGIN_V3",
    "POOLING",
    "QUANTIZE",
    "RAGGED_SOFTMAX",
    "RECURRENCE",
    "REDUCE",
    "RESIZE",
    "REVERSE_SEQUENCE",
    "SCALE",
    "SCATTER",
    "SELECT",
    "SHAPE",
    "SHUFFLE",
    "SLICE",
    "SOFTMAX",
    "SQUEEZE",
    "TOPK",
    "TRIP_LIMIT",
    "UNARY",
    "UNSQUEEZE",
]

# Orin DLA (TensorRT 10.x / JetPack). TensorRT 11.x does not support DLA.
# Source: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html
ORIN_DLA_LAYERS = {
    "ACTIVATION": "FP16, INT8; 2D; ReLU, Sigmoid, TanH, Clipped ReLU, Leaky ReLU",
    "CONCATENATION": "FP16, INT8; channel axis only",
    "CONVOLUTION": "FP16, INT8; 2D; kernel [1,32]; stride [1,8]; channels [1,8192]",
    "DECONVOLUTION": "FP16, INT8; 2D; no grouped/dilated; padding 0",
    "ELEMENTWISE": "FP16, INT8; 2D; Sum, Sub, Product, Max, Min, Div, Pow",
    "LRN": "FP16 (INT8 promoted to FP16); window 3/5/7/9; ACROSS_CHANNELS",
    "PARAMETRIC_RELU": "FP16, INT8; slope must be a build-time constant",
    "POOLING": "FP16, INT8; 2D; MAX/AVERAGE; window [1,8]; stride [1,16]",
    "REDUCE": "FP16, INT8; 4D; MAX; CHW axes",
    "RESIZE": "FP16, INT8; nearest [1,32], bilinear [1,4]",
    "SCALE": "FP16, INT8; Uniform, Per-Channel, ElementWise",
    "SHUFFLE": "FP16, INT8; 4D; no batch transpose",
    "SLICE": "FP16, INT8; 4D; static slicing",
    "SOFTMAX": "FP16, INT8; Orin DLA only; axis dim <= 1024 in optimized mode",
    "UNARY": "INT8; ABS, SIN, COS, ATAN",
}


@dataclass
class PlatformSpec:
    """CUDA 13.2 target overlay."""

    name: str
    display_name: str
    cuda: str
    tensorrt: str
    arch: str
    compute_capability: str
    gpu_precisions: List[str]
    dla: bool
    tensorrt_11: bool
    isaac_debian_dist: str = ""
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


# NVIDIA support-matrix facts for CUDA 13.2-class stacks (JetPack 7.x / TRT 10.13+
# / TRT 11.x CUDA 13 compatibility). Precision lists are hardware capabilities,
# not a promise every ONNX op builds in that dtype.
PLATFORMS: Dict[str, PlatformSpec] = {
    "x86": PlatformSpec(
        name="x86",
        display_name="Linux x86_64 (CUDA 13.2)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.13+ or 11.x (11.2.1 packages built against CUDA 13.3, compatible with CUDA 13.x)",
        arch="Ampere / Ada / Hopper / Blackwell dGPU",
        compute_capability="8.0–12.x (SKU-dependent)",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL", "FP64",
        ],
        dla=False,
        tensorrt_11=True,
        isaac_debian_dist="noble",
        notes=[
            "Isaac Debian dist: noble (Ubuntu 24.04 amd64).",
            "ONNX parser coverage matches onnx-tensorrt operators.md.",
            "FP8 requires Hopper or newer; FP4 requires Blackwell.",
            "BF16 is not available on every Ampere SKU (for example some GA10x).",
            "No DLA.",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
            "https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md",
        ],
    ),
    "orin": PlatformSpec(
        name="orin",
        display_name="Jetson Orin (CUDA 13.2 / JetPack 7.2)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.x from JetPack (TensorRT 11.2.1 does not support JetPack)",
        arch="Ampere GA10B iGPU",
        compute_capability="8.7",
        gpu_precisions=["FP32", "TF32", "FP16", "INT8", "INT32", "INT64", "UINT8", "BOOL"],
        dla=True,
        tensorrt_11=False,
        isaac_debian_dist="noble-jetpack",
        notes=[
            "Isaac Debian dist: noble-jetpack (JetPack 7.x / TensorRT 10.16).",
            "Stay on JetPack TensorRT 10.x for CUDA 13.2 Orin.",
            "No BF16, FP8, or FP4 on Orin.",
            "DLA is FP16/INT8 and only for the DLA layer subset; TensorRT 11.x has no DLA.",
            "Orin DLA Softmax is Orin-only (not Xavier).",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html",
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html",
        ],
    ),
    "thor": PlatformSpec(
        name="thor",
        display_name="Jetson AGX Thor (CUDA 13.x / JetPack 7.x)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.13–10.15.x from JetPack (not TensorRT 11.2.1)",
        arch="Blackwell (Thor GPU)",
        compute_capability="11.0",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL",
        ],
        dla=False,
        tensorrt_11=False,
        isaac_debian_dist="noble-jetpack",
        notes=[
            "Isaac Debian dist: noble-jetpack (same apt pocket as Orin).",
            "JetPack 7.x ships TensorRT 10.x; NVIDIA documents Thor as unsupported on TensorRT 11.2.1.",
            "Blackwell enables FP8/FP4; no Orin-style DLA catalog.",
            "Engines are not portable to Orin (different SM).",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html",
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
        ],
    ),
    "dgx": PlatformSpec(
        name="dgx",
        display_name="DGX Spark / DGX Blackwell (CUDA 13.2)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.15+ (Spark/GB10) or TensorRT 11.x on DGX B200-class x86",
        arch="Blackwell GB10 (Spark) or B200/GB200 (DGX)",
        compute_capability="10.0–12.x (SKU-dependent)",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL", "FP64",
        ],
        dla=False,
        tensorrt_11=True,
        isaac_debian_dist="noble-fastos",
        notes=[
            "Isaac Debian dist for Spark: noble-fastos. DGX B200/GB200 x86 uses noble, not FastOS.",
            "DGX Spark (GB10) is ARM and commonly TensorRT 10.15.x on CUDA 13.0–13.2.",
            "DGX B200/GB200 x86 can use TensorRT 11.x with CUDA 13.x.",
            "Use this catalog for Blackwell datacenter/desktop; use 'noble' / 'x86' for Hopper-only boxes.",
            "No DLA.",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.2.1.html",
        ],
    ),
    "noble": PlatformSpec(
        name="noble",
        display_name="Ubuntu 24.04 noble (amd64 / CUDA 13.2)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.13+ or 11.x (11.2.1 packages built against CUDA 13.3, compatible with CUDA 13.x)",
        arch="Ampere / Ada / Hopper / Blackwell dGPU",
        compute_capability="8.0–12.x (SKU-dependent)",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL", "FP64",
        ],
        dla=False,
        tensorrt_11=True,
        isaac_debian_dist="noble",
        notes=[
            "Isaac Debian dist noble: Ubuntu 24.04 amd64 (x86_64 dGPU).",
            "ONNX parser coverage matches onnx-tensorrt operators.md.",
            "FP8 requires Hopper or newer; FP4 requires Blackwell.",
            "BF16 is not available on every Ampere SKU (for example some GA10x).",
            "No DLA. Same overlay as the 'x86' SKU catalog.",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
            "https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md",
        ],
    ),
    "noble-fastos": PlatformSpec(
        name="noble-fastos",
        display_name="Ubuntu 24.04 noble-fastos (DGX Spark / GB10)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.15.x from FastOS (not TensorRT 11.x)",
        arch="Blackwell GB10 (Spark, ARM)",
        compute_capability="10.0",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL",
        ],
        dla=False,
        tensorrt_11=False,
        isaac_debian_dist="noble-fastos",
        notes=[
            "Isaac Debian dist noble-fastos: DGX Spark / FastOS arm64.",
            "CUDA toolkit on this pocket is often 13.0.x; TensorRT is 10.15.x.",
            "Not JetPack and not Ubuntu amd64 noble. Do not use this overlay for Orin/Thor.",
            "No DLA. Engines are not portable to Jetson or x86 dGPU.",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html",
        ],
    ),
    "noble-jetpack": PlatformSpec(
        name="noble-jetpack",
        display_name="Ubuntu 24.04 noble-jetpack (Jetson / JetPack 7)",
        cuda=TARGET_CUDA,
        tensorrt="TensorRT 10.16.x from JetPack 7.2 (not TensorRT 11.x)",
        arch="Orin GA10B iGPU and Thor Blackwell",
        compute_capability="8.7 (Orin) / 11.0 (Thor)",
        gpu_precisions=[
            "FP32", "TF32", "FP16", "BF16", "FP8", "FP4", "INT8", "INT4",
            "INT32", "INT64", "UINT8", "BOOL",
        ],
        dla=True,
        tensorrt_11=False,
        isaac_debian_dist="noble-jetpack",
        notes=[
            "Isaac Debian dist noble-jetpack: JetPack 7.x arm64 (Orin and Thor).",
            "Stay on JetPack TensorRT 10.16; TensorRT 11.x is not a JetPack path.",
            "Orin: no BF16/FP8/FP4; DLA FP16/INT8 for the DLA layer subset.",
            "Thor: Blackwell FP8/FP4; no DLA. Engines are not portable between Orin and Thor.",
            "This overlay keeps the union of GPU dtypes; drop BF16/FP8/FP4 yourself on Orin.",
            "DLA rows apply to Orin only.",
        ],
        sources=[
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html",
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html",
        ],
    ),
}

# Isaac ROS / Isaac Debian pocket names are the default generate targets.
DEFAULT_PLATFORMS = ("noble", "noble-fastos", "noble-jetpack")
PLATFORM_ALIASES = {
    "amd64": "noble",
    "x86_64": "noble",
    "linux-x86": "noble",
    "spark": "noble-fastos",
    "fastos": "noble-fastos",
    "dgx-spark": "noble-fastos",
    "jetpack": "noble-jetpack",
    "jetson": "noble-jetpack",
    "jp7": "noble-jetpack",
}


def resolve_platform_name(name: str) -> str:
    """Map aliases (amd64, spark, jetson) onto catalog keys."""
    cleaned = name.strip().lower().replace("_", "-")
    return PLATFORM_ALIASES.get(cleaned, cleaned)


@dataclass
class OnnxOpRow:
    name: str
    supported: bool
    types: List[str]
    restrictions: str

    def as_leapp_dict(
        self,
        headers: Optional[TensorRTHeaderCatalog] = None,
    ) -> Dict[str, object]:
        available = headers.data_types if headers is not None else None
        return {
            "supported": self.supported,
            "types": list(self.types),
            "nvinfer_datatypes": map_onnx_dtypes_to_nvinfer(
                self.types, available=available
            ),
            "restrictions": self.restrictions,
        }


@dataclass
class OperatorsDoc:
    tensorrt_version: str
    opset_min: int
    opset_max: int
    source: str
    ops: List[OnnxOpRow]


def normalize_dtype(token: str) -> str:
    cleaned = token.strip().upper().replace(" ", "")
    return TYPE_ALIASES.get(cleaned, cleaned)


KNOWN_DTYPES = {
    "FP32", "FP16", "BF16", "FP8", "FP4", "FP64",
    "INT8", "INT4", "INT32", "INT64", "UINT8", "BOOL", "TF32",
}


def parse_types(cell: str) -> List[str]:
    if not cell or not cell.strip():
        return []
    parts = [normalize_dtype(part) for part in cell.split(",")]
    return [part for part in parts if part in KNOWN_DTYPES]


def parse_operators_md(text: str, *, source: str = "operators.md") -> OperatorsDoc:
    """Parse the ONNX-TensorRT operators.md support table."""
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
        types = parse_types(cells[2] if len(cells) > 2 else "")
        restrictions = cells[3] if len(cells) > 3 else ""
        types_cell = cells[2].strip() if len(cells) > 2 else ""
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
    return OperatorsDoc(
        tensorrt_version=version,
        opset_min=opset_min,
        opset_max=opset_max,
        source=source,
        ops=ops,
    )


def load_operators_md(
    *,
    path: Optional[str] = None,
    url: str = DEFAULT_OPERATORS_MD_URL,
) -> OperatorsDoc:
    if path:
        text = Path(path).read_text(encoding="utf-8")
        return parse_operators_md(text, source=path)
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return parse_operators_md(text, source=url)


def _filter_types(types: Sequence[str], allowed: Sequence[str]) -> List[str]:
    allowed_set = set(allowed)
    # Integer/bool network types stay if the parser listed them, even when the
    # GPU precision table is focused on floating/quantized compute.
    keep_always = {"INT32", "INT64", "UINT8", "BOOL"}
    out: List[str] = []
    for dtype in types:
        if dtype in allowed_set or dtype in keep_always:
            out.append(dtype)
    return out


def apply_platform(doc: OperatorsDoc, platform: PlatformSpec) -> List[OnnxOpRow]:
    rows: List[OnnxOpRow] = []
    extra = []
    if not platform.tensorrt_11 and doc.tensorrt_version.startswith("11"):
        extra.append(
            "Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. "
            "Treat newly added 11.x-only ops as unverified."
        )
    for op in doc.ops:
        types = _filter_types(op.types, platform.gpu_precisions)
        restrictions = op.restrictions
        if extra:
            note = extra[0]
            restrictions = f"{restrictions}; {note}".strip("; ") if restrictions else note
        supported = op.supported
        if supported and op.types and not types:
            supported = False
            drop_note = (
                f"All listed dtypes ({', '.join(op.types)}) are unavailable on "
                f"{platform.display_name}"
            )
            restrictions = f"{restrictions}; {drop_note}".strip("; ")
        rows.append(
            OnnxOpRow(
                name=op.name,
                supported=supported,
                types=types,
                restrictions=restrictions,
            )
        )
    return rows


# Native layers that NVIDIA documents as TensorRT 11.x (not JetPack 10.x).
TRT11_ONLY_LAYERS = {
    "ATTENTION",
    "ATTENTION_INPUT",
    "ATTENTION_OUTPUT",
    "DYNAMIC_QUANTIZE",
    "ROTARY_EMBEDDING",
    "KVCACHE_UPDATE",
    "MOE",
    "DIST_COLLECTIVE",
}


def layer_names_for_catalog(
    headers: Optional[TensorRTHeaderCatalog] = None,
) -> List[str]:
    names = list(TENSORRT_LAYER_TYPES)
    if headers is None:
        return names
    seen = set(names)
    for name in headers.layer_type_names:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def native_layers_for_platform(
    platform: PlatformSpec,
    *,
    headers: Optional[TensorRTHeaderCatalog] = None,
) -> Dict[str, Dict[str, object]]:
    layers: Dict[str, Dict[str, object]] = {}
    trt11_only = TRT11_ONLY_LAYERS
    for name in layer_names_for_catalog(headers):
        gpu = True
        notes = []
        if name in trt11_only and not platform.tensorrt_11:
            gpu = False
            notes.append("Documented as TensorRT 11.x native layer; not in JetPack 10.x")
        dla = platform.dla and name in ORIN_DLA_LAYERS
        if dla:
            notes.append(ORIN_DLA_LAYERS[name])
        layers[name] = {
            "gpu": gpu,
            "dla": dla,
            "notes": " ".join(notes),
        }
    return layers


def probe_installed_tensorrt() -> Optional[Dict[str, object]]:
    try:
        import tensorrt as trt  # type: ignore
    except Exception:
        return None
    members = []
    layer_type = getattr(trt, "LayerType", None)
    if layer_type is not None and hasattr(layer_type, "__members__"):
        members = sorted(layer_type.__members__)
    return {
        "version": getattr(trt, "__version__", "unknown"),
        "layer_types": members,
    }


def build_platform_catalog(
    doc: OperatorsDoc,
    platform: PlatformSpec,
    *,
    headers: Optional[TensorRTHeaderCatalog] = None,
    supported_releases: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    ops = apply_platform(doc, platform)
    supported = [op.name for op in ops if op.supported]
    unsupported = [op.name for op in ops if not op.supported]
    nvinfer = (
        headers.as_dict()
        if headers is not None
        else {
            "source": [],
            "data_types": [],
            "layer_types": [],
            "layer_type_names": list(TENSORRT_LAYER_TYPES),
            "builder_flags": [],
            "network_definition_creation_flags": [],
            "strongly_typed": None,
            "strongly_typed_flag": STRONGLY_TYPED_FLAG,
            "strongly_typed_note": STRONGLY_TYPED_NOTE,
            "onnx_type_to_nvinfer": dict(ONNX_TYPE_TO_NVINFER),
        }
    )
    sources = list(platform.sources)
    if headers is not None:
        sources.extend(headers.source)
    return {
        "cuda": platform.cuda,
        "platform": platform.name,
        "isaac_debian_dist": platform.isaac_debian_dist,
        "display_name": platform.display_name,
        "tensorrt": platform.tensorrt,
        "arch": platform.arch,
        "compute_capability": platform.compute_capability,
        "parser_tensorrt_version": doc.tensorrt_version,
        "opset": {"min": doc.opset_min, "max": doc.opset_max},
        "source": doc.source,
        "gpu_precisions": platform.gpu_precisions,
        "dla": platform.dla,
        "tensorrt_11": platform.tensorrt_11,
        "notes": platform.notes,
        "sources": sources,
        "supported_releases": list(supported_releases or []),
        "nvinfer": nvinfer,
        "onnx_supported": supported,
        "onnx_unsupported": unsupported,
        "onnx_operators": {op.name: op.as_leapp_dict(headers) for op in ops},
        "tensorrt_layers": native_layers_for_platform(platform, headers=headers),
        "installed_tensorrt": probe_installed_tensorrt(),
    }


PYTHON_CATALOG_LOOKUP = '''
def _normalize_tensorrt_version(version: str) -> str:
    parts = [part.lstrip("vV") for part in str(version).strip().split(".") if part]
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    if parts:
        return parts[0]
    raise ValueError(f"Invalid TensorRT version: {version!r}")


def _normalize_release_tag(tag: str) -> str:
    cleaned = str(tag).strip()
    if cleaned[0] in "vV" and len(cleaned) > 1 and cleaned[1].isdigit():
        return "v" + cleaned[1:]
    if cleaned and cleaned[0].isdigit():
        return "v" + cleaned
    return cleaned


def _cuda_matches(entry: str, requested: str) -> bool:
    want = requested.strip().lower()
    have = entry.strip().lower()
    if have.endswith(".x"):
        return want == have[:-2] or want.startswith(have[:-1])
    return want == have or want.startswith(have + ".")


def _release_matches(catalog: TensorRTOnnxCatalogInfo, tensorrt: str) -> bool:
    key = _normalize_tensorrt_version(tensorrt)
    tag = _normalize_release_tag(tensorrt)
    if catalog.get("tensorrt") and _normalize_tensorrt_version(catalog["tensorrt"]) == key:
        return True
    for release in catalog.get("supported_releases") or []:
        if release == tensorrt or release == tag:
            return True
        if _normalize_tensorrt_version(release) == key:
            return True
    return False


def list_onnx_ops_catalogs() -> List[str]:
    return sorted(TENSORRT_ONNX_CATALOGS)


def onnx_ops_catalog(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> TensorRTOnnxCatalogInfo:
    if tensorrt:
        key = _normalize_tensorrt_version(tensorrt)
        if key in TENSORRT_ONNX_CATALOGS:
            catalog = TENSORRT_ONNX_CATALOGS[key]
        else:
            catalog = next(
                (
                    entry
                    for entry in TENSORRT_ONNX_CATALOGS.values()
                    if _release_matches(entry, tensorrt)
                ),
                None,
            )
        if catalog is None:
            raise KeyError(
                f"No ONNX operator catalog for TensorRT {tensorrt!r} "
                f"(have {list_onnx_ops_catalogs()})"
            )
        if cuda and not any(_cuda_matches(tag, cuda) for tag in catalog["cuda"]):
            raise KeyError(
                f"TensorRT {key} catalog does not list CUDA {cuda!r} "
                f"(have {catalog['cuda']})"
            )
        return catalog
    if cuda:
        matches = [
            TENSORRT_ONNX_CATALOGS[key]
            for key in sorted(TENSORRT_ONNX_CATALOGS, reverse=True)
            if any(_cuda_matches(tag, cuda) for tag in TENSORRT_ONNX_CATALOGS[key]["cuda"])
        ]
        if not matches:
            raise KeyError(
                f"No ONNX operator catalog lists CUDA {cuda!r} "
                f"(have {list_onnx_ops_catalogs()})"
            )
        return matches[0]
    return TENSORRT_ONNX_CATALOGS[TENSORRT_ONNX_DEFAULT_VERSION]


def onnx_ops_for_version(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> Dict[str, TensorRTOnnxOpInfo]:
    return onnx_ops_catalog(tensorrt, cuda=cuda)["ops"]


def supported_onnx_ops(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> set[str]:
    ops = onnx_ops_for_version(tensorrt, cuda=cuda)
    return {name for name, info in ops.items() if info["supported"]}


def unsupported_onnx_ops(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> set[str]:
    ops = onnx_ops_for_version(tensorrt, cuda=cuda)
    return {name for name, info in ops.items() if not info["supported"]}
'''.lstrip()


def render_python_catalog(
    doc: OperatorsDoc,
    *,
    supported_releases: Optional[Sequence[str]] = None,
) -> str:
    """Emit a tensorrt_onnx_ops.py-style dict from the unfiltered parser table."""
    releases = list(supported_releases or infer_supported_releases(doc, ()))
    lines = [
        "#",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "",
        '"""TensorRT ONNX operator support catalog.',
        "",
        f"Generated from TensorRT {doc.tensorrt_version}, ONNX opset "
        f"{doc.opset_min}-{doc.opset_max}:",
        f"{doc.source}",
        "",
        "Regenerate with:",
        "  isaac_deploy_trt generate --write-python \\",
        "      path/to/tensorrt_onnx_ops.py",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Dict, List, Optional, TypedDict",
        "",
        "",
        "class TensorRTOnnxOpInfo(TypedDict):",
        "    supported: bool",
        "    types: List[str]",
        "    nvinfer_datatypes: List[str]",
        "    restrictions: str",
        "",
        "",
        "class TensorRTOnnxCatalogInfo(TypedDict):",
        "    tensorrt: str",
        "    cuda: List[str]",
        "    opset_min: int",
        "    opset_max: int",
        "    source: str",
        "    supported_releases: List[str]",
        "    ops: Dict[str, TensorRTOnnxOpInfo]",
        "",
        "",
        f'TENSORRT_ONNX_DEFAULT_VERSION = "{doc.tensorrt_version}"',
        f"TENSORRT_ONNX_OPSET_MIN = {doc.opset_min}",
        f"TENSORRT_ONNX_OPSET_MAX = {doc.opset_max}",
        "",
        "TENSORRT_ONNX_OPS: Dict[str, TensorRTOnnxOpInfo] = {",
        "",
    ]
    for op in doc.ops:
        types = ", ".join(repr(dtype) for dtype in op.types)
        restrictions = op.restrictions.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f'    "{op.name}": {{')
        lines.append(f'        "supported": {op.supported},')
        lines.append(f"        \"types\": [{types}],")
        nvinfer_types = ", ".join(
            repr(dtype) for dtype in map_onnx_dtypes_to_nvinfer(op.types)
        )
        lines.append(f"        \"nvinfer_datatypes\": [{nvinfer_types}],")
        lines.append(f"        \"restrictions\": '{restrictions}',")
        lines.append("    },")
    lines.extend(
        [
            "}",
            "",
            "",
            "TENSORRT_ONNX_CATALOGS: Dict[str, TensorRTOnnxCatalogInfo] = {",
            "    TENSORRT_ONNX_DEFAULT_VERSION: {",
            '        "tensorrt": TENSORRT_ONNX_DEFAULT_VERSION,',
            '        "cuda": ["12.x", "13.x"],',
            '        "opset_min": TENSORRT_ONNX_OPSET_MIN,',
            '        "opset_max": TENSORRT_ONNX_OPSET_MAX,',
            f'        "source": {doc.source!r},',
            f'        "supported_releases": {releases!r},',
            '        "ops": TENSORRT_ONNX_OPS,',
            "    },",
            "}",
            "",
            "",
            PYTHON_CATALOG_LOOKUP,
        ]
    )
    return "\n".join(lines)


def render_markdown(catalog: Dict[str, object]) -> str:
    ops = catalog["onnx_operators"]
    lines = [
        f"# TensorRT layers — {catalog['display_name']}",
        "",
        f"- CUDA: `{catalog['cuda']}`",
        f"- TensorRT: {catalog['tensorrt']}",
        f"- Arch: {catalog['arch']} (CC {catalog['compute_capability']})",
        f"- Parser table: TensorRT {catalog['parser_tensorrt_version']}, "
        f"opset {catalog['opset']['min']}-{catalog['opset']['max']}",
        f"- GPU precisions: {', '.join(catalog['gpu_precisions'])}",
        f"- DLA: {catalog['dla']}",
        f"- Supported releases: {', '.join(catalog.get('supported_releases') or []) or '(unspecified)'}",
        "",
        "## Notes",
        "",
    ]
    for note in catalog["notes"]:
        lines.append(f"- {note}")
    nvinfer_info = catalog.get("nvinfer") or {}
    header_sources = nvinfer_info.get("source") or []
    if header_sources:
        lines.extend(["", "## TensorRT headers", ""])
        for src in header_sources:
            lines.append(f"- `{src}`")
        if nvinfer_info.get("strongly_typed_note"):
            lines.append(f"- {nvinfer_info['strongly_typed_note']}")
    lines.extend(["", "## ONNX operators (supported)", "", "| Operator | Types | nvinfer DataType | Restrictions |", "|---|---|---|---|"])
    for name, info in ops.items():
        if not info["supported"]:
            continue
        types = ", ".join(info["types"])
        nvinfer = ", ".join(info.get("nvinfer_datatypes") or [])
        restrictions = str(info["restrictions"]).replace("|", "\\|")
        lines.append(f"| {name} | {types} | {nvinfer} | {restrictions} |")
    lines.extend(["", "## ONNX operators (unsupported)", "", "| Operator | Restrictions |", "|---|---|"])
    for name, info in ops.items():
        if info["supported"]:
            continue
        restrictions = str(info["restrictions"]).replace("|", "\\|")
        lines.append(f"| {name} | {restrictions} |")
    lines.extend(["", "## Native TensorRT layers", "", "| Layer | GPU | DLA | Notes |", "|---|---|---|---|"])
    for name, info in catalog["tensorrt_layers"].items():
        notes = str(info["notes"]).replace("|", "\\|")
        lines.append(f"| {name} | {info['gpu']} | {info['dla']} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def _release_sort_key(tag: str) -> tuple:
    body = normalize_release_tag(tag).lstrip("vV")
    parts = []
    for piece in re.split(r"[.-]", body):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


def sort_release_tags(tags: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for tag in tags:
        normalized = normalize_release_tag(tag)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return sorted(unique, key=_release_sort_key)


def infer_supported_releases(
    doc: OperatorsDoc,
    releases: Sequence[str] = (),
) -> List[str]:
    if releases:
        return sort_release_tags(releases)
    if doc.tensorrt_version and doc.tensorrt_version != "unknown":
        return [normalize_release_tag(doc.tensorrt_version)]
    return []


def catalog_fingerprint(
    doc: OperatorsDoc,
    headers: Optional[TensorRTHeaderCatalog],
) -> tuple:
    ops = tuple(
        (op.name, op.supported, tuple(op.types), op.restrictions) for op in doc.ops
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


def database_id(releases: Sequence[str]) -> str:
    ordered = sort_release_tags(releases)
    return ordered[-1] if ordered else "untagged"


def write_catalogs(
    output_dir: Path,
    catalogs: Iterable[Dict[str, object]],
    *,
    formats: Sequence[str] = ("json", "md"),
    extra_index: Optional[Dict[str, object]] = None,
) -> List[Path]:
    catalog_list = list(catalogs)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for catalog in catalog_list:
        stem = f"tensorrt_layers_cuda{catalog['cuda'].replace('.', '')}_{catalog['platform']}"
        if "json" in formats:
            path = output_dir / f"{stem}.json"
            path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
            written.append(path)
        if "md" in formats:
            path = output_dir / f"{stem}.md"
            path.write_text(render_markdown(catalog), encoding="utf-8")
            written.append(path)
    index: Dict[str, object] = {
        "cuda": TARGET_CUDA,
        "platforms": [catalog["platform"] for catalog in catalog_list],
        "supported_releases": list(catalog_list[0].get("supported_releases") or [])
        if catalog_list
        else [],
        "files": [str(path.name) for path in written],
    }
    if extra_index:
        index.update(extra_index)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    written.append(index_path)
    return written


def _load_headers_for_release(
    *,
    tag: Optional[str],
    include_dir: Optional[str],
    fetch_headers: bool,
    headers: Optional[TensorRTHeaderCatalog],
    headers_by_release: Optional[Dict[str, TensorRTHeaderCatalog]],
) -> Optional[TensorRTHeaderCatalog]:
    if headers_by_release is not None:
        if tag is None:
            return headers
        return headers_by_release.get(tag) or headers_by_release.get(
            normalize_release_tag(tag)
        )
    if headers is not None:
        return headers
    if include_dir:
        return load_tensorrt_headers(include_dir=include_dir, fetch=False)
    if tag:
        return load_tensorrt_headers(ref=tag, fetch=True)
    if fetch_headers:
        return load_tensorrt_headers(fetch=True)
    return None


def generate(
    *,
    platforms: Sequence[str] = DEFAULT_PLATFORMS,
    operators_md: Optional[str] = None,
    operators_url: str = DEFAULT_OPERATORS_MD_URL,
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("json", "md"),
    write_python: Optional[Path] = None,
    include_dir: Optional[str] = None,
    fetch_headers: bool = False,
    headers: Optional[TensorRTHeaderCatalog] = None,
    releases: Sequence[str] = (),
    headers_by_release: Optional[Dict[str, TensorRTHeaderCatalog]] = None,
) -> Dict[str, object]:
    platforms = [resolve_platform_name(name) for name in platforms]
    unknown = [name for name in platforms if name not in PLATFORMS]
    if unknown:
        raise ValueError(f"Unknown platforms {unknown}; choose from {sorted(PLATFORMS)}")
    requested = [normalize_release_tag(tag) for tag in releases] if releases else [None]
    groups: Dict[tuple, Dict[str, object]] = {}
    group_order: List[tuple] = []
    for tag in requested:
        doc = load_operators_md(path=operators_md, url=operators_url)
        header_catalog = _load_headers_for_release(
            tag=tag,
            include_dir=include_dir,
            fetch_headers=fetch_headers,
            headers=headers,
            headers_by_release=headers_by_release,
        )
        fingerprint = catalog_fingerprint(doc, header_catalog)
        if fingerprint not in groups:
            groups[fingerprint] = {
                "doc": doc,
                "headers": header_catalog,
                "releases": [],
            }
            group_order.append(fingerprint)
        if tag is not None:
            groups[fingerprint]["releases"].append(tag)

    databases = []
    for fingerprint in group_order:
        group = groups[fingerprint]
        doc = group["doc"]
        header_catalog = group["headers"]
        supported = infer_supported_releases(doc, group["releases"])
        db_id = database_id(supported) if supported else "untagged"
        platform_catalogs = [
            build_platform_catalog(
                doc,
                PLATFORMS[name],
                headers=header_catalog,
                supported_releases=supported,
            )
            for name in platforms
        ]
        databases.append(
            {
                "id": db_id,
                "supported_releases": supported,
                "operators_doc": doc,
                "doc": {
                    "tensorrt_version": doc.tensorrt_version,
                    "opset_min": doc.opset_min,
                    "opset_max": doc.opset_max,
                    "source": doc.source,
                    "op_count": len(doc.ops),
                    "headers": None
                    if header_catalog is None
                    else header_catalog.source,
                },
                "catalogs": platform_catalogs,
            }
        )

    written: List[Path] = []
    if output_dir is not None:
        output_dir = Path(output_dir)
        if len(databases) <= 1:
            catalogs = databases[0]["catalogs"] if databases else []
            extra = {
                "databases": [
                    {
                        "id": databases[0]["id"],
                        "supported_releases": databases[0]["supported_releases"],
                    }
                ]
                if databases
                else [],
                "release_to_database": {
                    tag: databases[0]["id"]
                    for tag in databases[0]["supported_releases"]
                }
                if databases
                else {},
            }
            written = write_catalogs(
                output_dir, catalogs, formats=formats, extra_index=extra
            )
        else:
            release_to_db = {}
            db_index = []
            for database in databases:
                subdir = output_dir / database["id"]
                files = write_catalogs(subdir, database["catalogs"], formats=formats)
                written.extend(files)
                db_index.append(
                    {
                        "id": database["id"],
                        "supported_releases": database["supported_releases"],
                        "dir": database["id"],
                    }
                )
                for tag in database["supported_releases"]:
                    release_to_db[tag] = database["id"]
            index_path = output_dir / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "cuda": TARGET_CUDA,
                        "databases": db_index,
                        "release_to_database": release_to_db,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            written.append(index_path)

    python_path = None
    if write_python is not None and databases:
        canonical = databases[-1]
        write_python.parent.mkdir(parents=True, exist_ok=True)
        write_python.write_text(
            render_python_catalog(
                canonical["operators_doc"],
                supported_releases=canonical["supported_releases"],
            ),
            encoding="utf-8",
        )
        python_path = str(write_python)

    primary = databases[-1] if databases else None
    return {
        "doc": primary["doc"] if primary else {},
        "catalogs": primary["catalogs"] if primary else [],
        "databases": databases,
        "written": [str(path) for path in written],
        "python": python_path,
    }
