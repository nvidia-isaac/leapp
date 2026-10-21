#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Static TensorRT ONNX op catalog checks (optional TensorRT parse/build)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import onnx

from isaac_deploy_trt.tensorrt_onnx_ops import (
    TENSORRT_ONNX_DEFAULT_VERSION,
    onnx_ops_for_version,
)

logger = logging.getLogger(__name__)

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tensorrt_catalogs"


@dataclass
class TensorRTOnnxValidationResult:
    onnx_path: str
    used_ops: List[str] = field(default_factory=list)
    unsupported_ops: List[str] = field(default_factory=list)
    unknown_ops: List[str] = field(default_factory=list)
    parse_ok: Optional[bool] = None
    build_ok: Optional[bool] = None
    catalog_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.errors:
            return False
        if self.unsupported_ops or self.unknown_ops:
            return False
        if self.parse_ok is False or self.build_ok is False:
            return False
        return True


def collect_onnx_op_types(onnx_path: str) -> List[str]:
    model = onnx.load(onnx_path, load_external_data=False)
    return sorted({node.op_type for node in model.graph.node})


def _release_tag(version: str) -> str:
    cleaned = str(version).strip()
    if cleaned[:1] in "vV" and len(cleaned) > 1 and cleaned[1].isdigit():
        return "v" + cleaned[1:]
    if cleaned[:1].isdigit():
        return "v" + cleaned
    raise ValueError(f"Invalid TensorRT version: {version!r}")


def _cuda_matches(entry: str, requested: str) -> bool:
    want = requested.strip().lower()
    have = entry.strip().lower()
    if have.endswith(".x"):
        return want == have[:-2] or want.startswith(have[:-1])
    return want == have or want.startswith(have + ".")


def resolve_tensorrt_catalog(
    tensorrt: str,
    *,
    platform: str = "x86",
    cuda: Optional[str] = None,
) -> Path:
    """Return the platform catalog JSON for a TensorRT release.

    ``10.16`` and ``v10.16`` both resolve through ``tensorrt_catalogs/index.json``.
    A patch release such as ``10.16.1`` falls back to ``10.16`` when that
    database exists.
    """
    index_path = _CATALOG_ROOT / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"TensorRT catalog index not found: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    mapping = index.get("release_to_database") or {}
    tag = _release_tag(tensorrt)
    database = mapping.get(tag)
    if database is None:
        parts = tag[1:].split(".")
        if len(parts) >= 3:
            database = mapping.get("v" + ".".join(parts[:2]))
    if database is None:
        prefix = "v" + ".".join(tag[1:].split(".")[:2]) if "." in tag[1:] else tag
        matches = sorted({
            value for key, value in mapping.items()
            if key == prefix or key.startswith(prefix + ".")
        })
        if len(matches) == 1:
            database = matches[0]
        elif not matches:
            known = ", ".join(sorted(mapping))
            raise KeyError(
                f"No ONNX operator catalog for TensorRT {tensorrt!r} "
                f"(known releases: {known})"
            )
        else:
            raise KeyError(
                f"TensorRT {tensorrt!r} matches multiple catalogs {matches}. "
                "Pass a full release such as 10.16 or 11.2."
            )

    directory = _CATALOG_ROOT / str(database)
    files = sorted(directory.glob(f"tensorrt_layers_*_{platform}.json"))
    if cuda:
        files = [
            path for path in files
            if _cuda_matches(_catalog_cuda(path), cuda)
        ]
    if len(files) == 1:
        return files[0]
    available = sorted(
        path.name for path in directory.glob("tensorrt_layers_*.json")
    )
    if not files:
        raise KeyError(
            f"No {platform} catalog for TensorRT {tensorrt!r} in {directory} "
            f"(have {available})"
        )
    raise KeyError(
        f"Multiple catalogs for TensorRT {tensorrt!r} platform {platform}: "
        f"{[path.name for path in files]}. Pass --cuda to select one."
    )


def _catalog_cuda(path: Path) -> str:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    cuda = catalog.get("cuda")
    return "" if cuda is None else str(cuda)


def _ops_from_catalog_file(path: Path) -> Dict[str, dict]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    operators = catalog.get("onnx_operators") or {}
    if not isinstance(operators, dict) or not operators:
        raise ValueError(f"Catalog has no onnx_operators: {path}")
    return operators


def select_onnx_ops(
    *,
    tensorrt: Optional[str] = None,
    cuda: Optional[str] = None,
    platform: Optional[str] = None,
    catalog: Optional[str] = None,
) -> tuple[Dict[str, dict], Optional[str]]:
    """Return ``(operator map, catalog path)``.

    The baked-in parser table is used when no version, platform, or catalog
    file is requested, and when ``--tensorrt`` names that table (currently
    11.2) without a platform overlay. Any other release is loaded from
    ``tensorrt_catalogs``.
    """
    if catalog:
        path = Path(catalog).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Catalog not found: {path}")
        return _ops_from_catalog_file(path), str(path)

    if platform is None and catalog is None:
        try:
            return onnx_ops_for_version(tensorrt, cuda=cuda), None
        except KeyError:
            if tensorrt is None:
                raise

    version = tensorrt or TENSORRT_ONNX_DEFAULT_VERSION
    path = resolve_tensorrt_catalog(version, platform=platform or "x86", cuda=cuda)
    return _ops_from_catalog_file(path), str(path)


def check_onnx_ops_against_tensorrt(
    onnx_path: str,
    *,
    known_supported: Optional[Sequence[str]] = None,
    tensorrt: Optional[str] = None,
    cuda: Optional[str] = None,
    platform: Optional[str] = None,
    catalog: Optional[str] = None,
) -> TensorRTOnnxValidationResult:
    used_ops = collect_onnx_op_types(onnx_path)
    ops_map, catalog_path = select_onnx_ops(
        tensorrt=tensorrt,
        cuda=cuda,
        platform=platform,
        catalog=catalog,
    )
    supported = (
        set(known_supported)
        if known_supported is not None
        else {
            name for name, info in ops_map.items()
            if isinstance(info, dict) and info.get("supported")
        }
    )
    catalog_names = set(ops_map)
    unsupported = sorted(op for op in used_ops if op in catalog_names and op not in supported)
    unknown = sorted(op for op in used_ops if op not in catalog_names)
    result = TensorRTOnnxValidationResult(
        onnx_path=onnx_path,
        used_ops=used_ops,
        unsupported_ops=unsupported,
        unknown_ops=unknown,
        catalog_path=catalog_path,
    )
    if unsupported:
        result.errors.append(
            f"ONNX ops not marked supported by TensorRT catalog: {unsupported}"
        )
    if unknown:
        result.errors.append(
            f"ONNX ops not present in TensorRT catalog: {unknown}"
        )
    return result


def try_tensorrt_build(onnx_path: str, *, build: bool = True) -> TensorRTOnnxValidationResult:
    result = TensorRTOnnxValidationResult(onnx_path=onnx_path)
    try:
        import tensorrt as trt
    except ImportError as exc:
        result.parse_ok = False
        if build:
            result.build_ok = False
        result.errors.append(f"tensorrt is not installed: {exc}")
        return result

    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger_trt)
    with open(onnx_path, "rb") as handle:
        parse_ok = parser.parse(handle.read())
    result.parse_ok = bool(parse_ok)
    if not parse_ok:
        for i in range(parser.num_errors):
            result.errors.append(str(parser.get_error(i)))
        result.build_ok = False if build else None
        return result
    if not build:
        return result
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    result.build_ok = engine is not None
    if engine is None:
        result.errors.append("TensorRT build_serialized_network returned None")
    return result


def validate_onnx_for_tensorrt(
    onnx_path: str,
    *,
    build: bool = False,
    strict: bool = False,
    tensorrt: Optional[str] = None,
    cuda: Optional[str] = None,
    platform: Optional[str] = None,
    catalog: Optional[str] = None,
) -> TensorRTOnnxValidationResult:
    result = check_onnx_ops_against_tensorrt(
        onnx_path,
        tensorrt=tensorrt,
        cuda=cuda,
        platform=platform,
        catalog=catalog,
    )
    if build:
        built = try_tensorrt_build(onnx_path, build=True)
        result.parse_ok = built.parse_ok
        result.build_ok = built.build_ok
        result.errors.extend(built.errors)
    if result.ok:
        logger.info("TensorRT catalog check passed for %s", onnx_path)
    else:
        message = f"TensorRT issues for {onnx_path}: " + "; ".join(result.errors)
        if strict:
            raise ValueError(message)
        logger.warning(message)
    return result
