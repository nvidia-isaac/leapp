#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Layers that every checked-in TensorRT catalog marks GPU-supported."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_CATALOG_ROOT = Path(__file__).resolve().parents[2] / "tensorrt_catalogs"


def _release_sort_key(tag: str) -> tuple:
    parts = []
    for piece in tag.lstrip("vV").replace("-", ".").split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


def catalog_layer_files(root: Path | None = None) -> List[Path]:
    """Return every platform layer JSON under ``tensorrt_catalogs``."""
    base = root or _CATALOG_ROOT
    files = []
    for path in sorted(base.glob("*/tensorrt_layers_*.json")):
        files.append(path)
    return files


def _minimal_onnx_operators(catalogs: list) -> Dict[str, dict]:
    """ONNX ops supported only when every catalog marks them supported."""
    maps = [payload.get("onnx_operators") or {} for _path, payload in catalogs]
    maps = [ops for ops in maps if ops]
    if not maps:
        return {}
    names = set()
    for ops in maps:
        names.update(ops)
    result: Dict[str, dict] = {}
    for name in sorted(names):
        infos = [ops.get(name) for ops in maps]
        present = [info for info in infos if isinstance(info, dict)]
        supported = (
            len(present) == len(maps)
            and all(info.get("supported") for info in present)
        )
        type_sets = [set(info.get("types") or []) for info in present]
        types = sorted(set.intersection(*type_sets)) if type_sets and supported else []
        if supported and any(info.get("types") for info in present) and not types:
            supported = False
        result[name] = {
            "supported": supported,
            "types": types if supported else [],
            "restrictions": (
                ""
                if supported
                else "Not supported in every checked-in TensorRT catalog."
            ),
        }
    return result


def minimal_catalog_path() -> Path:
    """Checked-in minimal catalog used when no TensorRT version is requested."""
    return _CATALOG_ROOT / "minimal" / "tensorrt_layers.json"


def minimal_supported_layers(root: Path | None = None) -> Dict[str, object]:
    """Intersect native TensorRT layers across every release and platform.

    A layer is included only when every catalog file lists it and sets
    ``gpu`` true. Layers added in a later header, or marked unsupported on
    JetPack, are listed under ``excluded``.
    """
    files = catalog_layer_files(root)
    if not files:
        raise FileNotFoundError(f"No TensorRT layer catalogs under {root or _CATALOG_ROOT}")

    catalogs = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        catalogs.append((path, payload))

    names = set()
    for _path, payload in catalogs:
        names.update((payload.get("tensorrt_layers") or {}))

    releases = sorted({
        release
        for _path, payload in catalogs
        for release in (payload.get("supported_releases") or [])
    }, key=_release_sort_key)
    platforms = sorted({
        str(payload.get("platform"))
        for _path, payload in catalogs
        if payload.get("platform")
    })

    layers: Dict[str, Dict[str, object]] = {}
    excluded: List[Dict[str, object]] = []
    for name in sorted(names):
        missing = []
        gpu_false = []
        for path, payload in catalogs:
            table = payload.get("tensorrt_layers") or {}
            label = f"{path.parent.name}/{payload.get('platform') or path.stem}"
            if name not in table:
                missing.append(label)
                continue
            if not table[name].get("gpu"):
                gpu_false.append(label)
        if missing or gpu_false:
            excluded.append({
                "name": name,
                "missing_from": missing,
                "gpu_false_on": gpu_false,
            })
            continue
        layers[name] = {
            "gpu": True,
            "dla": False,
            "notes": "GPU-supported in every checked-in TensorRT catalog.",
        }

    return {
        "kind": "minimal_supported_layers",
        "rule": "gpu=true in every tensorrt_catalogs layer table",
        "releases": releases,
        "platforms": platforms,
        "catalog_count": len(catalogs),
        "layer_count": len(layers),
        "layers": layers,
        "onnx_operators": _minimal_onnx_operators(catalogs),
        "excluded": excluded,
        "notes": [
            "DLA is not part of this set. x86 and Thor catalogs have no DLA.",
            "onnx_operators is the intersection used when no TensorRT version is requested.",
        ],
    }


def format_minimal_layers(report: Dict[str, object]) -> str:
    """Render the minimal layer set as plain text."""
    layers = report["layers"]
    excluded = report["excluded"]
    lines = [
        f"Minimal TensorRT layers: {report['layer_count']}",
        f"releases: {', '.join(report['releases'])}",
        f"platforms: {', '.join(report['platforms'])}",
        f"catalogs: {report['catalog_count']}",
        "",
    ]
    for name in layers:
        lines.append(name)
    lines.append("")
    lines.append(f"Excluded: {len(excluded)}")
    for item in excluded:
        reason = []
        if item["missing_from"]:
            reason.append(f"missing from {len(item['missing_from'])} catalogs")
        if item["gpu_false_on"]:
            reason.append(f"gpu=false on {len(item['gpu_false_on'])} catalogs")
        lines.append(f"{item['name']}: {'; '.join(reason)}")
    return "\n".join(lines)
