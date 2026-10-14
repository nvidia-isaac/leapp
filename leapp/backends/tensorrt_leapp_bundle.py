#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Build a LEAPP export dir whose ONNX files are TensorRT-legal drop-ins.

Keeps original filenames, graph I/O, pipeline YAML, and ``*.onnx.data``
siblings so Isaac Deploy / Triton can consume the bundle without renaming
models. Graphs that need GraphSurgeon (UINT8 Cast, Resize antialias, rank-9
Reshape) are rewritten; others are copied byte-for-byte.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import onnx
import yaml

from isaac_deploy_trt.prepare import needs_tensorrt_rewrite
from isaac_deploy_trt.surgery import (
    apply_tensorrt_rewrites,
    export_model,
    graph_io_signature,
    load_graph,
    save_graph,
)
from isaac_deploy_trt.validate import check_onnx_ops_against_tensorrt
from leapp.utils.logging import _get_logger


_SKIP_ONNX_SUFFIXES = ("_trt.onnx",)


@dataclass
class AlignedOnnxResult:
    """One ONNX file in an aligned TensorRT LEAPP bundle."""

    filename: str
    rewritten: bool
    rewritten_nodes: List[str] = field(default_factory=list)
    md5sum: str = ""
    sha256sum: str = ""


@dataclass
class AlignedBundleResult:
    """Result of ``align_leapp_export_for_tensorrt``."""

    output_dir: str
    models: List[AlignedOnnxResult] = field(default_factory=list)
    yaml_paths: List[str] = field(default_factory=list)


def _file_hashes(path: str) -> tuple[str, str]:
    with open(path, "rb") as handle:
        data = handle.read()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def _copy_with_external_data(src_onnx: str, dst_onnx: str) -> None:
    """Link unchanged ONNX (and ``*.data``) into the aligned dir."""
    src_onnx = os.path.abspath(src_onnx)
    if os.path.lexists(dst_onnx):
        os.remove(dst_onnx)
    os.symlink(src_onnx, dst_onnx)
    src_dir = os.path.dirname(src_onnx)
    dst_dir = os.path.dirname(dst_onnx)
    stem = os.path.basename(src_onnx)
    for name in os.listdir(src_dir):
        if not (name.startswith(stem + ".") or name == stem + ".data"):
            continue
        dst_data = os.path.join(dst_dir, name)
        if os.path.lexists(dst_data):
            os.remove(dst_data)
        os.symlink(os.path.join(src_dir, name), dst_data)


def _has_external_data(onnx_path: str) -> bool:
    parent = os.path.dirname(onnx_path)
    stem = os.path.basename(onnx_path)
    return os.path.isfile(os.path.join(parent, stem + ".data"))


def _save_rewritten(
    graph,
    src_onnx: str,
    dst_onnx: str,
) -> None:
    if _has_external_data(src_onnx):
        model = export_model(graph)
        data_name = os.path.basename(dst_onnx) + ".data"
        data_path = os.path.join(os.path.dirname(dst_onnx), data_name)
        if os.path.exists(data_path):
            os.remove(data_path)
        onnx.save(
            model,
            dst_onnx,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
            convert_attribute=True,
        )
        return
    save_graph(graph, dst_onnx)


def _yaml_model_files(export_dir: str) -> Optional[List[str]]:
    names: List[str] = []
    for filename in sorted(os.listdir(export_dir)):
        if not filename.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(export_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            spec = yaml.safe_load(handle) or {}
        models = spec.get("models") or {}
        if not isinstance(models, dict):
            continue
        for model_cfg in models.values():
            params = (model_cfg or {}).get("parameters") or {}
            model_path = params.get("model_path")
            if model_path:
                names.append(os.path.basename(str(model_path)))
    # Preserve YAML order, drop duplicates.
    return list(dict.fromkeys(names)) or None


def _update_yaml_parameters(
    yaml_path: str,
    hashes_by_filename: Dict[str, tuple[str, str]],
    trt_compatible_by_filename: Dict[str, bool],
) -> None:
    with open(yaml_path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    models = spec.get("models")
    if not isinstance(models, dict):
        return
    changed = False
    for model_cfg in models.values():
        params = (model_cfg or {}).get("parameters") or {}
        model_path = params.get("model_path")
        if not model_path:
            continue
        filename = os.path.basename(str(model_path))
        if filename in hashes_by_filename:
            md5sum, sha256sum = hashes_by_filename[filename]
            if params.get("md5sum") != md5sum or params.get("sha256sum") != sha256sum:
                params["md5sum"] = md5sum
                params["sha256sum"] = sha256sum
                changed = True
        if filename in trt_compatible_by_filename:
            flag = bool(trt_compatible_by_filename[filename])
            if params.get("tensorrt_compatible") != flag:
                params["tensorrt_compatible"] = flag
                changed = True
    if not changed:
        return
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)


def align_leapp_export_for_tensorrt(
    export_dir: str,
    output_dir: str,
) -> AlignedBundleResult:
    """Write a sibling LEAPP dir with TensorRT-legal ONNX and aligned YAML.

    Filenames stay ``preprocess_video.onnx``, ``backbone.onnx``, etc. Graph I/O
    signatures are checked against the source models so Triton bindings still
    match. ``*_trt.onnx`` sidecars in the source dir are skipped.
    """
    export_dir = os.path.abspath(export_dir)
    output_dir = os.path.abspath(output_dir)
    if os.path.abspath(export_dir) == output_dir:
        raise ValueError("output_dir must be different from export_dir")
    os.makedirs(output_dir, exist_ok=True)
    logger = _get_logger()

    yaml_names = _yaml_model_files(export_dir)
    if yaml_names is not None:
        onnx_files = [
            os.path.join(export_dir, name)
            for name in yaml_names
            if os.path.isfile(os.path.join(export_dir, name))
        ]
    else:
        onnx_files = [
            os.path.join(export_dir, name)
            for name in sorted(os.listdir(export_dir))
            if name.endswith(".onnx")
            and not any(name.endswith(suffix) for suffix in _SKIP_ONNX_SUFFIXES)
        ]

    result = AlignedBundleResult(output_dir=output_dir)
    hashes: Dict[str, tuple[str, str]] = {}
    trt_compatible: Dict[str, bool] = {}

    for src in onnx_files:
        filename = os.path.basename(src)
        dst = os.path.join(output_dir, filename)
        source_model = onnx.load(src, load_external_data=False)
        source_io = graph_io_signature(source_model)

        if not needs_tensorrt_rewrite(source_model):
            _copy_with_external_data(src, dst)
            logger.info(f"Copied {filename} (already TensorRT-aligned I/O)")
            rewritten_nodes: List[str] = []
        else:
            graph = load_graph(src)
            rewritten_nodes = apply_tensorrt_rewrites(graph)
            _save_rewritten(graph, src, dst)
            aligned = onnx.load(dst, load_external_data=False)
            aligned_io = graph_io_signature(aligned)
            if aligned_io != source_io:
                raise RuntimeError(
                    f"TensorRT rewrite changed graph I/O for {filename}: "
                    f"{source_io} -> {aligned_io}"
                )
            logger.info(
                f"Rewrote {filename} for TensorRT ({len(rewritten_nodes)} node(s))"
            )

        md5sum, sha256sum = _file_hashes(dst)
        hashes[filename] = (md5sum, sha256sum)
        trt_compatible[filename] = bool(
            check_onnx_ops_against_tensorrt(dst).ok
        )
        result.models.append(
            AlignedOnnxResult(
                filename=filename,
                rewritten=bool(rewritten_nodes),
                rewritten_nodes=rewritten_nodes,
                md5sum=md5sum,
                sha256sum=sha256sum,
            )
        )

    for filename in os.listdir(export_dir):
        if not filename.endswith((".yaml", ".yml")):
            continue
        src_yaml = os.path.join(export_dir, filename)
        dst_yaml = os.path.join(output_dir, filename)
        shutil.copy2(src_yaml, dst_yaml)
        _update_yaml_parameters(dst_yaml, hashes, trt_compatible)
        result.yaml_paths.append(dst_yaml)

    return result


def rewrite_leapp_export_for_tensorrt(export_dir: str) -> AlignedBundleResult:
    """Apply TensorRT GraphSurgeon rewrites in place under ``export_dir``.

    Uses a temporary sibling directory so graph I/O and YAML checksums stay
    aligned, then copies rewritten ONNX / YAML back onto the compiled graph.
    Unchanged ONNX files are left as they were.
    """
    export_dir = os.path.abspath(export_dir)
    tmp_dir = tempfile.mkdtemp(prefix=".leapp_trt_opt_", dir=export_dir)
    try:
        result = align_leapp_export_for_tensorrt(export_dir, tmp_dir)
        for name in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, name)
            dst = os.path.join(export_dir, name)
            if os.path.islink(src) or not os.path.isfile(src):
                continue
            shutil.copy2(src, dst)
        result.output_dir = export_dir
        result.yaml_paths = [
            os.path.join(export_dir, os.path.basename(path))
            for path in result.yaml_paths
        ]
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
