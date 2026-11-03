#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Rewrite one ONNX file or a directory of ONNX files for TensorRT."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import onnx

from isaac_deploy_trt.surgery import (
    apply_tensorrt_rewrites,
    graph_io_signature,
    load_graph,
    save_graph,
)
from isaac_deploy_trt.tensorrt_onnx_ops import (
    BOOL_ONLY_SURGEON_OPS,
    SURGEON_UNSUPPORTED_OPS,
)
from isaac_deploy_trt.validate import select_onnx_ops

logger = logging.getLogger(__name__)


@dataclass
class RewriteFileResult:
    filename: str
    rewritten: bool
    rewritten_nodes: List[str] = field(default_factory=list)
    output_path: str = ""
    catalog_path: str = ""


def _tensor_elem_type(model: onnx.ModelProto, name: str):
    infos = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for info in infos:
        if info.name == name and info.type.HasField("tensor_type"):
            return int(info.type.tensor_type.elem_type)
    return None


def surgeon_ops_for_target(
    *,
    tensorrt: Optional[str] = None,
    platform: Optional[str] = None,
    cuda: Optional[str] = None,
    catalog: Optional[str] = None,
) -> Tuple[Optional[frozenset], Optional[str]]:
    """Return surgeon ops the selected catalog marks unsupported, plus its path.

    No TensorRT release selects the minimal catalog: an operator is decomposed
    only when it is unsupported in every checked-in TensorRT catalog and a
    GraphSurgeon rewrite exists. ``--platform`` applies only with ``--tensorrt``.
    """
    ops_map, catalog_path = select_onnx_ops(
        tensorrt=tensorrt,
        cuda=cuda,
        platform=platform,
        catalog=catalog,
    )
    unsupported = {
        name
        for name, info in ops_map.items()
        if isinstance(info, dict) and not info.get("supported")
    }
    selected = frozenset(unsupported & SURGEON_UNSUPPORTED_OPS)
    logger.info(
        "TensorRT rewrite catalog %s; surgeon ops: %s",
        catalog_path or tensorrt or "baked-in",
        ", ".join(sorted(selected)) or "(none)",
    )
    return selected, catalog_path


def needs_tensorrt_rewrite(
    model: onnx.ModelProto,
    surgeon_ops: Optional[Sequence[str]] = None,
) -> bool:
    allowed = SURGEON_UNSUPPORTED_OPS if surgeon_ops is None else set(surgeon_ops)
    inits = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type in BOOL_ONLY_SURGEON_OPS:
            if node.op_type not in allowed:
                continue
            elem = _tensor_elem_type(model, node.input[0]) if node.input else None
            if elem == 9:  # BOOL
                return True
            continue
        if node.op_type in allowed:
            return True
        if node.op_type == "Cast":
            to_attr = next((a.i for a in node.attribute if a.name == "to"), None)
            if to_attr == 2:
                return True
        if node.op_type == "Resize":
            antialias = next((a.i for a in node.attribute if a.name == "antialias"), 0)
            if int(antialias) != 0:
                return True
        if node.op_type == "Reshape" and len(node.input) >= 2:
            shape_init = inits.get(node.input[1])
            if shape_init is not None and len(shape_init.dims) == 1 and shape_init.dims[0] > 8:
                return True
    return False


def external_data_locations(model: onnx.ModelProto) -> set[str]:
    """Relative paths named by tensors stored outside the ONNX protobuf."""
    tensors = list(model.graph.initializer) + list(model.graph.sparse_initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                tensors.append(attr.t)
            tensors.extend(attr.tensors)
    locations = set()
    for tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value:
                locations.add(entry.value)
    return locations


def copy_external_data(src_onnx: str, dst_onnx: str, model: Optional[onnx.ModelProto] = None) -> None:
    """Copy external weight files referenced by ``src_onnx`` next to ``dst_onnx``.

    GraphSurgeon loads the protobuf without those weights, and the rewritten
    model keeps the same relative locations. The files have to travel with it.
    """
    if model is None:
        model = onnx.load(src_onnx, load_external_data=False)
    src_dir = os.path.dirname(os.path.abspath(src_onnx))
    dst_dir = os.path.dirname(os.path.abspath(dst_onnx))
    for location in sorted(external_data_locations(model)):
        if os.path.isabs(location):
            continue
        src_path = os.path.abspath(os.path.join(src_dir, location))
        if os.path.commonpath([src_dir, src_path]) != src_dir:
            raise ValueError(
                f"{os.path.basename(src_onnx)} external data location escapes "
                f"its directory: {location}"
            )
        if not os.path.isfile(src_path):
            raise FileNotFoundError(
                f"{os.path.basename(src_onnx)} references external data "
                f"{location!r}, but {src_path} is missing"
            )
        dst_path = os.path.abspath(os.path.join(dst_dir, location))
        if src_path == dst_path:
            continue
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        shutil.copy2(src_path, dst_path)


def _copy_with_sidecars(src_onnx: str, dst_onnx: str) -> None:
    shutil.copy2(src_onnx, dst_onnx)
    src_dir = os.path.dirname(os.path.abspath(src_onnx))
    dst_dir = os.path.dirname(os.path.abspath(dst_onnx))
    stem = os.path.basename(src_onnx)
    for name in os.listdir(src_dir):
        if name == stem:
            continue
        if name.startswith(stem + ".") or name == stem + ".data":
            shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))


def rewrite_onnx_file(
    src: str,
    dst: str,
    *,
    tensorrt: Optional[str] = None,
    platform: Optional[str] = None,
    cuda: Optional[str] = None,
    catalog: Optional[str] = None,
) -> RewriteFileResult:
    """Rewrite ``src`` in place-of-copy at ``dst``. Preserves graph I/O.

    ``tensorrt`` and ``platform`` select the operator catalog. Unsupported-op
    decomposition runs only for operators that catalog marks unsupported.
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    filename = os.path.basename(src)
    surgeon_ops, catalog_path = surgeon_ops_for_target(
        tensorrt=tensorrt,
        platform=platform,
        cuda=cuda,
        catalog=catalog,
    )
    source_model = onnx.load(src, load_external_data=False)
    source_io = graph_io_signature(source_model)

    if not needs_tensorrt_rewrite(source_model, surgeon_ops=surgeon_ops):
        _copy_with_sidecars(src, dst)
        logger.info("Copied %s (no TensorRT GraphSurgeon edits needed)", filename)
        return RewriteFileResult(
            filename=filename,
            rewritten=False,
            output_path=dst,
            catalog_path=catalog_path or "",
        )

    graph = load_graph(src)
    rewritten_nodes = apply_tensorrt_rewrites(graph, surgeon_ops=surgeon_ops)
    save_graph(graph, dst)
    copy_external_data(src, dst, source_model)
    aligned = onnx.load(dst, load_external_data=False)
    aligned_io = graph_io_signature(aligned)
    if aligned_io != source_io:
        raise RuntimeError(
            f"TensorRT rewrite changed graph I/O for {filename}: "
            f"{source_io} -> {aligned_io}"
        )
    logger.info("Rewrote %s (%s node(s))", filename, len(rewritten_nodes))
    return RewriteFileResult(
        filename=filename,
        rewritten=True,
        rewritten_nodes=rewritten_nodes,
        output_path=dst,
        catalog_path=catalog_path or "",
    )


def rewrite_directory(
    input_dir: str,
    output_dir: str,
    *,
    tensorrt: Optional[str] = None,
    platform: Optional[str] = None,
    cuda: Optional[str] = None,
    catalog: Optional[str] = None,
) -> List[RewriteFileResult]:
    """Rewrite every ``*.onnx`` under ``input_dir`` into ``output_dir``."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    if input_dir == output_dir:
        raise ValueError("output_dir must differ from input_dir")
    os.makedirs(output_dir, exist_ok=True)
    results: List[RewriteFileResult] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".onnx"):
            continue
        results.append(
            rewrite_onnx_file(
                os.path.join(input_dir, name),
                os.path.join(output_dir, name),
                tensorrt=tensorrt,
                platform=platform,
                cuda=cuda,
                catalog=catalog,
            )
        )
    return results
