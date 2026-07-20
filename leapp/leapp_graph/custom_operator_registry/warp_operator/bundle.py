#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Self-contained Warp APIC bundle serialization and FX graph embedding.

Purpose
-------
During LEAPP trace, each Warp segment's APIC graph is saved to a **temporary**
``.wrp`` file (plus a sibling ``*_modules/`` tree) by ``WarpOp``. That layout is
fine on the build machine but is not portable inside an exported ``.pt2`` or
``.onnx`` artifact.

This module provides:

1. **WRPB archive** (``pack_bundle``, ``build_archive``) — read a captured
   ``.wrp`` file and its sibling ``*_modules/`` directory from disk and pack
   them into a single byte blob (magic ``WRPB``, version 1). The format matches
   the experiment prototype in ``experiment_warp/onnx_embedded_wrp/wrp_bundle.py``.

2. **Trace-time embedding** — ``WarpOp`` packs the saved bundle via
   ``pack_bundle`` and ``TracedTensorNode.create_warp_proxy`` registers the
   bytes as a ``uint8`` buffer and wires a ``get_attr`` as the op's bundle
   input.

Pipeline position
-----------------
::

    trace (WarpOp)  →  save .wrp to temp dir, pack WRPB + wire get_attr (trace)
                    →  export .pt2 / .onnx (warp_operator lowering)
                    →  ONNX-only: patch WrpRunner string attrs (onnx_export_backend)

``iter_warp_segments_from_graph`` is a small helper shared with save/ONNX passes
to list ``WarpSegment`` objects in ``warp_runner`` node order.

Why a separate module
---------------------
``warp_operator`` owns op registration, fake/eager kernels, and ONNX
lowering. This file owns **on-disk → in-graph bytes** wiring only. They could
be merged into one export-support module, but keeping WRPB serialization here
avoids coupling a binary archive format to ``torch.library`` registration.
``datatypes.warp.warp_segment`` holds segment *state*; this module handles
segment *payload* embedding at export time.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.fx as fx

from leapp.utils.logging import _get_logger

MAGIC = b"WRPB"
VERSION = 1
WRP_FILENAME = "segment.wrp"


def _collect_bundle_files(wrp_path: Path) -> list[tuple[str, bytes]]:
    wrp_path = wrp_path.resolve()
    if not wrp_path.is_file():
        _get_logger().fatal(
            f".wrp file not found: {wrp_path}",
            error_type=FileNotFoundError,
        )

    parent = wrp_path.parent
    files: list[tuple[str, bytes]] = [(wrp_path.name, wrp_path.read_bytes())]

    modules_dir = wrp_path.with_name(wrp_path.stem + "_modules")
    if modules_dir.is_dir():
        for entry in sorted(modules_dir.rglob("*")):
            if entry.is_file():
                rel = entry.relative_to(parent).as_posix()
                files.append((rel, entry.read_bytes()))

    return files


def build_archive(files: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", len(files))
    for rel_path, data in files:
        rel_bytes = rel_path.encode("utf-8")
        out += struct.pack("<I", len(rel_bytes))
        out += rel_bytes
        out += struct.pack("<Q", len(data))
        out += data
    return bytes(out)


def pack_bundle(wrp_path: Path) -> bytes:
    """Return WRPB archive bytes for a captured APIC bundle."""
    wrp_path = wrp_path.resolve()
    files = _collect_bundle_files(wrp_path)
    return build_archive(files)


def _is_warp_runner_node(node: fx.Node, warp_op) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if target is warp_op:
        return True
    if (
        isinstance(target, torch._ops.OpOverload)
        and target.overloadpacket is warp_op.overloadpacket
    ):
        return True
    return False


def iter_warp_segments_from_graph(graph: fx.Graph) -> list:
    """Return warp segment objects in ``warp_runner`` node order."""
    from .schema import get_op

    warp_op = get_op().default
    segments = []
    for node in graph.nodes:
        if not _is_warp_runner_node(node, warp_op):
            continue
        segment = node.meta.get("leapp_warp_segment")
        if segment is not None:
            segments.append(segment)
    return segments

