#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Self-contained Warp APIC bundle serialization and FX graph embedding.

Purpose
-------
During LEAPP trace, each Warp segment's APIC graph is saved to a **temporary**
``.wrp`` file (plus a sibling ``*_modules/`` tree) by ``TracedTensorNode``.
That layout is fine on the build machine but is not portable inside an exported
``.pt2`` or ``.onnx`` artifact.

This module closes that gap in two steps:

1. **WRPB archive** (``pack_bundle``, ``build_archive``) — read ``segment.wrp``
   and its ``segment_modules/`` directory from disk and pack them into a single
   byte blob (magic ``WRPB``, version 1). The format matches the experiment
   prototype in ``experiment_warp/onnx_embedded_wrp/wrp_bundle.py``.

2. **FX embed pass** (``embed_warp_bundles_in_graph``) — before
   ``torch.export`` or ONNX dynamo export, replace each ``leapp::warp_runner``
   node's zero-length bundle placeholder with a ``get_attr`` to a registered
   ``uint8`` buffer holding the WRPB bytes. The exported graph is then
   self-contained: no external ``.wrp`` paths at inference time.

Pipeline position
-----------------
::

    trace (WarpOp)  →  save .wrp to temp dir (traced_node)
                    →  pack WRPB + wire into FX graph (this module, pre_compile)
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

import hashlib
import struct
from pathlib import Path

import torch
import torch.fx as fx

MAGIC = b"WRPB"
VERSION = 1


def _collect_bundle_files(wrp_path: Path) -> list[tuple[str, bytes]]:
    wrp_path = wrp_path.resolve()
    if not wrp_path.is_file():
        raise FileNotFoundError(f".wrp file not found: {wrp_path}")

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


def pack_bundle(wrp_path: Path) -> tuple[bytes, str]:
    """Return ``(archive_bytes, wrp_filename)`` for a captured APIC bundle."""
    wrp_path = wrp_path.resolve()
    files = _collect_bundle_files(wrp_path)
    archive = build_archive(files)
    return archive, wrp_path.name


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


def embed_warp_bundles_in_graph(graph_module: fx.GraphModule) -> int:
    """Wire saved WRPB archives into ``warp_runner`` nodes as CPU ``uint8`` inputs.

    Each ``leapp::warp_runner`` call receives the bundle as its last argument
    (a ``get_attr`` to a registered buffer). Segments are correlated via
    ``node.meta['leapp_warp_segment']`` in graph order. Called from the Warp
    ``pre_compile`` hook before ``onnx-dynamo`` or ``exported-program`` export.
    """
    from .schema import get_op
    from leapp.utils.logging import _get_logger

    warp_op = get_op().default
    warp_nodes = [
        node for node in graph_module.graph.nodes if _is_warp_runner_node(node, warp_op)
    ]
    if not warp_nodes:
        return 0

    embedded = 0
    for index, node in enumerate(warp_nodes):
        segment = node.meta.get("leapp_warp_segment")
        if segment is None or segment.wrp_path is None:
            raise ValueError(
                f"Warp runner node '{node.name}' has no saved APIC bundle "
                "(segment.wrp_path is missing)."
            )

        archive, _wrp_name = pack_bundle(Path(segment.wrp_path))
        try:
            from .metadata import decode_runtime_metadata, encode_runtime_metadata

            metadata = decode_runtime_metadata(node.args[1])
            metadata.setdefault("bundle", {})
            metadata["bundle"].update(
                {
                    "format": "WRPB",
                    "version": VERSION,
                    "num_bytes": len(archive),
                    "sha256": hashlib.sha256(archive).hexdigest(),
                }
            )
            node.update_arg(1, encode_runtime_metadata(metadata))
        except Exception as exc:
            raise ValueError(
                f"Warp runner node '{node.name}' has invalid runtime metadata: {exc}"
            ) from exc

        buffer_name = f"_warp_bundle_{index}"
        bundle_tensor = torch.frombuffer(bytearray(archive), dtype=torch.uint8).clone()
        graph_module.register_buffer(buffer_name, bundle_tensor, persistent=True)

        with graph_module.graph.inserting_before(node):
            bundle_node = graph_module.graph.create_node(
                "get_attr",
                buffer_name,
                (),
                {},
                name=f"{node.name}_bundle",
            )

        args = list(node.args)
        if len(args) < 2:
            raise ValueError(
                f"Warp runner node '{node.name}' expected at least 2 args, got {len(args)}."
            )
        if len(args) == 2:
            args.append(bundle_node)
        else:
            args[2] = bundle_node
        node.args = tuple(args)
        embedded += 1
        _get_logger().debug(
            f"Embedded Warp bundle ({len(archive)} bytes) on FX node '{node.name}'."
        )

    graph_module.recompile()
    return embedded
