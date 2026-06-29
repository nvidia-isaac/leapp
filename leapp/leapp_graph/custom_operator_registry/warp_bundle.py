#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Pack a Warp APIC ``.wrp`` graph plus its ``_modules/`` directory into one blob.

Archive layout matches ``experiment_warp/onnx_embedded_wrp/wrp_bundle.py`` (WRPB v1).
"""

from __future__ import annotations

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
    from leapp.leapp_graph.custom_operator_registry import warp_custom_op

    warp_op = warp_custom_op.get_op().default
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
    from leapp.leapp_graph.custom_operator_registry import warp_custom_op
    from leapp.utils.logging import _get_logger

    warp_op = warp_custom_op.get_op().default
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
        if len(args) < 4:
            raise ValueError(
                f"Warp runner node '{node.name}' expected at least 4 args, got {len(args)}."
            )
        if len(args) == 4:
            args.append(bundle_node)
        else:
            args[4] = bundle_node
        node.args = tuple(args)
        embedded += 1
        _get_logger().debug(
            f"Embedded Warp bundle ({len(archive)} bytes) on FX node '{node.name}'."
        )

    graph_module.recompile()
    return embedded
