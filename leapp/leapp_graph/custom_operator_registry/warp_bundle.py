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
