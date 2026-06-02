"""Pack a Warp APIC `.wrp` graph plus its `_modules/` directory into one blob.

A captured APIC graph on disk is two things:

    <stem>.wrp
    <stem>_modules/<hash>.cubin   (+ .meta, .o, ...)

`wp_apic_load_graph` only accepts a file path and resolves the sibling
`<stem>_modules/` directory relative to the `.wrp`. To make the artifact fully
self-contained inside an ONNX model, we pack the whole bundle into a single
little-endian archive and store it as a `uint8` tensor initializer.

Storing the bundle as an initializer (rather than a node attribute) is what lets
ONNX's external-data mechanism spill large bundles into a sibling
`<model>.onnx.data` file, exactly like LEAPP does for big weights. Attributes
cannot be externalized and are bounded by the 2 GB protobuf limit; initializers
are not.

Archive layout (all integers little-endian):

    magic           4 bytes   "WRPB"
    version         uint32    == 1
    num_entries     uint32
    repeat num_entries times:
        path_len    uint32
        path        path_len bytes (UTF-8, forward-slash relative path)
        data_len    uint64
        data        data_len bytes

The C++ kernel decodes this exact format, extracts every entry into a temp
directory preserving relative paths, then loads `<temp>/<wrp_name>`.
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
    """Return (archive_bytes, wrp_filename) for a captured APIC bundle."""
    wrp_path = wrp_path.resolve()
    files = _collect_bundle_files(wrp_path)
    archive = build_archive(files)
    return archive, wrp_path.name
