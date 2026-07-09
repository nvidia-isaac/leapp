#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared helpers for warp-lang functional and unit tests."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WARP_RUNTIME_DIR = (
    _REPO_ROOT
    / "leapp"
    / "leapp_graph"
    / "custom_operator_registry"
    / "warp_operator"
    / "runtime"
)
_WARP_CUSTOM_OP_LIB = _WARP_RUNTIME_DIR / "build" / "libleapp_wrp_onnx_custom_op.so"


def ensure_warp_onnx_custom_op_library() -> str:
    """Return the Warp ONNX custom-op library path, building it if needed.

    TODO: Remove the cmake build fallback once the ONNX custom op is always
    shipped prebuilt with the package/CI image. After that, this helper should
    only resolve the bundled library path and set LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY.
    """
    if _WARP_CUSTOM_OP_LIB.is_file():
        return str(_WARP_CUSTOM_OP_LIB)

    build_dir = _WARP_RUNTIME_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(_WARP_RUNTIME_DIR),
            "-B",
            str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", "8"],
        check=True,
    )
    if not _WARP_CUSTOM_OP_LIB.is_file():
        raise FileNotFoundError(
            f"Warp ONNX custom op library was not built: {_WARP_CUSTOM_OP_LIB}"
        )
    return str(_WARP_CUSTOM_OP_LIB)


class WarpTestCase(unittest.TestCase):
    """Base unittest class that registers the Warp ONNX custom op for tests.

    TODO: Remove once the ONNX custom op is always prebuilt and discovered
    automatically; warp tests can then inherit LEAPPFunctionalTestBase only.
    """

    @classmethod
    def setUpClass(cls):
        lib_path = ensure_warp_onnx_custom_op_library()
        os.environ["LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY"] = lib_path
