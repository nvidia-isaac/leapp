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

import warp as wp

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

    DEVICE = "cuda"

    def _launch_add(self, values, value):
        output = wp.empty_like(values)
        wp.launch(
            self.kernels.add_scalar,
            dim=values.size,
            inputs=[values, wp.float32(value)],
            outputs=[output],
            device=values.device,
        )
        return output

    def _launch_increment_in_place(self, values):
        wp.launch(
            self.kernels.increment_in_place,
            dim=values.size,
            inputs=[values],
            device=values.device,
        )
        return values

    def _launch_manual_add(self, values, value, node_name=None):
        from leapp.leapp import _MANAGER as annotate

        if node_name is None:
            node_name = self.NODE_NAME
        with annotate.warp_op(node_name, device=values.device):
            output = self._launch_add(values, value)
        return output

    def _torch_roundtrip(self, values, value):
        tensor = wp.to_torch(values)
        tensor = tensor + value
        return wp.from_torch(tensor)

    def _numpy_roundtrip(self, values, value):
        array = values.numpy()
        array = array + value
        return wp.from_numpy(array, device=values.device)

    def _assert_compiled_segments(self, node, expected_segments):
        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), expected_segments)
        self.assertEqual(
            [segment.runner_name for segment in node.warp_segments],
            [f"warp_segment_{index}" for index in range(expected_segments)],
        )
        self.assertTrue(
            all(segment.apic_graph is not None for segment in node.warp_segments)
        )
        self.verify_all_models_exist(self.NODE_NAME)

    class kernels:
        @wp.kernel
        def add_scalar(
            src: wp.array(dtype=wp.float32),
            value: wp.float32,
            dst: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()
            dst[i] = src[i] + value

        @wp.kernel
        def divide_in_place(
            data: wp.array(dtype=wp.float32),
            divisor: wp.float32,
        ):
            i = wp.tid()
            data[i] = data[i] / divisor

        @wp.kernel
        def increment_in_place(data: wp.array(dtype=wp.float32)):
            i = wp.tid()
            data[i] = data[i] + 1.0

        @wp.kernel
        def average_three(
            a: wp.array(dtype=wp.float32),
            b: wp.array(dtype=wp.float32),
            c: wp.array(dtype=wp.float32),
            out: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()
            out[i] = (a[i] + b[i] + c[i]) / 3.0

        @wp.kernel
        def scale_and_split(
            data: wp.array(dtype=wp.float32),
            out_sum: wp.array(dtype=wp.float32),
            out_diff: wp.array(dtype=wp.float32),
        ):
            i = wp.tid()
            original = data[i]
            scaled = original / 2.0
            data[i] = scaled
            out_sum[i] = scaled + original
            out_diff[i] = original - scaled

    @classmethod
    def setUpClass(cls):
        lib_path = ensure_warp_onnx_custom_op_library()
        os.environ["LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY"] = lib_path
