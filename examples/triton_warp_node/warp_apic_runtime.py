#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Deploy-side runtime core for replaying a LEAPP Warp (APIC) node from a ``.wrp`` artifact.

Self-contained (NO leapp dependency) — this is the *runtime* side that ships into a Triton
python-backend model (or, later, a custom C++ backend / a ros2_control WarpRunner). It loads a
``<name>.wrp`` + sidecar ``<name>.warpmeta.json`` (written by leapp's export-side
``save_warp_node``), verifies the compiled ``<name>_modules/`` kernels, and replays the captured
APIC graph on torch tensors.

Mirrors the export-side ``_WarpGraphCallable`` contract and its fail-loudly stance: dtype/size/
checksum/device divergences raise rather than silently corrupting results.
"""
import hashlib
import json
import math
import os
from typing import Dict, List

import torch
import warp as wp

_STR_TO_WARP_DTYPE = {"float32": wp.float32, "float64": wp.float64, "int32": wp.int32, "int64": wp.int64}
_STR_TO_TORCH = {"float32": torch.float32, "float64": torch.float64, "int32": torch.int32, "int64": torch.int64}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class WarpApicRunner:
    """Loads a ``.wrp`` (+ sidecar meta + verified modules) and replays it on torch tensors."""

    def __init__(self, wrp_path: str, device: str = None):
        if not os.path.exists(wrp_path):
            raise FileNotFoundError(f"warp artifact not found: {wrp_path}")
        meta_path = os.path.splitext(wrp_path)[0] + ".warpmeta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"warp meta sidecar not found: {meta_path}")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.input_names: List[str] = self.meta["inputs"]
        self.output_names: List[str] = self.meta["outputs"]
        self.input_dtypes: List[str] = self.meta["input_dtypes"]
        self.output_dtypes: List[str] = self.meta["output_dtypes"]
        self.output_shapes: List[list] = self.meta["output_shapes"]

        self._verify_modules(wrp_path, self.meta)

        device_type = self.meta.get("device_type", "cuda")
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"{os.path.basename(wrp_path)} was captured for CUDA but no CUDA device is available")
        self.device = device or ("cuda" if device_type == "cuda" else "cpu")

        wp.init()
        self.graph = wp.capture_load(wrp_path, device=self.device)

    @staticmethod
    def _verify_modules(wrp_path: str, meta: dict):
        expected = meta.get("modules_sha256")
        if not expected:
            return
        modules_dir = os.path.splitext(wrp_path)[0] + "_modules"
        if not os.path.isdir(modules_dir):
            raise FileNotFoundError(f"warp modules dir missing: {modules_dir}")
        for fn, want in expected.items():
            fp = os.path.join(modules_dir, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"warp module file missing: {fp}")
            got = _sha256(fp)
            if got != want:
                raise ValueError(f"SHA256 mismatch for warp module {fn}: expected {want}, got {got}")

    def run_torch(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Bind named torch inputs, replay the APIC graph, return named torch outputs."""
        for name, dstr in zip(self.input_names, self.input_dtypes):
            if name not in inputs:
                raise KeyError(f"warp node missing input '{name}'; got {list(inputs)}")
            t = inputs[name]
            expected = _STR_TO_TORCH[dstr]
            if t.dtype != expected:
                raise TypeError(
                    f"warp input '{name}' expected dtype {dstr}, got {t.dtype} "
                    "(no silent reinterpretation)")
            tt = t.to(self.device).contiguous().reshape(-1)
            self.graph.set_param(name, wp.from_torch(tt, dtype=_STR_TO_WARP_DTYPE[dstr]))

        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

        out = {}
        for name, dstr, shape in zip(self.output_names, self.output_dtypes, self.output_shapes):
            numel = int(math.prod(shape)) if shape else 1
            buf = wp.empty(numel, dtype=_STR_TO_WARP_DTYPE[dstr], device=self.device)
            self.graph.get_param(name, buf)  # raises on byte-size mismatch
            out[name] = wp.to_torch(buf).reshape(tuple(shape))
        return out
