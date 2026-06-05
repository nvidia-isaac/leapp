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
from contextlib import nullcontext
from typing import Dict, List

import torch
import warp as wp

_STR_TO_WARP_DTYPE = {
    "float16": wp.float16, "float32": wp.float32, "float64": wp.float64,
    "int8": wp.int8, "int16": wp.int16, "int32": wp.int32, "int64": wp.int64,
    "uint8": wp.uint8, "bool": wp.bool,
}
_STR_TO_TORCH = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "uint8": torch.uint8, "bool": torch.bool,
}
_DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "int8": 1, "int16": 2,
                "int32": 4, "int64": 8, "uint8": 1, "bool": 1}


def _require_dtype(dstr):
    if dstr not in _STR_TO_WARP_DTYPE:
        raise ValueError(
            f"warp node dtype '{dstr}' not supported by this runtime "
            f"(supported: {sorted(_STR_TO_WARP_DTYPE)})")
    return dstr


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
        """Bind named torch inputs, replay the APIC graph, return named torch outputs.

        Stream-correct (H1): bind, launch, and read back all on the *producer's* CUDA stream
        (torch's current stream), so the torch producing work -> set_param D2D -> capture_launch ->
        get_param read-back are ordered on one stream (no cross-stream race), and we sync only that
        stream (not the whole device, which would defeat KIND_GPU instance parallelism).
        """
        params = self.graph.params
        torch_stream = torch.cuda.current_stream(self.device) if self.device.startswith("cuda") else None
        wp_stream = wp.Stream(self.device, cuda_stream=torch_stream.cuda_stream) if torch_stream else None

        ctx = wp.ScopedStream(wp_stream) if wp_stream else nullcontext()
        with ctx:
            for name, dstr in zip(self.input_names, self.input_dtypes):
                if name not in inputs:
                    raise KeyError(f"warp node missing input '{name}'; got {list(inputs)}")
                t = inputs[name]
                if t.dtype != _STR_TO_TORCH[_require_dtype(dstr)]:
                    raise TypeError(
                        f"warp input '{name}' expected dtype {dstr}, got {t.dtype} "
                        "(no silent reinterpretation)")
                # I1: reject a batched/wrong-size input loudly (region byte size is fixed at capture).
                expected_numel = params[name]["size"] // _DTYPE_BYTES[dstr]
                if t.numel() != expected_numel:
                    raise ValueError(
                        f"warp input '{name}' expected {expected_numel} elements (this node is "
                        f"non-batching), got {t.numel()} (shape {tuple(t.shape)})")
                tt = t.to(self.device).contiguous().reshape(-1)
                self.graph.set_param(name, wp.from_torch(tt, dtype=_STR_TO_WARP_DTYPE[dstr]))

            wp.capture_launch(self.graph, stream=wp_stream)

            out = {}
            for name, dstr, shape in zip(self.output_names, self.output_dtypes, self.output_shapes):
                numel = int(math.prod(shape)) if shape else 1
                buf = wp.empty(numel, dtype=_STR_TO_WARP_DTYPE[_require_dtype(dstr)], device=self.device)
                self.graph.get_param(name, buf)  # raises on byte-size mismatch
                out[name] = wp.to_torch(buf).reshape(tuple(shape))

        if wp_stream:
            wp.synchronize_stream(wp_stream)
        else:
            wp.synchronize_device(self.device)
        return out
