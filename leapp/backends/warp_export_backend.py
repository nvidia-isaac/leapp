#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
"""PROTOTYPE: NVIDIA Warp export backend for LEAPP (Approach A — Warp as a peer node-kind).

A "warp" node's artifact is an APIC capture: ``<name>.wrp`` + ``<name>_modules/`` (compiled
kernels) produced by ``wp.capture_save`` over a coarse ``wp.ScopedCapture(apic=True)`` region
that may contain many kernel launches / ops (per Miles Macklin's feedback: one .wrp per region,
not per launch). A sidecar ``<name>.warpmeta.json`` records the named APIC bindings (= node I/O
names) plus their dtypes/shapes and the ``_modules/`` checksums, so the loader is self-contained.

This backend makes a warp node load+run inside the existing Python ``InferenceManager`` exactly
like a jit/onnx node: ``compiled_model(*torch_inputs) -> torch.Tensor | tuple``. The same .wrp is
the artifact a future C++ runtime would replay directly via the ``wp_apic_*`` C API (no ONNX).

Design stance (after code review): every divergence from the recorded contract FAILS LOUDLY
rather than silently producing a wrong-but-green result — input dtype mismatch, output
shape/size mismatch, missing/altered ``_modules/`` files, and CUDA-artifact-on-no-CUDA all raise.
Supported dtypes are limited to those in ``_STR_TO_WARP_DTYPE`` (extend as needed).
"""
import hashlib
import json
import math
import os
import shutil
from typing import Any, Dict, Tuple

import torch
import warp as wp

from leapp.backends.export_backend import ExportBackend

WARP_BACKEND = "warp"
_WARP_DTYPE_TO_STR = {wp.float32: "float32", wp.float64: "float64", wp.int32: "int32", wp.int64: "int64"}
_STR_TO_WARP_DTYPE = {v: k for k, v in _WARP_DTYPE_TO_STR.items()}
_STR_TO_TORCH = {"float32": torch.float32, "float64": torch.float64,
                 "int32": torch.int32, "int64": torch.int64}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _meta_path(wrp_path: str) -> str:
    return os.path.splitext(wrp_path)[0] + ".warpmeta.json"


def _modules_dir_for(wrp_path: str) -> str:
    return os.path.splitext(wrp_path)[0] + "_modules"


class _WarpGraphCallable:
    """Wraps a loaded APIC ``wp.Graph`` as a ``(*torch_inputs) -> torch`` callable.

    Binds inputs by name via ``set_param(wp.from_torch(...))``, replays with ``capture_launch``,
    and reads outputs back into declared-shape tensors with ``get_param(name, out)``. All
    dtype/shape divergences from the captured contract raise.
    """

    def __init__(self, graph, meta: Dict[str, Any], device: str):
        self.graph = graph
        self.device = device
        self.input_names = meta["inputs"]
        self.output_names = meta["outputs"]
        self.input_dtypes = meta["input_dtypes"]
        self.output_dtypes = meta["output_dtypes"]
        self.output_shapes = meta["output_shapes"]

    def __call__(self, *torch_inputs):
        if len(torch_inputs) != len(self.input_names):
            raise ValueError(
                f"warp node expected {len(self.input_names)} inputs {self.input_names}, "
                f"got {len(torch_inputs)}")

        for name, dstr, t in zip(self.input_names, self.input_dtypes, torch_inputs):
            expected = _STR_TO_TORCH.get(dstr)
            if expected is None:
                raise ValueError(f"warp node '{name}': unsupported declared dtype '{dstr}'")
            if t.dtype != expected:
                raise TypeError(
                    f"warp node input '{name}' expected dtype {dstr}, got {t.dtype}. "
                    "This backend does not reinterpret dtypes (would corrupt results).")
            tt = t.to(self.device).contiguous().reshape(-1)
            self.graph.set_param(name, wp.from_torch(tt, dtype=_STR_TO_WARP_DTYPE[dstr]))

        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

        outs = []
        for name, dstr, shape in zip(self.output_names, self.output_dtypes, self.output_shapes):
            numel = int(math.prod(shape)) if shape else 1
            buf = wp.empty(numel, dtype=_STR_TO_WARP_DTYPE[dstr], device=self.device)
            self.graph.get_param(name, buf)  # raises on byte-size mismatch
            outs.append(wp.to_torch(buf).reshape(tuple(shape)))
        return outs[0] if len(outs) == 1 else tuple(outs)


class WarpExportBackend(ExportBackend):
    """LEAPP export backend for NVIDIA Warp APIC ``.wrp`` artifacts."""

    def get_backend_model_type(self):
        return WARP_BACKEND

    def get_backend_metadata(self) -> dict:
        return {}

    def compile(self, m: torch.nn.Module = None) -> Any:
        # The APIC graph is captured at trace time (externally for this prototype); nothing to compile.
        return None

    def save(self, save_path: str) -> Tuple[str, str, str]:
        # BYOM-style: relocate a pre-captured .wrp (+ _modules/ + .warpmeta.json) into the bundle.
        model_path = self.backend_params.get("model_path")
        if not model_path:
            return None, None, None
        os.makedirs(save_path, exist_ok=True)
        dest = os.path.join(save_path, os.path.basename(model_path))
        if os.path.abspath(model_path) != os.path.abspath(dest):
            shutil.copy2(model_path, dest)
            mods = _modules_dir_for(model_path)
            if os.path.isdir(mods):
                shutil.copytree(mods, _modules_dir_for(dest), dirs_exist_ok=True)
            if os.path.exists(_meta_path(model_path)):
                shutil.copy2(_meta_path(model_path), _meta_path(dest))
            model_path = dest
        return model_path, _md5(model_path), _sha256(model_path)

    def load(self, model_path: str, sha256sum: str):
        actual = _sha256(model_path)
        if actual != sha256sum:
            raise ValueError(
                f"SHA256 checksum mismatch for {model_path}: expected {sha256sum}, got {actual}")

        meta_path = _meta_path(model_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"warp node metadata sidecar not found: {meta_path} "
                "(expected next to the .wrp; carries the APIC binding names/dtypes/shapes)")
        with open(meta_path) as f:
            meta = json.load(f)

        # Verify the COMPILED KERNELS (_modules/) — they are what actually executes.
        self._verify_modules(model_path, meta)

        device_type = meta.get("device_type", "cuda")
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"warp node {os.path.basename(model_path)} was captured for CUDA "
                "but no CUDA device is available; refusing to load (a CUDA .wrp cannot replay on CPU).")
        device = "cuda" if device_type == "cuda" else "cpu"

        graph = wp.capture_load(model_path, device=device)
        self.compiled_model = _WarpGraphCallable(graph, meta, device)
        self.runtime_device = device
        self.compiled_module = None

    @staticmethod
    def _verify_modules(model_path: str, meta: Dict[str, Any]):
        expected = meta.get("modules_sha256")
        if not expected:
            return  # nothing recorded (e.g. a kernel-less capture)
        modules_dir = _modules_dir_for(model_path)
        if not os.path.isdir(modules_dir):
            raise FileNotFoundError(
                f"warp node modules dir missing: {modules_dir} (compiled kernels not shipped)")
        for fn, want in expected.items():
            fp = os.path.join(modules_dir, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"warp node module file missing: {fp}")
            got = _sha256(fp)
            if got != want:
                raise ValueError(
                    f"SHA256 mismatch for warp module {fn}: expected {want}, got {got}")


def save_warp_node(graph, save_dir: str, node_name: str,
                   inputs: Dict[str, "wp.array"], outputs: Dict[str, "wp.array"]) -> dict:
    """Serialize a captured APIC graph as a LEAPP warp node and return its YAML ``models`` entry.

    ``inputs``/``outputs`` map node-port name -> the wp.array used at capture (defines the APIC
    named binding, byte size, dtype and shape). Writes ``<name>.wrp``, ``<name>_modules/`` and a
    self-describing ``<name>.warpmeta.json`` sidecar into ``save_dir``.
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, node_name)
    wp.capture_save(graph, base, inputs=inputs, outputs=outputs)
    wrp_path = base + ".wrp"

    def _dstr(arr):
        d = _WARP_DTYPE_TO_STR.get(arr.dtype)
        if d is None:
            raise ValueError(f"unsupported warp array dtype {arr.dtype} for node '{node_name}'")
        return d

    def _desc(name, arr):
        return {"name": name, "dtype": _dstr(arr), "shape": list(arr.shape), "type": "tensor"}

    modules_dir = base + "_modules"
    modules_sha256 = {}
    if os.path.isdir(modules_dir):
        for fn in sorted(os.listdir(modules_dir)):
            fp = os.path.join(modules_dir, fn)
            if os.path.isfile(fp):
                modules_sha256[fn] = _sha256(fp)

    # Self-describing sidecar consumed by WarpExportBackend.load().
    meta = {
        "inputs": list(inputs.keys()),
        "outputs": list(outputs.keys()),
        "input_dtypes": [_dstr(a) for a in inputs.values()],
        "output_dtypes": [_dstr(a) for a in outputs.values()],
        "output_shapes": [list(a.shape) for a in outputs.values()],
        "device_type": "cuda",
        "modules_dir": os.path.basename(modules_dir),
        "modules_sha256": modules_sha256,
    }
    with open(_meta_path(wrp_path), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "inputs": [_desc(n, a) for n, a in inputs.items()],
        "outputs": [_desc(n, a) for n, a in outputs.items()],
        "parameters": {
            "model_path": os.path.basename(wrp_path),
            "md5sum": _md5(wrp_path),
            "sha256sum": _sha256(wrp_path),
            "backend": WARP_BACKEND,
            "warp_version": wp.__version__,
            "modules_dir": os.path.basename(modules_dir),
            "modules_sha256": modules_sha256,  # multi-file integrity (verified at load)
            "device_type": "cuda",
        },
    }
