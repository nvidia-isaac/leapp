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
"""NVIDIA Warp export backend for LEAPP (Approach A — Warp as a peer node-kind).

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
Supported dtypes are defined in ``leapp.backends.warp_dtypes._REGISTRY`` (extend there).
"""
import hashlib
import json
import math
import os
import shutil
from contextlib import nullcontext
from typing import Any, Dict, Tuple

import torch
import warp as wp

from leapp.backends.export_backend import ExportBackend
from leapp.backends import warp_dtypes as wd

WARP_BACKEND = "warp"
_STR_TO_TORCH = {
    name: getattr(torch, wd.scalar_base_str(name))
    for name in wd._REGISTRY
    if hasattr(torch, wd.scalar_base_str(name))
}


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
        # Warp struct dtypes (e.g. "vec3f"); equal to the scalar base for plain scalars.
        self.input_warp_dtypes = meta.get("input_warp_dtypes", meta["input_dtypes"])
        self.output_warp_dtypes = meta.get("output_warp_dtypes", meta["output_dtypes"])
        # APIC binding names used in set_param / get_param. These are prefixed to avoid
        # collisions when a LEAPP input port and output port share the same name (e.g. both
        # named "out0"). Falls back to the plain port names for older .wrp files without
        # apic_input_names / apic_output_names in the meta.
        self.apic_input_names = meta.get("apic_input_names", self.input_names)
        self.apic_output_names = meta.get("apic_output_names", self.output_names)

    def __call__(self, *torch_inputs):
        if len(torch_inputs) != len(self.input_names):
            raise ValueError(
                f"warp node expected {len(self.input_names)} inputs {self.input_names}, "
                f"got {len(torch_inputs)}")

        # Bind/launch/read-back on the producer's CUDA stream (torch's current stream) so the
        # whole region is ordered on one stream; sync only that stream, not the whole device.
        torch_stream = torch.cuda.current_stream(self.device) if str(self.device).startswith("cuda") else None
        wp_stream = wp.Stream(self.device, cuda_stream=torch_stream.cuda_stream) if torch_stream else None
        ctx = wp.ScopedStream(wp_stream) if wp_stream else nullcontext()

        outs = []
        with ctx:
            for apic_name, leapp_name, base_dstr, wdstr, t in zip(
                    self.apic_input_names, self.input_names,
                    self.input_dtypes, self.input_warp_dtypes, torch_inputs):
                expected = _STR_TO_TORCH.get(base_dstr)
                if expected is None:
                    raise ValueError(f"warp node '{leapp_name}': unsupported declared dtype '{base_dstr}'")
                if t.dtype != expected:
                    raise TypeError(
                        f"warp node input '{leapp_name}' expected dtype {base_dstr}, got {t.dtype}. "
                        "No silent reinterpretation — pass the correct dtype to avoid corrupting results.")
                count = wd.scalar_count(wdstr)
                trailing = wd.trailing_shape(wdstr)
                tt = t.to(self.device).contiguous()
                tt = tt.reshape(-1) if count == 1 else tt.reshape((-1,) + trailing)
                self.graph.set_param(apic_name, wp.from_torch(tt, dtype=wd.str_to_warp_dtype(wdstr)))

            wp.capture_launch(self.graph, stream=wp_stream)

            for apic_name, base_dstr, wdstr, shape in zip(
                    self.apic_output_names,
                    self.output_dtypes, self.output_warp_dtypes, self.output_shapes):
                numel = int(math.prod(shape)) if shape else 1
                n_elements = numel // wd.scalar_count(wdstr)  # exact: shape includes trailing dims
                buf = wp.empty(n_elements, dtype=wd.str_to_warp_dtype(wdstr), device=self.device)
                self.graph.get_param(apic_name, buf)  # raises on byte-size mismatch
                outs.append(wp.to_torch(buf).reshape(tuple(shape)))

        if wp_stream:
            wp.synchronize_stream(wp_stream)
        else:
            wp.synchronize_device(self.device)
        return outs[0] if len(outs) == 1 else tuple(outs)


class WarpExportBackend(ExportBackend):
    """LEAPP export backend for NVIDIA Warp APIC ``.wrp`` artifacts."""

    def get_backend_model_type(self):
        return WARP_BACKEND

    def get_backend_metadata(self) -> dict:
        return {}

    def compile(self, m: torch.nn.Module = None) -> Any:
        # The APIC graph is built and loaded by WarpRegionNode.compile_model() (or the standalone warp_node capture); there is nothing to compile here.
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
    # Use name-prefixed APIC binding names to avoid collisions when a LEAPP input port and
    # output port share the same name (e.g. both "out0" in an auto-split warp segment). The
    # APIC names are stored in the meta under apic_input_names / apic_output_names and used
    # by _WarpGraphCallable for set_param / get_param.
    apic_inputs = {f"_in_{n}": a for n, a in inputs.items()}
    apic_outputs = {f"_out_{n}": a for n, a in outputs.items()}
    wp.capture_save(graph, base, inputs=apic_inputs, outputs=apic_outputs)
    wrp_path = base + ".wrp"

    def _wstr(arr):
        return wd.warp_dtype_to_str(arr.dtype)

    def _torch_view_shape(arr):
        return list(arr.shape) + list(wd.trailing_shape(wd.warp_dtype_to_str(arr.dtype)))

    def _desc(name, arr):
        wdstr = _wstr(arr)
        return {"name": name, "dtype": wd.scalar_base_str(wdstr),
                "shape": _torch_view_shape(arr), "warp_dtype": wdstr, "type": "tensor"}

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
        # APIC binding names (prefixed) for set_param / get_param — distinct from LEAPP port names.
        "apic_input_names": list(apic_inputs.keys()),
        "apic_output_names": list(apic_outputs.keys()),
        "input_dtypes": [wd.scalar_base_str(_wstr(a)) for a in inputs.values()],
        "output_dtypes": [wd.scalar_base_str(_wstr(a)) for a in outputs.values()],
        "input_warp_dtypes": [_wstr(a) for a in inputs.values()],
        "output_warp_dtypes": [_wstr(a) for a in outputs.values()],
        "output_shapes": [_torch_view_shape(a) for a in outputs.values()],
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
