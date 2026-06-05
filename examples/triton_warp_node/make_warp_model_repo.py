#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Build a Triton model repo for a single Warp (APIC) node via the LEAPP-owned generator.

Captures a coarse Warp region (2 launches: ``out = relu(in*2 + 1)``) to a ``.wrp``, writes a tiny
LEAPP-style config for a one-node ``backend: warp`` graph, and runs
``leapp_runtimes.triton.create_triton_model_repo`` — i.e. the exact code path the deployment
runtime (and isaac_ros_deploy) uses. Used by the demos and tests in this folder.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import warp as wp
import yaml

# Make the repo-root importable so `leapp_runtimes` resolves however this is launched.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from leapp_runtimes.triton.create_triton_model_repo import create_triton_model_repo  # noqa: E402

N = 8
SCALE = 2.0
BIAS = 1.0
IN_NAME = "in"
OUT_NAME = "out"


@wp.kernel
def _affine_k(x: wp.array(dtype=wp.float32), scale: wp.float32, bias: wp.float32,
              out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * scale + bias


@wp.kernel
def _relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def capture_graph(out_dir, device="cuda:0"):
    """Capture the coarse warp region to ``<out_dir>/graph.wrp`` (+ modules + .warpmeta.json)."""
    os.makedirs(out_dir, exist_ok=True)
    wp.init()
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as cap:
        wp.launch(_affine_k, dim=N, inputs=[x, SCALE, BIAS], outputs=[y], device=device)
        wp.launch(_relu_k, dim=N, inputs=[y], outputs=[y], device=device)
    base = os.path.join(out_dir, "graph")
    wp.capture_save(cap.graph, base, inputs={IN_NAME: x}, outputs={OUT_NAME: y})
    mdir = base + "_modules"
    msha = {fn: _sha256(os.path.join(mdir, fn)) for fn in sorted(os.listdir(mdir))} if os.path.isdir(mdir) else {}
    meta = {"inputs": [IN_NAME], "outputs": [OUT_NAME], "input_dtypes": ["float32"],
            "output_dtypes": ["float32"], "output_shapes": [[N]], "device_type": "cuda",
            "modules_dir": "graph_modules", "modules_sha256": msha}
    with open(base + ".warpmeta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return base + ".wrp"


def build_repo(repo_dir):
    """Capture a warp node and generate a Triton repo (warp_node + ensemble) via the LEAPP generator.

    Returns the warp node's model-version dir (``<repo>/warp_node/1``).
    """
    import tempfile
    repo_dir = str(repo_dir)
    # Stage the capture OUTSIDE the repo — Triton treats every dir in the repo as a model.
    stage = tempfile.mkdtemp(prefix="warp_capture_")
    wrp = capture_graph(stage)

    def io(name):
        return {"name": name, "dtype": "float32", "shape": [N], "type": "tensor"}

    config = {
        "models": {
            "warp_node": {
                "inputs": [io(IN_NAME)], "outputs": [io(OUT_NAME)],
                "parameters": {"model_path": wrp, "backend": "warp",
                               "md5sum": "x", "sha256sum": _sha256(wrp)},
            }
        },
        "pipeline": {"data_flow": {}, "feedback_flow": {},
                     "inputs": {"warp_node": [IN_NAME]}, "outputs": {"warp_node": [OUT_NAME]}},
        "system information": {"leapp config version": "1.1"},
    }
    cfg_path = os.path.join(stage, "config.yaml")
    os.makedirs(repo_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    create_triton_model_repo(Path(cfg_path), Path(repo_dir))
    return os.path.join(repo_dir, "warp_node", "1")


def reference(x):
    import torch
    return torch.relu(x * SCALE + BIAS)


if __name__ == "__main__":
    import tempfile
    repo = tempfile.mkdtemp(prefix="warp_triton_repo_")
    build_repo(repo)
    print("built model repo at", repo)
    for root, _, files in os.walk(repo):
        for fn in sorted(files):
            print("  ", os.path.relpath(os.path.join(root, fn), repo))
