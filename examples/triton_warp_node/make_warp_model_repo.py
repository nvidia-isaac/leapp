#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Build a Triton model repository for a Warp (APIC) node + a single-step ensemble.

Captures a coarse Warp region (2 launches: ``out = relu(in*2 + 1)``) to ``graph.wrp`` and lays out
exactly what `create_triton_model_repo.py`'s `_create_warp_model_dir` should emit:

    <repo>/warp_node/config.pbtxt          (backend:"python", KIND_GPU, FORCE_CPU_ONLY_INPUT_TENSORS:"no")
    <repo>/warp_node/1/model.py            (the python-backend model)
    <repo>/warp_node/1/warp_apic_runtime.py
    <repo>/warp_node/1/graph.wrp
    <repo>/warp_node/1/graph.warpmeta.json
    <repo>/warp_node/1/graph_modules/
    <repo>/ensemble/config.pbtxt           (platform:"ensemble", one step -> warp_node)
    <repo>/ensemble/1/                      (empty)

Self-contained: the deploy side does NOT depend on the leapp export package.
"""
import hashlib
import json
import os
import shutil

import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
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


WARP_CONFIG_PBTXT = '''\
backend: "python"
max_batch_size: 0
input [ {{ name: "{inn}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
output [ {{ name: "{out}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
instance_group [ {{ kind: KIND_GPU }} ]
parameters: {{ key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {{ string_value: "no" }} }}
'''

ENSEMBLE_CONFIG_PBTXT = '''\
platform: "ensemble"
max_batch_size: 0
input [ {{ name: "{inn}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
output [ {{ name: "{out}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
ensemble_scheduling {{
  step [
    {{
      model_name: "warp_node"
      model_version: -1
      input_map {{ key: "{inn}" value: "{inn}" }}
      output_map {{ key: "{out}" value: "{out}" }}
    }}
  ]
}}
'''


def capture_graph(version_dir, device="cuda:0"):
    wp.init()
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as cap:
        wp.launch(_affine_k, dim=N, inputs=[x, SCALE, BIAS], outputs=[y], device=device)
        wp.launch(_relu_k, dim=N, inputs=[y], outputs=[y], device=device)
    base = os.path.join(version_dir, "graph")
    wp.capture_save(cap.graph, base, inputs={IN_NAME: x}, outputs={OUT_NAME: y})

    modules_dir = base + "_modules"
    modules_sha256 = {}
    if os.path.isdir(modules_dir):
        for fn in sorted(os.listdir(modules_dir)):
            fp = os.path.join(modules_dir, fn)
            if os.path.isfile(fp):
                modules_sha256[fn] = _sha256(fp)
    meta = {
        "inputs": [IN_NAME], "outputs": [OUT_NAME],
        "input_dtypes": ["float32"], "output_dtypes": ["float32"],
        "output_shapes": [[N]], "device_type": "cuda",
        "modules_dir": "graph_modules", "modules_sha256": modules_sha256,
    }
    with open(base + ".warpmeta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_repo(repo_dir):
    shutil.rmtree(repo_dir, ignore_errors=True)
    wn_v1 = os.path.join(repo_dir, "warp_node", "1")
    os.makedirs(wn_v1, exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "ensemble", "1"), exist_ok=True)

    # capture artifacts into the version dir
    capture_graph(wn_v1)
    # backend code
    shutil.copy2(os.path.join(HERE, "model.py"), os.path.join(wn_v1, "model.py"))
    shutil.copy2(os.path.join(HERE, "warp_apic_runtime.py"), os.path.join(wn_v1, "warp_apic_runtime.py"))
    # configs
    with open(os.path.join(repo_dir, "warp_node", "config.pbtxt"), "w") as f:
        f.write(WARP_CONFIG_PBTXT.format(inn=IN_NAME, out=OUT_NAME, n=N))
    with open(os.path.join(repo_dir, "ensemble", "config.pbtxt"), "w") as f:
        f.write(ENSEMBLE_CONFIG_PBTXT.format(inn=IN_NAME, out=OUT_NAME, n=N))
    return wn_v1


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
