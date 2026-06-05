#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-step ensemble demo: a Warp node BETWEEN two neighbor steps.

Part 1 (generator / real ONNX) — OPTIONAL: if the patched ``create_triton_model_repo.py`` is
available (env ``LEAPP_WARP_GENERATOR_DIR``, default ``/tmp/leapp-warp/gen``; this is the
isaac_ros_deploy change, applied via ``create_triton_model_repo.warp.patch``), build real
pre.onnx + warp .wrp + post.onnx, run the generator, and assert it emits a correct
``onnx -> warp -> onnx`` ensemble. Skipped if that generator is not on this machine.

Part 2 (live / GPU-resident) — needs ``nvidia-pytriton``: the dockerless bundle ships only the
python backend, so the ONNX neighbors can't be served here (they run in the real Triton container
the deploy uses). We serve a live ``python -> warp -> python`` ensemble (python affine stand-ins
exercise the SAME Triton GPU-residency contract as onnx steps), infer end-to-end, validate
numerics, and assert each internal step->step edge delivered a GPU tensor (is_cpu=False).

Pipeline:  obs --[pre: *1.5 - 0.2]--> h --[warp: relu(x*2+1)]--> y --[post: *0.5]--> out

Run: python examples/triton_warp_node/multistep_ensemble.py   (Part 2 needs nvidia-pytriton)
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import warp as wp
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _triton_serve import TritonServer, discover_tritonserver

N = 8


@wp.kernel
def _affine_k(x: wp.array(dtype=wp.float32), s: wp.float32, b: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s + b


@wp.kernel
def _relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def capture_warp(version_dir, in_name="x", out_name="y"):
    os.makedirs(version_dir, exist_ok=True)
    wp.init()
    x = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    y = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    wp.load_module(device="cuda:0")
    with wp.ScopedCapture(device="cuda:0", force_module_load=True, apic=True) as cap:
        wp.launch(_affine_k, dim=N, inputs=[x, 2.0, 1.0], outputs=[y], device="cuda:0")
        wp.launch(_relu_k, dim=N, inputs=[y], outputs=[y], device="cuda:0")
    base = os.path.join(version_dir, "graph")
    wp.capture_save(cap.graph, base, inputs={in_name: x}, outputs={out_name: y})
    mdir = base + "_modules"
    msha = {fn: _sha(os.path.join(mdir, fn)) for fn in sorted(os.listdir(mdir))} if os.path.isdir(mdir) else {}
    meta = {"inputs": [in_name], "outputs": [out_name], "input_dtypes": ["float32"],
            "output_dtypes": ["float32"], "output_shapes": [[N]], "device_type": "cuda",
            "modules_dir": "graph_modules", "modules_sha256": msha}
    open(base + ".warpmeta.json", "w").write(json.dumps(meta, indent=2))
    return base + ".wrp"


def export_onnx_affine(path, in_name, out_name, scale, bias):
    class Affine(torch.nn.Module):
        def forward(self, t):
            return t * scale + bias
    torch.onnx.export(Affine().eval(), (torch.zeros(N),), str(path),
                      input_names=[in_name], output_names=[out_name], dynamo=False)


def reference(obs):
    return np.maximum((obs * 1.5 - 0.2) * 2.0 + 1.0, 0.0) * 0.5


# ---------------- Part 1: generator emits onnx -> warp -> onnx (optional) ----------------
def part1_generator(work):
    print("\n========== PART 1: patched generator on a real onnx -> warp -> onnx graph ==========")
    gen_dir = os.environ.get("LEAPP_WARP_GENERATOR_DIR", "/tmp/leapp-warp/gen")
    gen_py = os.path.join(gen_dir, "create_triton_model_repo.py")
    if not os.path.exists(gen_py):
        print(f"SKIP: patched generator not found at {gen_py} "
              "(apply create_triton_model_repo.warp.patch to isaac_ros_deploy and point "
              "LEAPP_WARP_GENERATOR_DIR at it).")
        return None
    spec = importlib.util.spec_from_file_location("patched_gen", gen_py)
    gen = importlib.util.module_from_spec(spec)
    sys.path.insert(0, gen_dir)
    spec.loader.exec_module(gen)

    stage = os.path.join(work, "p1_stage")
    os.makedirs(stage, exist_ok=True)
    wrp = capture_warp(stage, "x", "y")
    export_onnx_affine(os.path.join(stage, "pre.onnx"), "obs", "h", 1.5, -0.2)
    export_onnx_affine(os.path.join(stage, "post.onnx"), "z", "out", 0.5, 0.0)

    def io(name):
        return {"name": name, "dtype": "float32", "shape": [N], "type": "tensor"}

    cfg = {
        "models": {
            "pre": {"inputs": [io("obs")], "outputs": [io("h")],
                    "parameters": {"model_path": os.path.join(stage, "pre.onnx"), "backend": "onnx",
                                   "md5sum": "x", "sha256sum": _sha(os.path.join(stage, "pre.onnx"))}},
            "warp_node": {"inputs": [io("x")], "outputs": [io("y")],
                          "parameters": {"model_path": wrp, "backend": "warp",
                                         "md5sum": "x", "sha256sum": _sha(wrp)}},
            "post": {"inputs": [io("z")], "outputs": [io("out")],
                     "parameters": {"model_path": os.path.join(stage, "post.onnx"), "backend": "onnx",
                                    "md5sum": "x", "sha256sum": _sha(os.path.join(stage, "post.onnx"))}},
        },
        "pipeline": {"data_flow": {"pre/h": ["warp_node/x"], "warp_node/y": ["post/z"]},
                     "feedback_flow": {}, "inputs": {"pre": ["obs"]}, "outputs": {"post": ["out"]}},
        "system information": {"leapp config version": "1.1"},
    }
    from pathlib import Path
    cfg_path = os.path.join(stage, "config.yaml")
    open(cfg_path, "w").write(yaml.safe_dump(cfg, sort_keys=False))
    repo = os.path.join(work, "p1_repo")
    shutil.rmtree(repo, ignore_errors=True)
    gen.create_triton_model_repo(Path(cfg_path), Path(repo))

    ens = open(os.path.join(repo, "ensemble", "config.pbtxt")).read()
    checks = {
        "pre onnxruntime_onnx": 'platform: "onnxruntime_onnx"' in open(os.path.join(repo, "pre", "config.pbtxt")).read(),
        "post onnxruntime_onnx": 'platform: "onnxruntime_onnx"' in open(os.path.join(repo, "post", "config.pbtxt")).read(),
        "warp python backend": 'backend: "python"' in open(os.path.join(repo, "warp_node", "config.pbtxt")).read(),
        "ensemble 3 steps": all(f'model_name: "{m}"' in ens for m in ("pre", "warp_node", "post")),
        "internal tensors wired": "_internal_h" in ens and "_internal_y" in ens,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print("PART 1:", "PASS — generator emits a correct onnx->warp->onnx ensemble" if ok else "FAIL")
    return ok


# ---------------- Part 2: live python -> warp -> python (GPU-resident) ----------------
_PY_CFG = '''\
name: "{name}"
backend: "python"
max_batch_size: 0
input [ {{ name: "{inn}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
output [ {{ name: "{out}" data_type: TYPE_FP32 dims: [ {n} ] }} ]
instance_group [ {{ kind: KIND_GPU }} ]
parameters: {{ key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {{ string_value: "no" }} }}
parameters: {{ key: "scale" value: {{ string_value: "{scale}" }} }}
parameters: {{ key: "bias" value: {{ string_value: "{bias}" }} }}
'''
_WARP_CFG = '''\
name: "warp_node"
backend: "python"
max_batch_size: 0
input [ {{ name: "x" data_type: TYPE_FP32 dims: [ {n} ] }} ]
output [ {{ name: "y" data_type: TYPE_FP32 dims: [ {n} ] }} ]
instance_group [ {{ kind: KIND_GPU }} ]
parameters: {{ key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {{ string_value: "no" }} }}
'''
_ENS_CFG = '''\
platform: "ensemble"
max_batch_size: 0
input [ {{ name: "obs" data_type: TYPE_FP32 dims: [ {n} ] }} ]
output [ {{ name: "out" data_type: TYPE_FP32 dims: [ {n} ] }} ]
ensemble_scheduling {{
  step [
    {{ model_name: "pre" model_version: -1 input_map {{ key: "obs" value: "obs" }} output_map {{ key: "h" value: "_internal_h" }} }},
    {{ model_name: "warp_node" model_version: -1 input_map {{ key: "x" value: "_internal_h" }} output_map {{ key: "y" value: "_internal_y" }} }},
    {{ model_name: "post" model_version: -1 input_map {{ key: "z" value: "_internal_y" }} output_map {{ key: "out" value: "out" }} }}
  ]
}}
'''


def _py_model_dir(repo, name, inn, out, scale, bias):
    vd = os.path.join(repo, name, "1")
    os.makedirs(vd, exist_ok=True)
    shutil.copy2(os.path.join(HERE, "affine_passthrough_model.py"), os.path.join(vd, "model.py"))
    open(os.path.join(repo, name, "config.pbtxt"), "w").write(
        _PY_CFG.format(name=name, inn=inn, out=out, n=N, scale=scale, bias=bias))


def build_live_repo(repo):
    shutil.rmtree(repo, ignore_errors=True)
    _py_model_dir(repo, "pre", "obs", "h", 1.5, -0.2)
    _py_model_dir(repo, "post", "z", "out", 0.5, 0.0)
    wv = os.path.join(repo, "warp_node", "1")
    capture_warp(wv, "x", "y")
    shutil.copy2(os.path.join(HERE, "model.py"), os.path.join(wv, "model.py"))
    shutil.copy2(os.path.join(HERE, "warp_apic_runtime.py"), os.path.join(wv, "warp_apic_runtime.py"))
    open(os.path.join(repo, "warp_node", "config.pbtxt"), "w").write(_WARP_CFG.format(n=N))
    os.makedirs(os.path.join(repo, "ensemble", "1"), exist_ok=True)
    open(os.path.join(repo, "ensemble", "config.pbtxt"), "w").write(_ENS_CFG.format(n=N))


def part2_live(work):
    print("\n========== PART 2: live python -> warp -> python ensemble (GPU-resident) ==========")
    if discover_tritonserver() is None:
        print("SKIP: nvidia-pytriton (bundled tritonserver) not installed in this environment.")
        return None
    repo = os.path.join(work, "p2_repo")
    build_live_repo(repo)
    obs = np.linspace(-1.0, 2.0, N, dtype=np.float32)
    with TritonServer(repo, http_port=8810) as ts:
        out = ts.infer("ensemble", {"obs": obs}, ["out"])["out"]
        logtext = open(ts.log_path).read()

    ref = reference(obs)
    err = float(np.abs(out - ref).max())
    numeric_ok = np.allclose(out, ref, atol=1e-5)
    warp_gpu = "[warp_node] input 'x' is_cpu=False" in logtext
    post_gpu = "[post] input 'z' is_cpu=False" in logtext
    print("ensemble out:", out)
    print("reference   :", ref)
    print(f"max_abs_err = {err}")
    print(f"  [{'PASS' if numeric_ok else 'FAIL'}] end-to-end numerics (3-step ensemble)")
    print(f"  [{'PASS' if warp_gpu else 'FAIL'}] internal edge pre->warp delivered a GPU tensor")
    print(f"  [{'PASS' if post_gpu else 'FAIL'}] internal edge warp->post delivered a GPU tensor")
    ok = numeric_ok and warp_gpu and post_gpu
    print("PART 2:", "PASS — live multi-step ensemble, GPU-resident internal handoffs" if ok else "FAIL")
    return ok


def main():
    work = tempfile.mkdtemp(prefix="leapp_warp_multistep_")
    p1 = part1_generator(work)
    p2 = part2_live(work)
    print("\n================= SUMMARY =================")
    print(f"PART 1 (generator emits onnx->warp->onnx) : {'SKIP' if p1 is None else ('PASS' if p1 else 'FAIL')}")
    print(f"PART 2 (live multi-step GPU-resident)     : {'SKIP' if p2 is None else ('PASS' if p2 else 'FAIL')}")
    failed = [p for p in (p1, p2) if p is False]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
