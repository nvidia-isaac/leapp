#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-step ensemble validation: a warp node BETWEEN two neighbor steps.

Part 1 (generator / real ONNX): builds real pre.onnx + warp .wrp + post.onnx, runs the PATCHED
`create_triton_model_repo.py`, and asserts it emits a correct `onnx -> warp -> onnx` ensemble
(onnxruntime_onnx steps + a warp python step + internal `_internal_*` tensors). Structural proof
the generator handles the real heterogeneous graph.

Part 2 (live / GPU-resident handoff): the dockerless PyTriton bundle ships ONLY the python backend,
so the ONNX neighbors can't be served here (they run in the real Triton container the deploy uses).
We therefore serve a live `python -> warp -> python` ensemble (python affine stand-ins exercise the
SAME Triton GPU-residency contract as onnx steps), infer end-to-end, validate numerics, and ASSERT
each internal step->step edge delivered a GPU tensor (is_cpu==False) — the multi-step GPU-resident
handoff the single-node test could not show.

Pipeline:  obs --[pre: *1.5 - 0.2]--> h --[warp: relu(x*2+1)]--> y --[post: *0.5]--> out

Run: /tmp/leapp-warp/venv/bin/python examples/triton_warp_node/multistep_ensemble.py
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = "/tmp/leapp-warp/gen"           # patched create_triton_model_repo.py + _warp_templates
sys.path.insert(0, GEN_DIR)
from create_triton_model_repo import create_triton_model_repo  # the PATCHED generator

VENV = "/tmp/leapp-warp/venv"
SP = f"{VENV}/lib/python3.12/site-packages"
TS_DIR = f"{SP}/pytriton/tritonserver"
N = 8
WORK = Path("/tmp/leapp-warp/multistep")


# ---------------- warp capture (binding names x -> y) ----------------
@wp.kernel
def _affine_k(x: wp.array(dtype=wp.float32), s: wp.float32, b: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s + b


@wp.kernel
def _relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    Path(base + ".warpmeta.json").write_text(json.dumps(meta, indent=2))
    return base + ".wrp"


def export_onnx_affine(path, in_name, out_name, scale, bias):
    class Affine(torch.nn.Module):
        def forward(self, t):
            return t * scale + bias
    torch.onnx.export(Affine().eval(), (torch.zeros(N),), str(path),
                      input_names=[in_name], output_names=[out_name], dynamo=False)


def reference(obs):
    h = obs * 1.5 - 0.2
    y = np.maximum(h * 2.0 + 1.0, 0.0)
    return y * 0.5


# ---------------- Part 1: generator emits onnx -> warp -> onnx ----------------
def part1_generator():
    print("\n========== PART 1: patched generator on a real onnx -> warp -> onnx graph ==========")
    stage = WORK / "p1_stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    wrp = capture_warp(str(stage), "x", "y")
    export_onnx_affine(stage / "pre.onnx", "obs", "h", 1.5, -0.2)
    export_onnx_affine(stage / "post.onnx", "z", "out", 0.5, 0.0)

    def io(name, dt="float32"):
        return {"name": name, "dtype": dt, "shape": [N], "type": "tensor"}

    cfg = {
        "models": {
            "pre": {"inputs": [io("obs")], "outputs": [io("h")],
                    "parameters": {"model_path": str(stage / "pre.onnx"), "backend": "onnx",
                                   "md5sum": "x", "sha256sum": _sha(stage / "pre.onnx")}},
            "warp_node": {"inputs": [io("x")], "outputs": [io("y")],
                          "parameters": {"model_path": wrp, "backend": "warp",
                                         "md5sum": "x", "sha256sum": _sha(wrp)}},
            "post": {"inputs": [io("z")], "outputs": [io("out")],
                     "parameters": {"model_path": str(stage / "post.onnx"), "backend": "onnx",
                                    "md5sum": "x", "sha256sum": _sha(stage / "post.onnx")}},
        },
        "pipeline": {"data_flow": {"pre/h": ["warp_node/x"], "warp_node/y": ["post/z"]},
                     "feedback_flow": {}, "inputs": {"pre": ["obs"]}, "outputs": {"post": ["out"]}},
        "system information": {"leapp config version": "1.1"},
    }
    cfg_path = stage / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    repo = WORK / "p1_repo"
    shutil.rmtree(repo, ignore_errors=True)
    create_triton_model_repo(cfg_path, repo)

    pre_cfg = (repo / "pre" / "config.pbtxt").read_text()
    post_cfg = (repo / "post" / "config.pbtxt").read_text()
    warp_cfg = (repo / "warp_node" / "config.pbtxt").read_text()
    ens_cfg = (repo / "ensemble" / "config.pbtxt").read_text()
    print("--- generated ensemble/config.pbtxt ---\n" + ens_cfg)

    checks = {
        "pre is onnxruntime_onnx": 'platform: "onnxruntime_onnx"' in pre_cfg,
        "post is onnxruntime_onnx": 'platform: "onnxruntime_onnx"' in post_cfg,
        "warp_node is python backend": 'backend: "python"' in warp_cfg,
        "warp_node GPU + no-CPU-force": "KIND_GPU" in warp_cfg and "FORCE_CPU_ONLY_INPUT_TENSORS" in warp_cfg,
        "ensemble has 3 steps": all(f'model_name: "{m}"' in ens_cfg for m in ("pre", "warp_node", "post")),
        "internal GPU tensors wired": "_internal_h" in ens_cfg and "_internal_y" in ens_cfg,
        "warp artifacts copied": (repo / "warp_node" / "1" / "graph.wrp").exists()
        and (repo / "warp_node" / "1" / "graph_modules").is_dir(),
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
    vd = repo / name / "1"
    vd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.path.join(HERE, "affine_passthrough_model.py"), vd / "model.py")
    (repo / name / "config.pbtxt").write_text(_PY_CFG.format(name=name, inn=inn, out=out, n=N, scale=scale, bias=bias))


def build_live_repo(repo):
    shutil.rmtree(repo, ignore_errors=True)
    _py_model_dir(repo, "pre", "obs", "h", 1.5, -0.2)
    _py_model_dir(repo, "post", "z", "out", 0.5, 0.0)
    wv = repo / "warp_node" / "1"
    wv.mkdir(parents=True, exist_ok=True)
    capture_warp(str(wv), "x", "y")
    shutil.copy2(os.path.join(HERE, "model.py"), wv / "model.py")
    shutil.copy2(os.path.join(HERE, "warp_apic_runtime.py"), wv / "warp_apic_runtime.py")
    (repo / "warp_node" / "config.pbtxt").write_text(_WARP_CFG.format(n=N))
    (repo / "ensemble" / "1").mkdir(parents=True, exist_ok=True)
    (repo / "ensemble" / "config.pbtxt").write_text(_ENS_CFG.format(n=N))


def part2_live():
    print("\n========== PART 2: live python -> warp -> python ensemble (GPU-resident) ==========")
    repo = WORK / "p2_repo"
    build_live_repo(repo)
    subprocess.run(["pkill", "-9", "-f", "tritonserver"], capture_output=True)
    time.sleep(1)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{TS_DIR}/lib:{SP}/torch/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = SP
    env["WARP_TRITON_SITE_PACKAGES"] = SP
    env["WARP_TRITON_DEBUG_DEVICE"] = "1"
    logpath = WORK / "p2_serve.log"
    log = open(logpath, "w")
    proc = subprocess.Popen(
        [f"{TS_DIR}/bin/tritonserver", f"--model-repository={repo}",
         f"--backend-directory={TS_DIR}/backends", "--http-port=8795", "--grpc-port=8796",
         "--metrics-port=8797", "--exit-timeout-secs=3"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    out = None
    try:
        import tritonclient.http as http
        client = http.InferenceServerClient("localhost:8795")
        ready = False
        for _ in range(90):
            if proc.poll() is not None:
                break
            try:
                if client.is_server_ready():
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            print("server not ready; see", logpath)
            return False
        obs = np.linspace(-1.0, 2.0, N, dtype=np.float32)
        inp = http.InferInput("obs", [N], "FP32")
        inp.set_data_from_numpy(obs)
        res = client.infer("ensemble", [inp], outputs=[http.InferRequestedOutput("out")])
        out = res.as_numpy("out")
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    obs = np.linspace(-1.0, 2.0, N, dtype=np.float32)
    ref = reference(obs)
    err = float(np.abs(out - ref).max())
    numeric_ok = np.allclose(out, ref, atol=1e-5)

    # GPU-resident handoff proof: the internal-edge consumers (warp_node, post) must report is_cpu=False.
    logtext = logpath.read_text()
    warp_gpu = "[warp_node] input 'x' is_cpu=False" in logtext
    post_gpu = "[post] input 'z' is_cpu=False" in logtext

    print("obs        :", obs)
    print("ensemble   :", out)
    print("reference  :", ref)
    print(f"max_abs_err = {err}")
    print(f"  [{'PASS' if numeric_ok else 'FAIL'}] end-to-end numerics (3-step ensemble)")
    print(f"  [{'PASS' if warp_gpu else 'FAIL'}] internal edge pre->warp delivered a GPU tensor (warp input is_cpu=False)")
    print(f"  [{'PASS' if post_gpu else 'FAIL'}] internal edge warp->post delivered a GPU tensor (post input is_cpu=False)")
    ok = numeric_ok and warp_gpu and post_gpu
    print("PART 2:", "PASS — live multi-step ensemble, GPU-resident internal handoffs" if ok else "FAIL")
    return ok


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    p1 = part1_generator()
    p2 = part2_live()
    print("\n================= SUMMARY =================")
    print(f"PART 1 (generator emits onnx->warp->onnx)        : {'PASS' if p1 else 'FAIL'}")
    print(f"PART 2 (live multi-step GPU-resident handoff)     : {'PASS' if p2 else 'FAIL'}")
    print("NOTE: a *live* onnxruntime-backend run needs the full Triton container (the dockerless"
          " PyTriton bundle ships only the python backend); Part 1 validates the generated onnx config,"
          " Part 2 validates the live GPU-resident multi-step handoff with python stand-ins.")
    return 0 if (p1 and p2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
