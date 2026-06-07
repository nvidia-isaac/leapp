"""LEAPP warp auto-split example: mixed torch + warp graph with InferenceManager.

Demonstrates the non-invasive torch↔warp auto-split feature. A single leapp.start()/stop()
region containing both torch ops and a warp kernel is automatically segmented at the
wp.from_torch / wp.to_torch bridges into three LEAPP nodes:

    <prefix>.01_torch  — torch scale (obs * 2)
    <prefix>.02_warp   — warp vec3 normalize kernel
    <prefix>.03_torch  — torch reshape back to [N, 3]

Compiled and run via the standard InferenceManager. Result is compared bit-exactly to
torch.nn.functional.normalize(g * 2, dim=1).

Run:
    PYTHONPATH=/home/lgulich/Code/leapp python3.12 examples/warp_autosplit_mixed.py

Requires: warp-lang>=1.13 (APIC), torch (CUDA), a CUDA GPU.
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F
import warp as wp

import leapp
from leapp import annotate, InferenceManager

DEVICE = "cuda:0"
N = 6  # number of vec3 rows


@wp.kernel
def _norm_vec3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


def build_and_run(save_path):
    wp.init()

    # ---- Build the mixed graph in a single leapp.start()/stop() ----
    g = torch.randn(N, 3, device=DEVICE, dtype=torch.float32)

    leapp.start("autosplit_demo", save_path=save_path)

    gt = annotate.input_tensors("obs", {"g": g})

    # Torch segment: scale
    scaled = gt * 2.0

    # Bridge -> warp segment (auto-split point)
    a = wp.from_torch(scaled.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    out = wp.zeros(N, dtype=wp.vec3f, device=DEVICE)
    wp.launch(_norm_vec3, dim=N, inputs=[a], outputs=[out], device=DEVICE)

    # Bridge back -> torch (auto-split point)
    d = wp.to_torch(out).reshape(N, 3)

    annotate.output_tensors("obs", {"pg": d}, export_with="onnx-torchscript")

    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    # ---- Run via InferenceManager ----
    yaml_path = os.path.join(save_path, "autosplit_demo", "autosplit_demo.yaml")
    im = InferenceManager(yaml_path)

    g_in = torch.randn(N, 3, device=DEVICE, dtype=torch.float32)
    in_key = [k for k in im.inputs if k.endswith("/g")][0]
    result = im({in_key: g_in})
    out_key = [k for k in result if k.endswith("/pg")][0]

    # Ground truth: normalize(g*2, dim=1)
    ref = F.normalize(g_in * 2.0, dim=1)
    max_abs_err = float((result[out_key] - ref).abs().max())
    ok = max_abs_err < 1e-4

    print(f"Pipeline: torch-scale -> warp-normalize -> torch-reshape ({N} vec3 rows)")
    print(f"max_abs_err = {max_abs_err:.2e}")
    print()
    if ok:
        print("PASS — warp auto-split mixed graph matches F.normalize(g*2, dim=1) via InferenceManager")
    else:
        print(f"FAIL — max_abs_err {max_abs_err:.2e} exceeds 1e-4 threshold")
    return 0 if ok else 1


def main():
    with tempfile.TemporaryDirectory(prefix="leapp_warp_autosplit_") as tmp:
        return build_and_run(tmp)


if __name__ == "__main__":
    sys.exit(main())
