"""PROTOTYPE: a mixed PyTorch + NVIDIA-Warp LEAPP graph, run via the Python InferenceManager.

Demonstrates Approach A ("Warp as a peer node-kind"): a LEAPP bundle that contains a torch
(jit) node AND a warp (APIC .wrp) node, wired by an ordinary data_flow edge, executed in order
by the existing InferenceManager — no ONNX wrapper involved. The warp node is a *coarse* region
(2 kernel launches captured into one .wrp), matching Miles Macklin's "one .wrp per region" guidance.

Pipeline:   obs --[encoder: jit]--> h --[warpmap: warp]--> y
            encoder(obs) = obs*3 - 1            (TorchScript, elementwise)
            warpmap(h)   = relu(h*SCALE + BIAS) (2 warp kernels in one APIC capture)

Run:  PYTHONPATH=/home/lgulich/Code/leapp python3.12 examples/warp_mixed_graph_prototype.py
Requires: warp-lang>=1.13 (APIC), torch (CUDA), a CUDA GPU.
"""
import os
import shutil
import tempfile

import torch
import warp as wp
import yaml

import leapp  # noqa: F401  (ensures local package import works)
from leapp import InferenceManager
from leapp.backends.warp_export_backend import save_warp_node

DEVICE = "cuda:0"
N = 8
SCALE = 2.0
BIAS = 0.5
BUNDLE = os.path.join(tempfile.gettempdir(), "leapp_warp_mixed", "torchwarp")


# ---------------- torch node (exported as TorchScript) ----------------
class Encoder(torch.nn.Module):
    def forward(self, obs):
        return obs * 3.0 - 1.0


# ---------------- warp node kernels (a coarse 2-launch region) ----------------
@wp.kernel
def affine_k(x: wp.array(dtype=wp.float32), scale: wp.float32, bias: wp.float32,
             out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * scale + bias


@wp.kernel
def relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_bundle():
    shutil.rmtree(BUNDLE, ignore_errors=True)
    os.makedirs(BUNDLE, exist_ok=True)
    wp.init()

    # --- 1. torch encoder -> encoder.pt (backend: jit) ---
    enc = torch.jit.script(Encoder().eval())
    enc_path = os.path.join(BUNDLE, "encoder.pt")
    enc.save(enc_path)
    encoder_node = {
        "inputs": [{"name": "obs", "dtype": "float32", "shape": [N], "type": "tensor"}],
        "outputs": [{"name": "h", "dtype": "float32", "shape": [N], "type": "tensor"}],
        "parameters": {"model_path": "encoder.pt", "md5sum": _md5(enc_path),
                       "sha256sum": _sha256(enc_path), "backend": "jit"},
    }

    # --- 2. warp coarse region -> warpmap.wrp (+ _modules/ + .warpmeta.json) (backend: warp) ---
    x = wp.zeros(N, dtype=wp.float32, device=DEVICE)
    y = wp.zeros(N, dtype=wp.float32, device=DEVICE)
    wp.load_module(device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as cap:
        wp.launch(affine_k, dim=N, inputs=[x, SCALE, BIAS], outputs=[y], device=DEVICE)
        wp.launch(relu_k, dim=N, inputs=[y], outputs=[y], device=DEVICE)
    warpmap_node = save_warp_node(cap.graph, BUNDLE, "warpmap", inputs={"x": x}, outputs={"y": y})

    # --- 3. assemble the LEAPP bundle YAML ---
    spec = {
        "models": {"encoder": encoder_node, "warpmap": warpmap_node},
        "pipeline": {
            "data_flow": {"encoder/h": ["warpmap/x"]},   # torch output -> warp input
            "feedback_flow": {},
            "inputs": {"encoder": ["obs"]},
            "outputs": {"warpmap": ["y"]},
        },
        "system information": {
            "leapp config version": "1.1", "leapp version": str(getattr(leapp, "__version__", "?")),
            "warp version": str(wp.__version__), "torch version": str(torch.__version__),
        },
    }
    yaml_path = os.path.join(BUNDLE, "torchwarp.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return yaml_path


def main():
    yaml_path = build_bundle()
    print("=== bundle ===")
    for f in sorted(os.listdir(BUNDLE)):
        print("  ", f)
    print("\n=== yaml ===")
    print(open(yaml_path).read())

    im = InferenceManager(yaml_path)
    print("inputs :", im.inputs)
    print("outputs:", im.outputs)

    # Deterministic input spanning negative->positive so relu clamps SOME (not all) values,
    # exercising both the affine kernel's arithmetic and the relu kernel's clamping.
    obs = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float32)
    out = im({"encoder/obs": obs})
    y = out["warpmap/y"]

    # eager reference for the WHOLE mixed pipeline
    h_ref = obs * 3.0 - 1.0
    y_ref = torch.relu(h_ref * SCALE + BIAS)

    err = float((y - y_ref).abs().max())
    matches = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    n_pos = int((y > 0).sum())
    n_zero = int((y == 0).sum())
    print(f"\nobs                       : {obs.detach().cpu().numpy()}")
    print(f"mixed-graph output y      : {y.detach().cpu().numpy()}")
    print(f"eager reference y_ref     : {y_ref.detach().cpu().numpy()}")
    print(f"max_abs_err = {err} | nonzero={n_pos} clamped={n_zero}")
    # Meaningful test: outputs must match AND be non-trivial (some positive, some relu-clamped).
    ok = matches and n_pos > 0 and n_zero > 0
    print("\nP1 RESULT:", "PASS — torch->warp mixed LEAPP graph round-trips via InferenceManager"
          if ok else f"FAIL (matches={matches}, nonzero={n_pos}, clamped={n_zero})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
