#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Functional tests for the prototype Warp (APIC) node-kind in LEAPP.

Guarded: skipped unless ``warp-lang`` (with APIC) and a CUDA GPU are available.
Validates that a warp ``.wrp`` node loads + runs inside the existing InferenceManager,
both standalone and wired downstream of a torch (jit) node.
"""
import os

import pytest
import torch
import yaml

wp = pytest.importorskip("warp", reason="warp-lang not installed")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warp APIC node requires a CUDA GPU")

from leapp import InferenceManager  # noqa: E402
from leapp.backends.warp_export_backend import save_warp_node  # noqa: E402

DEVICE = "cuda:0"
N = 8
SCALE = 2.0
BIAS = 0.5


@wp.kernel
def _affine_k(x: wp.array(dtype=wp.float32), scale: wp.float32, bias: wp.float32,
              out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * scale + bias


@wp.kernel
def _relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def _capture_warp_node(save_dir, name):
    wp.init()
    x = wp.zeros(N, dtype=wp.float32, device=DEVICE)
    y = wp.zeros(N, dtype=wp.float32, device=DEVICE)
    wp.load_module(device=DEVICE)
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as cap:
        wp.launch(_affine_k, dim=N, inputs=[x, SCALE, BIAS], outputs=[y], device=DEVICE)
        wp.launch(_relu_k, dim=N, inputs=[y], outputs=[y], device=DEVICE)
    return save_warp_node(cap.graph, save_dir, name, inputs={"x": x}, outputs={"y": y})


def _write_yaml(path, spec):
    with open(path, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)


def test_warp_only_node_round_trips(tmp_path):
    """A single coarse warp node (2 launches in one .wrp) runs via InferenceManager."""
    bundle = str(tmp_path)
    node = _capture_warp_node(bundle, "warpmap")
    spec = {
        "models": {"warpmap": node},
        "pipeline": {"data_flow": {}, "feedback_flow": {},
                     "inputs": {"warpmap": ["x"]}, "outputs": {"warpmap": ["y"]}},
        "system information": {"leapp config version": "1.1"},
    }
    yaml_path = os.path.join(bundle, "g.yaml")
    _write_yaml(yaml_path, spec)

    im = InferenceManager(yaml_path)
    assert im.inputs == ["warpmap/x"]
    assert im.outputs == ["warpmap/y"]

    x = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float32)
    y = im({"warpmap/x": x})["warpmap/y"]
    y_ref = torch.relu(x * SCALE + BIAS)
    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    # non-trivial: relu clamps some but not all
    assert int((y > 0).sum()) > 0 and int((y == 0).sum()) > 0


def test_mixed_torch_warp_graph(tmp_path):
    """torch (jit) node -> warp (.wrp) node, wired by data_flow, run end-to-end."""
    bundle = str(tmp_path)

    class Encoder(torch.nn.Module):
        def forward(self, obs):
            return obs * 3.0 - 1.0

    import hashlib
    enc_path = os.path.join(bundle, "encoder.pt")
    torch.jit.script(Encoder().eval()).save(enc_path)
    enc_bytes = open(enc_path, "rb").read()
    encoder_node = {
        "inputs": [{"name": "obs", "dtype": "float32", "shape": [N], "type": "tensor"}],
        "outputs": [{"name": "h", "dtype": "float32", "shape": [N], "type": "tensor"}],
        "parameters": {"model_path": "encoder.pt",
                       "md5sum": hashlib.md5(enc_bytes).hexdigest(),
                       "sha256sum": hashlib.sha256(enc_bytes).hexdigest(), "backend": "jit"},
    }
    warp_node = _capture_warp_node(bundle, "warpmap")
    spec = {
        "models": {"encoder": encoder_node, "warpmap": warp_node},
        "pipeline": {"data_flow": {"encoder/h": ["warpmap/x"]}, "feedback_flow": {},
                     "inputs": {"encoder": ["obs"]}, "outputs": {"warpmap": ["y"]}},
        "system information": {"leapp config version": "1.1"},
    }
    yaml_path = os.path.join(bundle, "g.yaml")
    _write_yaml(yaml_path, spec)

    im = InferenceManager(yaml_path)
    obs = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float32)
    y = im({"encoder/obs": obs})["warpmap/y"]
    y_ref = torch.relu((obs * 3.0 - 1.0) * SCALE + BIAS)
    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-5)
    assert int((y > 0).sum()) > 0 and int((y == 0).sum()) > 0


def test_warp_backend_checksum_mismatch_raises(tmp_path):
    """Tampering with the .wrp must fail the sha256 gate at load (no silent corruption)."""
    bundle = str(tmp_path)
    node = _capture_warp_node(bundle, "warpmap")
    spec = {
        "models": {"warpmap": node},
        "pipeline": {"data_flow": {}, "feedback_flow": {},
                     "inputs": {"warpmap": ["x"]}, "outputs": {"warpmap": ["y"]}},
        "system information": {"leapp config version": "1.1"},
    }
    # Corrupt the recorded checksum so load() must reject it.
    spec["models"]["warpmap"]["parameters"]["sha256sum"] = "0" * 64
    yaml_path = os.path.join(bundle, "g.yaml")
    _write_yaml(yaml_path, spec)
    with pytest.raises(ValueError, match="SHA256 checksum mismatch"):
        InferenceManager(yaml_path)


def test_warp_module_tamper_raises(tmp_path):
    """Altering a compiled kernel in _modules/ must fail the modules_sha256 gate at load."""
    bundle = str(tmp_path)
    node = _capture_warp_node(bundle, "warpmap")
    # Corrupt one compiled-kernel file (the code that actually runs on the GPU).
    mod_dir = os.path.join(bundle, "warpmap_modules")
    victim = os.path.join(mod_dir, sorted(os.listdir(mod_dir))[0])
    with open(victim, "ab") as f:
        f.write(b"\x00tampered")
    spec = {
        "models": {"warpmap": node},
        "pipeline": {"data_flow": {}, "feedback_flow": {},
                     "inputs": {"warpmap": ["x"]}, "outputs": {"warpmap": ["y"]}},
        "system information": {"leapp config version": "1.1"},
    }
    yaml_path = os.path.join(bundle, "g.yaml")
    _write_yaml(yaml_path, spec)
    with pytest.raises(ValueError, match="SHA256 mismatch for warp module"):
        InferenceManager(yaml_path)


def test_warp_input_dtype_mismatch_raises(tmp_path):
    """Feeding a non-declared dtype must raise (no silent byte-reinterpretation)."""
    bundle = str(tmp_path)
    node = _capture_warp_node(bundle, "warpmap")
    spec = {
        "models": {"warpmap": node},
        "pipeline": {"data_flow": {}, "feedback_flow": {},
                     "inputs": {"warpmap": ["x"]}, "outputs": {"warpmap": ["y"]}},
        "system information": {"leapp config version": "1.1"},
    }
    yaml_path = os.path.join(bundle, "g.yaml")
    _write_yaml(yaml_path, spec)
    im = InferenceManager(yaml_path)
    bad = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float64)  # declared is float32
    with pytest.raises(TypeError, match="expected dtype float32"):
        im({"warpmap/x": bad})
