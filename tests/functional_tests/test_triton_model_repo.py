#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the LEAPP-owned LEAPP-graph -> Triton model repository generator.

``leapp_runtimes.triton.create_triton_model_repo`` turns a LEAPP YAML + per-node artifacts into a
Triton model repository + ensemble. These tests assert the generated repo structure for ONNX-only,
warp-only, and mixed ``onnx -> warp -> onnx`` graphs, plus the generation-time dtype guard for warp
nodes. Generation needs torch + onnx (always available); the warp cases also need warp + CUDA.
"""
import hashlib
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)                                            # leapp_runtimes
sys.path.insert(0, os.path.join(_REPO, "examples", "triton_warp_node"))  # make_warp_model_repo

from leapp_runtimes.triton.create_triton_model_repo import (  # noqa: E402
    create_triton_model_repo, _generate_warp_model_config,
)


def _has_warp_cuda():
    try:
        import warp  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


_WARP_CUDA = _has_warp_cuda()
N = 8


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _io(name):
    return {"name": name, "dtype": "float32", "shape": [N], "type": "tensor"}


def _onnx_affine(path, in_name, out_name, scale, bias):
    class Affine(torch.nn.Module):
        def forward(self, t):
            return t * scale + bias
    torch.onnx.export(Affine().eval(), (torch.zeros(N),), str(path),
                      input_names=[in_name], output_names=[out_name], dynamo=False)


def _write_cfg(tmp_path, models, pipeline):
    cfg = {"models": models, "pipeline": pipeline, "system information": {"leapp config version": "1.1"}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def test_onnx_only_ensemble(tmp_path):
    """Two chained ONNX nodes generate onnxruntime_onnx steps + an ensemble (no warp/CUDA needed)."""
    _onnx_affine(tmp_path / "a.onnx", "x", "mid", 2.0, 0.0)
    _onnx_affine(tmp_path / "b.onnx", "mid", "y", 0.5, 0.0)
    models = {
        "a": {"inputs": [_io("x")], "outputs": [_io("mid")],
              "parameters": {"model_path": str(tmp_path / "a.onnx"), "backend": "onnx",
                             "md5sum": "x", "sha256sum": _sha(tmp_path / "a.onnx")}},
        "b": {"inputs": [_io("mid")], "outputs": [_io("y")],
              "parameters": {"model_path": str(tmp_path / "b.onnx"), "backend": "onnx",
                             "md5sum": "x", "sha256sum": _sha(tmp_path / "b.onnx")}},
    }
    pipeline = {"data_flow": {"a/mid": ["b/mid"]}, "feedback_flow": {},
                "inputs": {"a": ["x"]}, "outputs": {"b": ["y"]}}
    repo = tmp_path / "repo"
    create_triton_model_repo(_write_cfg(tmp_path, models, pipeline), repo)

    assert 'platform: "onnxruntime_onnx"' in (repo / "a" / "config.pbtxt").read_text()
    assert 'platform: "onnxruntime_onnx"' in (repo / "b" / "config.pbtxt").read_text()
    ens = (repo / "ensemble" / "config.pbtxt").read_text()
    assert 'platform: "ensemble"' in ens
    assert 'model_name: "a"' in ens and 'model_name: "b"' in ens
    assert "_internal_mid" in ens  # a/mid -> b/mid carried internally
    assert (repo / "a" / "1" / "model.onnx").exists()


@pytest.mark.skipif(not _WARP_CUDA, reason="warp node generation needs warp + CUDA to capture a .wrp")
def test_warp_node_generated(tmp_path):
    """A backend:warp node yields a python-backend model dir with the .wrp + templates copied in."""
    import make_warp_model_repo as twn
    stage = tmp_path / "stage"
    wrp = twn.capture_graph(str(stage))  # writes graph.wrp + graph_modules/ + graph.warpmeta.json
    models = {"warp_node": {"inputs": [_io("in")], "outputs": [_io("out")],
                            "parameters": {"model_path": wrp, "backend": "warp",
                                           "md5sum": "x", "sha256sum": _sha(wrp)}}}
    pipeline = {"data_flow": {}, "feedback_flow": {},
                "inputs": {"warp_node": ["in"]}, "outputs": {"warp_node": ["out"]}}
    repo = tmp_path / "repo"
    create_triton_model_repo(_write_cfg(tmp_path, models, pipeline), repo)

    cfg = (repo / "warp_node" / "config.pbtxt").read_text()
    assert 'backend: "python"' in cfg
    assert "KIND_GPU" in cfg and "FORCE_CPU_ONLY_INPUT_TENSORS" in cfg
    v1 = repo / "warp_node" / "1"
    assert (v1 / "graph.wrp").exists() and (v1 / "graph_modules").is_dir()
    assert (v1 / "model.py").exists() and (v1 / "warp_apic_runtime.py").exists()  # templates copied


@pytest.mark.skipif(not _WARP_CUDA, reason="onnx->warp->onnx needs warp + CUDA")
def test_onnx_warp_onnx_ensemble(tmp_path):
    """A mixed onnx -> warp -> onnx graph yields a 3-step ensemble wired by internal tensors."""
    import make_warp_model_repo as twn
    stage = tmp_path / "stage"
    wrp = twn.capture_graph(str(stage))  # bindings in/out
    _onnx_affine(tmp_path / "pre.onnx", "obs", "in", 1.5, -0.2)
    _onnx_affine(tmp_path / "post.onnx", "out", "act", 0.5, 0.0)
    models = {
        "pre": {"inputs": [_io("obs")], "outputs": [_io("in")],
                "parameters": {"model_path": str(tmp_path / "pre.onnx"), "backend": "onnx",
                               "md5sum": "x", "sha256sum": _sha(tmp_path / "pre.onnx")}},
        "warp_node": {"inputs": [_io("in")], "outputs": [_io("out")],
                      "parameters": {"model_path": wrp, "backend": "warp",
                                     "md5sum": "x", "sha256sum": _sha(wrp)}},
        "post": {"inputs": [_io("out")], "outputs": [_io("act")],
                 "parameters": {"model_path": str(tmp_path / "post.onnx"), "backend": "onnx",
                                "md5sum": "x", "sha256sum": _sha(tmp_path / "post.onnx")}},
    }
    pipeline = {"data_flow": {"pre/in": ["warp_node/in"], "warp_node/out": ["post/out"]},
                "feedback_flow": {}, "inputs": {"pre": ["obs"]}, "outputs": {"post": ["act"]}}
    repo = tmp_path / "repo"
    create_triton_model_repo(_write_cfg(tmp_path, models, pipeline), repo)

    assert 'platform: "onnxruntime_onnx"' in (repo / "pre" / "config.pbtxt").read_text()
    assert 'backend: "python"' in (repo / "warp_node" / "config.pbtxt").read_text()
    assert 'platform: "onnxruntime_onnx"' in (repo / "post" / "config.pbtxt").read_text()
    ens = (repo / "ensemble" / "config.pbtxt").read_text()
    assert all(f'model_name: "{m}"' in ens for m in ("pre", "warp_node", "post"))
    assert "_internal_in" in ens and "_internal_out" in ens


def test_warp_unsupported_dtype_raises():
    """Warp node I/O with a dtype the runtime can't handle fails at generation time, not at serve."""
    with pytest.raises(ValueError, match="does not support"):
        _generate_warp_model_config(
            "warp_node",
            [{"name": "in", "dtype": "bfloat16", "shape": [N]}],
            [{"name": "out", "dtype": "float32", "shape": [N]}],
        )
