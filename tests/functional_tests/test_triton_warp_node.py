#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for running a LEAPP Warp (APIC) node as a Triton python-backend ensemble step.

Two tiers, both GPU/warp-guarded:
  * Tier A (no server) — exercises the real ``leapp_runtimes/triton/warp_node/model.py`` ``execute()``
    path via a faithful ``triton_python_backend_utils`` (pb_utils) DLPack mock, plus the runtime's
    loud-failure guards. Needs only ``warp`` + ``torch`` + CUDA.
  * Tier B (live) — serves a real single-node Triton ensemble via the PyTriton-bundled
    ``tritonserver`` (discovered from the installed ``pytriton`` package) and infers over HTTP.
    Skipped cleanly when ``pytriton`` is not importable, so the suite stays CI-safe.

Run (default repo env): ``PYTHONPATH=<repo> python3.12 -m pytest tests/functional_tests/test_triton_warp_node.py -q``
Tier B additionally runs when executed under a venv that has ``nvidia-pytriton`` installed.
"""
import importlib.util
import os
import signal
import subprocess
import sys
import time
import types

import pytest

wp = pytest.importorskip("warp", reason="warp-lang not installed")
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warp APIC node requires a CUDA GPU")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EXAMPLES = os.path.join(_REPO, "examples", "triton_warp_node")
sys.path.insert(0, _REPO)        # leapp_runtimes
sys.path.insert(0, _EXAMPLES)    # make_warp_model_repo (demo repo builder)

import make_warp_model_repo as twn  # noqa: E402  (capture + repo builder)
from leapp_runtimes.triton.warp_node.warp_apic_runtime import WarpApicRunner  # noqa: E402


# --------------------------- pb_utils DLPack mock (Tier A) ---------------------------
class _FakeTensor:
    def __init__(self, name, t):
        self._name, self._t = name, t

    def name(self):
        return self._name

    def is_cpu(self):
        return not self._t.is_cuda

    def to_dlpack(self):
        return torch.utils.dlpack.to_dlpack(self._t)

    @staticmethod
    def from_dlpack(name, capsule):
        return _FakeTensor(name, torch.from_dlpack(capsule))

    def torch_tensor(self):
        return self._t


class _FakeResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


class _FakeRequest:
    def __init__(self, tensors):
        self._by_name = {t.name(): t for t in tensors}


def _install_fake_pb_utils():
    m = types.ModuleType("triton_python_backend_utils")
    m.Tensor = _FakeTensor
    m.InferenceRequest = _FakeRequest
    m.InferenceResponse = _FakeResponse
    m.get_input_tensor_by_name = lambda req, name: req._by_name[name]
    sys.modules["triton_python_backend_utils"] = m


def _load_model_py(version_dir):
    spec = importlib.util.spec_from_file_location(
        "warp_triton_model_under_test", os.path.join(version_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, version_dir)
    spec.loader.exec_module(mod)
    return mod


def test_warp_python_backend_model_standalone(tmp_path):
    """The real model.py execute() path round-trips on GPU via DLPack (no Triton server)."""
    repo = str(tmp_path / "repo")
    twn.build_repo(repo)
    version_dir = os.path.join(repo, "warp_node", "1")

    _install_fake_pb_utils()
    model = _load_model_py(version_dir).TritonPythonModel()
    model.initialize({"model_repository": repo, "model_version": "1", "model_name": "warp_node",
                      "model_instance_kind": "GPU", "model_instance_device_id": "0"})

    x = torch.linspace(-1.0, 2.0, twn.N, device="cuda", dtype=torch.float32)
    [resp] = model.execute([_FakeRequest([_FakeTensor(twn.IN_NAME, x)])])
    out = {t.name(): t.torch_tensor() for t in resp.output_tensors}[twn.OUT_NAME]

    ref = torch.relu(x * twn.SCALE + twn.BIAS)
    assert out.is_cuda                                  # output stayed on GPU
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
    assert int((out > 0).sum()) > 0 and int((out == 0).sum()) > 0  # non-trivial


def _make_runner(tmp_path):
    version_dir = tmp_path / "warp_node" / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    twn.capture_graph(str(version_dir))
    return WarpApicRunner(str(version_dir / "graph.wrp"))


def test_warp_runtime_rejects_wrong_dtype(tmp_path):
    runner = _make_runner(tmp_path)
    bad = torch.linspace(-1, 2, twn.N, device="cuda", dtype=torch.float64)  # declared float32
    with pytest.raises(TypeError, match="expected dtype float32"):
        runner.run_torch({twn.IN_NAME: bad})


def test_warp_runtime_rejects_batched_input(tmp_path):
    runner = _make_runner(tmp_path)
    batched = torch.zeros(4, twn.N, device="cuda", dtype=torch.float32)  # non-batching node
    with pytest.raises(ValueError, match="non-batching"):
        runner.run_torch({twn.IN_NAME: batched})


# --------------------------- Tier B: live Triton ensemble ---------------------------
def _discover_tritonserver():
    try:
        import pytriton  # noqa: F401
    except Exception:
        return None
    ts = os.path.join(os.path.dirname(pytriton.__file__), "tritonserver")
    binp = os.path.join(ts, "bin", "tritonserver")
    return ts if os.path.exists(binp) else None

_TS_DIR = _discover_tritonserver()


@pytest.mark.skipif(_TS_DIR is None, reason="nvidia-pytriton (bundled tritonserver) not installed")
def test_warp_node_live_triton_ensemble(tmp_path):
    """Serve a real single-node Triton ensemble with the warp .wrp and infer over HTTP."""
    import numpy as np
    import tritonclient.http as http

    # Make the python-backend stub for this interpreter's version discoverable.
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    stub_dst = os.path.join(_TS_DIR, "backends", "python", "triton_python_backend_stub")
    stub_src = os.path.join(_TS_DIR, "python_backend_stubs", pyver, "triton_python_backend_stub")
    if not os.path.exists(stub_dst):
        if not os.path.exists(stub_src):
            pytest.skip(f"no python_backend_stub for py{pyver} in pytriton bundle")
        import shutil
        shutil.copy2(stub_src, stub_dst)
        os.chmod(stub_dst, 0o755)

    repo = str(tmp_path / "repo")
    twn.build_repo(repo)

    sp = os.path.dirname(os.path.dirname(torch.__file__))     # this interpreter's site-packages
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{_TS_DIR}/lib:{os.path.dirname(torch.__file__)}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = sp
    env["WARP_TRITON_SITE_PACKAGES"] = sp
    subprocess.run(["pkill", "-9", "-f", "tritonserver"], capture_output=True)
    time.sleep(1)
    log = open(str(tmp_path / "serve.log"), "w")
    proc = subprocess.Popen(
        [os.path.join(_TS_DIR, "bin", "tritonserver"), f"--model-repository={repo}",
         f"--backend-directory={_TS_DIR}/backends", "--http-port=8805", "--grpc-port=8806",
         "--metrics-port=8807", "--exit-timeout-secs=3"],
        env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        client = http.InferenceServerClient("localhost:8805")
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
        assert ready, f"tritonserver did not become ready; see {tmp_path/'serve.log'}"

        x = np.linspace(-1.0, 2.0, twn.N, dtype=np.float32)
        inp = http.InferInput(twn.IN_NAME, [twn.N], "FP32")
        inp.set_data_from_numpy(x)
        res = client.infer("ensemble", [inp], outputs=[http.InferRequestedOutput(twn.OUT_NAME)])
        out = res.as_numpy(twn.OUT_NAME)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    ref = np.maximum(x * twn.SCALE + twn.BIAS, 0.0)
    assert np.allclose(out, ref, atol=1e-5)
