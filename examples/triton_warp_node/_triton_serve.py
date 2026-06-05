#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Dockerless helper: discover the PyTriton-bundled ``tritonserver`` and serve a model repo.

Portable — no hardcoded paths. The tritonserver binary and the python-backend stub are discovered
from the installed ``nvidia-pytriton`` package; torch's site-packages (for the dockerless
``WARP_TRITON_SITE_PACKAGES`` shim) are derived from ``torch.__file__``. Returns None from
``discover_tritonserver()`` when pytriton is not installed, so callers can skip cleanly.

In a real Triton deployment you do NOT use any of this — Triton + the python/onnx backends + torch
+ warp live in the container. This only exists to validate the warp ensemble step dockerless.
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch


def discover_tritonserver():
    """Return the PyTriton-bundled tritonserver dir, or None if pytriton is not installed."""
    try:
        import pytriton
    except Exception:
        return None
    ts_dir = os.path.join(os.path.dirname(pytriton.__file__), "tritonserver")
    return ts_dir if os.path.exists(os.path.join(ts_dir, "bin", "tritonserver")) else None


def _ensure_python_stub(ts_dir):
    """The python backend looks for the stub in backends/python/; pytriton stores per-version
    stubs separately. Copy the one matching this interpreter into place."""
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    dst = os.path.join(ts_dir, "backends", "python", "triton_python_backend_stub")
    src = os.path.join(ts_dir, "python_backend_stubs", pyver, "triton_python_backend_stub")
    if not os.path.exists(dst):
        if not os.path.exists(src):
            raise RuntimeError(f"no python_backend_stub for py{pyver} in the pytriton bundle")
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)


class TritonServer:
    """Context manager that serves a model repository on a real (dockerless) tritonserver."""

    def __init__(self, repo, http_port=8800, log_path=None, debug_device=True):
        self.repo = repo
        self.http_port = http_port
        self.ts_dir = discover_tritonserver()
        self.log_path = log_path or os.path.join(tempfile.gettempdir(), f"leapp_triton_{http_port}.log")
        self.debug_device = debug_device
        self.proc = None
        self.client = None

    def __enter__(self):
        if self.ts_dir is None:
            raise RuntimeError("nvidia-pytriton (bundled tritonserver) is not installed")
        _ensure_python_stub(self.ts_dir)
        sp = os.path.dirname(os.path.dirname(torch.__file__))  # this interpreter's site-packages
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (f"{self.ts_dir}/lib:{os.path.dirname(torch.__file__)}/lib:"
                                  + env.get("LD_LIBRARY_PATH", ""))
        env["PYTHONPATH"] = sp
        env["WARP_TRITON_SITE_PACKAGES"] = sp  # let the python-backend model find torch/warp
        if self.debug_device:
            env.setdefault("WARP_TRITON_DEBUG_DEVICE", "1")
        subprocess.run(["pkill", "-9", "-f", "tritonserver"], capture_output=True)
        time.sleep(1)
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [os.path.join(self.ts_dir, "bin", "tritonserver"),
             f"--model-repository={self.repo}", f"--backend-directory={self.ts_dir}/backends",
             f"--http-port={self.http_port}", f"--grpc-port={self.http_port + 1}",
             f"--metrics-port={self.http_port + 2}", "--exit-timeout-secs=3"],
            env=env, stdout=self._log, stderr=subprocess.STDOUT)
        import tritonclient.http as http
        self.client = http.InferenceServerClient(f"localhost:{self.http_port}")
        for _ in range(90):
            if self.proc.poll() is not None:
                raise RuntimeError(f"tritonserver exited early; see {self.log_path}")
            try:
                if self.client.is_server_ready():
                    return self
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"tritonserver did not become ready; see {self.log_path}")

    def infer(self, model, inputs, outputs):
        """inputs: {name: np.ndarray(float32)}; outputs: [name]. Returns {name: np.ndarray}."""
        import tritonclient.http as http
        ins = []
        for name, arr in inputs.items():
            t = http.InferInput(name, list(arr.shape), "FP32")
            t.set_data_from_numpy(arr)
            ins.append(t)
        res = self.client.infer(model, ins, outputs=[http.InferRequestedOutput(o) for o in outputs])
        return {o: res.as_numpy(o) for o in outputs}

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log", None):
            self._log.close()
