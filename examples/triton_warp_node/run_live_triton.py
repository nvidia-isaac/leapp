#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Drive a REAL (dockerless) Triton server on the warp model repo and infer the ensemble.

Starts the pytriton-bundled `tritonserver` as a subprocess on the generated model repository
(warp_node python-backend model + a single-step ensemble), waits for readiness, runs an
inference through the HTTP client, and validates against a torch reference. Tears the server down.

This proves a Warp `.wrp` runs as a python-backend model INSIDE a live Triton ensemble — the exact
shape `create_triton_model_repo.py` emits for Runtime A.

Run: /tmp/leapp-warp/venv/bin/python examples/triton_warp_node/run_live_triton.py
"""
import os
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_warp_model_repo as builder  # noqa: E402

VENV = "/tmp/leapp-warp/venv"
SP = os.path.join(VENV, "lib/python3.12/site-packages")
TS_DIR = os.path.join(SP, "pytriton/tritonserver")
TS_BIN = os.path.join(TS_DIR, "bin/tritonserver")
REPO = "/tmp/leapp-warp/triton_repo"
HTTP_PORT = 8765


def _ensure_stub():
    # pytriton stores per-version stubs separately; the python backend looks in backends/python/.
    dst = os.path.join(TS_DIR, "backends/python/triton_python_backend_stub")
    src = os.path.join(TS_DIR, "python_backend_stubs/3.12/triton_python_backend_stub")
    if not os.path.exists(dst) and os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)


def main():
    subprocess.run(["pkill", "-9", "-f", "tritonserver"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "triton_python_backend_stub"], capture_output=True)
    time.sleep(1)
    _ensure_stub()
    builder.build_repo(REPO)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{TS_DIR}/lib:{SP}/torch/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = SP
    env["WARP_TRITON_SITE_PACKAGES"] = SP  # let model.py find torch/warp in this venv

    log = open("/tmp/leapp-warp/triton_live.log", "w")
    proc = subprocess.Popen(
        [TS_BIN, f"--model-repository={REPO}", f"--backend-directory={TS_DIR}/backends",
         f"--http-port={HTTP_PORT}", "--grpc-port=8766", "--metrics-port=8767",
         "--exit-timeout-secs=3"],
        env=env, stdout=log, stderr=subprocess.STDOUT)

    try:
        import tritonclient.http as http
        client = http.InferenceServerClient(f"localhost:{HTTP_PORT}")
        ready = False
        for _ in range(90):
            if proc.poll() is not None:
                print("server exited early, rc=", proc.returncode)
                break
            try:
                if client.is_server_ready():
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            print("server not ready; see /tmp/leapp-warp/triton_live.log")
            return 1

        print("server live:", client.is_server_live(), "| ensemble ready:", client.is_model_ready("ensemble"))
        x = np.linspace(-1.0, 2.0, builder.N, dtype=np.float32)
        inp = http.InferInput("in", [builder.N], "FP32")
        inp.set_data_from_numpy(x)
        res = client.infer("ensemble", [inp], outputs=[http.InferRequestedOutput("out")])
        out = res.as_numpy("out")
        ref = np.maximum(x * builder.SCALE + builder.BIAS, 0.0)
        err = float(np.abs(out - ref).max())
        ok = np.allclose(out, ref, atol=1e-5)
        print("ensemble in :", x)
        print("ensemble out:", out)
        print("reference   :", ref)
        print("max_abs_err =", err)
        print("\nLIVE TRITON ENSEMBLE:",
              "PASS - warp .wrp ran as a python-backend model inside a real Triton ensemble" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
