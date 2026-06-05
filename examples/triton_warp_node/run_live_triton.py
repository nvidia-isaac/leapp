#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Demo: serve a single-node Warp ensemble on a real (dockerless) Triton server and infer.

Builds the warp model repo (one `.wrp` python-backend node + a one-step ensemble) and serves it
via the PyTriton-bundled tritonserver, proving the warp `.wrp` runs as a python-backend model
inside a real Triton ensemble over HTTP. Skips cleanly if `nvidia-pytriton` is not installed.

Run: python examples/triton_warp_node/run_live_triton.py   (in an env with nvidia-pytriton)
The canonical/portable validation is tests/functional_tests/test_triton_warp_node.py.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_warp_model_repo as builder
from _triton_serve import TritonServer, discover_tritonserver


def main():
    if discover_tritonserver() is None:
        print("SKIP: nvidia-pytriton (bundled tritonserver) not installed in this environment.")
        return 0

    repo = tempfile.mkdtemp(prefix="warp_triton_repo_")
    builder.build_repo(repo)

    x = np.linspace(-1.0, 2.0, builder.N, dtype=np.float32)
    with TritonServer(repo, http_port=8800) as ts:
        print("server live:", ts.client.is_server_live(), "| ensemble ready:", ts.client.is_model_ready("ensemble"))
        out = ts.infer("ensemble", {builder.IN_NAME: x}, [builder.OUT_NAME])[builder.OUT_NAME]

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


if __name__ == "__main__":
    raise SystemExit(main())
