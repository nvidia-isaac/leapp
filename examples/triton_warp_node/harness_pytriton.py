#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Validate the Warp node under a REAL (dockerless) Triton server via PyTriton.

Stands up an in-process tritonserver (bundled in nvidia-pytriton), binds the Warp APIC replay as
the model's inference function, and drives it through the Triton client — proving warp executes
inside a live Triton server. (Zero-copy GPU/DLPack residency is validated separately by
harness_standalone.py; PyTriton's simple binding marshals via numpy, so here we move to CUDA inside
the infer fn. The production path is the python-backend model.py emitted into the model repo.)

Run: /tmp/leapp-warp/venv/bin/python examples/triton_warp_node/harness_pytriton.py
"""
import os

import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

import make_warp_model_repo as builder
from warp_apic_runtime import WarpApicRunner

REPO = "/tmp/leapp-warp/triton_repo"
_vd = builder.build_repo(REPO)
_runner = WarpApicRunner(os.path.join(_vd, "graph.wrp"))


@batch
def _warp_infer(**inputs):
    x = torch.from_numpy(inputs["in"]).cuda()  # [B, N] float32
    rows = [_runner.run_torch({"in": x[i]})["out"] for i in range(x.shape[0])]
    return {"out": torch.stack(rows, 0).detach().cpu().numpy()}


def main():
    with Triton() as triton:
        triton.bind(
            model_name="warp_node",
            infer_func=_warp_infer,
            inputs=[Tensor(name="in", dtype=np.float32, shape=(builder.N,))],
            outputs=[Tensor(name="out", dtype=np.float32, shape=(builder.N,))],
            config=ModelConfig(max_batch_size=4),
        )
        triton.run()
        from pytriton.client import ModelClient
        x = np.linspace(-1.0, 2.0, builder.N, dtype=np.float32)
        with ModelClient("localhost", "warp_node", init_timeout_s=120) as client:
            result = client.infer_sample(**{"in": x})
        out = result["out"]

    ref = torch.relu(torch.from_numpy(x) * builder.SCALE + builder.BIAS).numpy()
    err = float(np.abs(out - ref).max())
    ok = np.allclose(out, ref, rtol=1e-4, atol=1e-5)
    print("triton out :", out)
    print("reference  :", ref)
    print("max_abs_err =", err)
    print("\nPYTRITON RESULT:", "PASS — warp node served by a live Triton server" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
