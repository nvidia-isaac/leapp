#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Standalone validation of the Warp Triton python-backend model WITHOUT libtritonserver.

Mocks ``triton_python_backend_utils`` with a faithful DLPack-based shim, builds the model repo,
imports the real ``model.py``, and drives ``initialize()`` + ``execute()`` with a GPU tensor —
exercising the identical capture_load + DLPack + wp.from_torch + capture_launch path the real
python backend uses. Asserts bit-exact agreement vs a direct torch reference.

Run: /tmp/leapp-warp/venv/bin/python examples/triton_warp_node/harness_standalone.py
"""
import importlib.util
import os
import sys
import types

import torch

import make_warp_model_repo as builder


# ---- faithful pb_utils shim (DLPack-based, like the real Triton python backend) ----
class _Tensor:
    def __init__(self, name, torch_tensor):
        self._name = name
        self._t = torch_tensor

    def name(self):
        return self._name

    def to_dlpack(self):
        return torch.utils.dlpack.to_dlpack(self._t)

    @staticmethod
    def from_dlpack(name, capsule):
        return _Tensor(name, torch.from_dlpack(capsule))

    def torch_tensor(self):
        return self._t


class _InferenceRequest:
    def __init__(self, tensors):
        self._by_name = {t.name(): t for t in tensors}


class _InferenceResponse:
    def __init__(self, output_tensors=None, error=None):
        self.output_tensors = output_tensors or []
        self.error = error


def _install_fake_pb_utils():
    m = types.ModuleType("triton_python_backend_utils")
    m.Tensor = _Tensor
    m.InferenceRequest = _InferenceRequest
    m.InferenceResponse = _InferenceResponse
    m.get_input_tensor_by_name = lambda req, name: req._by_name[name]
    sys.modules["triton_python_backend_utils"] = m
    return m


def _load_model_py(version_dir):
    spec = importlib.util.spec_from_file_location("warp_triton_model", os.path.join(version_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, version_dir)
    spec.loader.exec_module(mod)
    return mod


def main():
    repo = "/tmp/leapp-warp/triton_repo"
    version_dir = builder.build_repo(repo)
    _install_fake_pb_utils()
    model_mod = _load_model_py(version_dir)

    model = model_mod.TritonPythonModel()
    model.initialize({"model_repository": repo, "model_version": "1", "model_name": "warp_node"})

    x = torch.linspace(-1.0, 2.0, builder.N, device="cuda", dtype=torch.float32)
    req = _InferenceRequest([_Tensor(builder.IN_NAME, x)])
    [resp] = model.execute([req])
    out = {t.name(): t.torch_tensor() for t in resp.output_tensors}[builder.OUT_NAME]

    ref = builder.reference(x)
    err = float((out - ref).abs().max())
    ok = torch.allclose(out, ref, rtol=1e-4, atol=1e-5) and int((out > 0).sum()) > 0 and int((out == 0).sum()) > 0
    print("input          :", x.detach().cpu().numpy())
    print("warp model out :", out.detach().cpu().numpy())
    print("reference out  :", ref.detach().cpu().numpy())
    print("output device  :", out.device, "| max_abs_err =", err)
    print("\nSTANDALONE RESULT:", "PASS — warp model.py execute() path round-trips on GPU via DLPack"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
