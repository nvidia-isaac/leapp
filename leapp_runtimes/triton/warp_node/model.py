#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Triton PYTHON-BACKEND model that replays a LEAPP Warp (APIC) node as one ensemble step.

This is the template that `create_triton_model_repo.py`'s `_create_warp_model_dir` would emit
into `<warp_node>/1/model.py` (alongside `graph.wrp`, `graph.warpmeta.json`, `graph_modules/`,
and `warp_apic_runtime.py`). It loads the `.wrp` once in `initialize()` and, in `execute()`,
bridges Triton tensors to Warp **zero-copy via DLPack** (`torch.from_dlpack` → `wp.from_torch`),
replays the captured APIC graph, and returns DLPack outputs — so warp↔onnx tensors stay on the
GPU between ensemble steps (requires `KIND_GPU` + `FORCE_CPU_ONLY_INPUT_TENSORS:"no"`).
"""
import os
import sys

# Make this model dir importable (for warp_apic_runtime). In a real deploy the python-backend's
# env already has torch+warp; WARP_TRITON_SITE_PACKAGES is only set in dockerless prototype runs
# where torch/warp live in a separate venv the Triton stub's interpreter doesn't see by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_extra_sp = os.environ.get("WARP_TRITON_SITE_PACKAGES")
if _extra_sp:
    sys.path.insert(0, _extra_sp)

import torch  # noqa: E402
import triton_python_backend_utils as pb_utils  # noqa: E402

from warp_apic_runtime import WarpApicRunner  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        import glob
        version_dir = os.path.dirname(os.path.abspath(__file__))
        wrps = sorted(glob.glob(os.path.join(version_dir, "*.wrp")))
        if len(wrps) != 1:
            raise RuntimeError(f"expected exactly one .wrp in {version_dir}, found {wrps}")
        # I2: bind to the GPU Triton actually placed this instance on (multi-GPU servers).
        device = None
        if str(args.get("model_instance_kind", "")).upper() == "GPU":
            device = f"cuda:{int(args.get('model_instance_device_id', 0))}"
        self.runner = WarpApicRunner(wrps[0], device=device)

    def execute(self, requests):
        debug_dev = os.environ.get("WARP_TRITON_DEBUG_DEVICE")
        responses = []
        for request in requests:
            inputs = {}
            for name in self.runner.input_names:
                in_t = pb_utils.get_input_tensor_by_name(request, name)
                if debug_dev:
                    # Prove GPU-resident step->step handoff: is_cpu()==False means the upstream
                    # step's GPU output reached us without a host copy.
                    sys.stderr.write(f"[warp_node] input '{name}' is_cpu={in_t.is_cpu()}\n")
                    sys.stderr.flush()
                # Zero-copy Triton -> torch (-> warp) over DLPack.
                inputs[name] = torch.from_dlpack(in_t.to_dlpack())

            outs = self.runner.run_torch(inputs)

            out_tensors = [
                pb_utils.Tensor.from_dlpack(name, torch.utils.dlpack.to_dlpack(outs[name].contiguous()))
                for name in self.runner.output_names
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        return responses
