#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Trivial GPU python-backend model: out = in*scale + bias (elementwise, DLPack zero-copy).

Used ONLY in the dockerless multi-step ensemble test as a stand-in for the ONNX neighbor steps
(the PyTriton bundle ships only the python backend; the real deploy uses onnxruntime). It exercises
the SAME Triton GPU-residency contract as an onnx step: KIND_GPU + FORCE_CPU_ONLY_INPUT_TENSORS:"no"
+ DLPack in/out, so intermediate tensors stay on the GPU across the ensemble step->step handoff.
"""
import json
import os
import sys

# Bootstrap torch path BEFORE importing torch (dockerless prototype only; no-op in real deploy).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_extra = os.environ.get("WARP_TRITON_SITE_PACKAGES")
if _extra:
    sys.path.insert(0, _extra)

import torch  # noqa: E402
import triton_python_backend_utils as pb_utils  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        self.scale = float(params.get("scale", "1.0"))
        self.bias = float(params.get("bias", "0.0"))
        self.in_name = cfg["input"][0]["name"]
        self.out_name = cfg["output"][0]["name"]
        self.label = cfg["name"]
        self.device = "cuda"
        if str(args.get("model_instance_kind", "")).upper() == "GPU":
            self.device = f"cuda:{int(args.get('model_instance_device_id', 0))}"

    def execute(self, requests):
        dbg = os.environ.get("WARP_TRITON_DEBUG_DEVICE")
        responses = []
        for request in requests:
            in_t = pb_utils.get_input_tensor_by_name(request, self.in_name)
            if dbg:
                sys.stderr.write(f"[{self.label}] input '{self.in_name}' is_cpu={in_t.is_cpu()}\n")
                sys.stderr.flush()
            x = torch.from_dlpack(in_t.to_dlpack()).to(self.device)
            y = (x * self.scale + self.bias).contiguous()
            out_t = pb_utils.Tensor.from_dlpack(self.out_name, torch.utils.dlpack.to_dlpack(y))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
        return responses
