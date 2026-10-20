#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Repeated denoiser calls should not duplicate exported weights."""

import onnx
import pytest
import torch

import leapp
from leapp import annotate, InferenceManager


@pytest.mark.parametrize("global_patching", [False, True])
@pytest.mark.parametrize(
    "backend,params",
    [
        ("onnx", {}),
        ("onnx", {"optimize": False}),
        ("onnx-torchscript", {}),
        ("exported-program", {}),
        ("jit-script", {}),
        ("jit-trace", {}),
    ],
)
def test_reused_parameter_export(tmp_path, backend, params, global_patching):
    # Exceed onnxscript's small-initializer deduplication threshold.
    dim = 64
    linear = torch.nn.Linear(dim, dim).eval()

    def forward(x):
        for _ in range(4):
            x = x + torch.tanh(linear(x)) / 4
        return x

    leapp.start("graph", save_path=str(tmp_path), global_patching=global_patching)
    try:
        with torch.no_grad():
            for _ in range(2):
                x = annotate.input_tensors("policy", {"x": torch.randn(1, dim)})
                annotate.output_tensors(
                    "policy",
                    {"y": forward(x)},
                    export_with=backend,
                    backend_params=params,
                )
    finally:
        leapp.stop()
    assert leapp.compile_graph(visualize=False, validate=True, strict=True) == {
        "policy": True
    }

    if backend.startswith("onnx"):
        initializers = onnx.load(tmp_path / "graph" / "policy.onnx").graph.initializer
        assert sum(tuple(t.dims) == (dim, dim) for t in initializers) == 1
        assert sum(onnx.numpy_helper.to_array(t).nbytes for t in initializers) < (
            2 * dim * dim * 4
        )

    manager = InferenceManager(str(tmp_path / "graph" / "graph.yaml"))
    with torch.no_grad():
        for scale in (0.2, 1.0, 3.0):
            fresh = torch.randn(1, dim) * scale
            actual = manager.run_policy({"policy/x": fresh})["policy/y"]
            torch.testing.assert_close(actual, forward(fresh), rtol=1e-4, atol=1e-5)
