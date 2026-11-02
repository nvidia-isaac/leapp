#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deep graph finalization through normal export and bundle reload."""

import sys

import torch

import leapp


def test_deep_chain_onnx_bundle_reload(tmp_path):
    """Cover ONNX lowering and runtime replay beyond unit-level FX finalization."""
    recursion_limit = sys.getrecursionlimit()
    depth = max(1500, recursion_limit + 100)
    leapp.start(name="deep_chain", save_path=str(tmp_path))
    try:
        x = leapp.annotate.input_tensors("chain", {"x": torch.ones(2)})
        for _ in range(depth):
            x = x + 0.001
        leapp.annotate.output_tensors("chain", {"y": x}, export_with="onnx")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = leapp.InferenceManager(str(tmp_path / "deep_chain" / "deep_chain.yaml"))
    for fresh in (torch.tensor([-3.0, 0.0]), torch.tensor([0.25, 7.0])):
        expected = fresh.clone()
        for _ in range(depth):
            expected = expected + 0.001
        actual = manager.run_policy({"chain/x": fresh})["chain/y"]
        # Repeated float32 adds can be reassociated by backend optimizers.
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert sys.getrecursionlimit() == recursion_limit
