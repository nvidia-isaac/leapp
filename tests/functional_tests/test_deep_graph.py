#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Deep graph finalization through normal export and bundle reload."""

import sys

import torch

import leapp


def test_deep_branching_graph_onnx_bundle_reload(tmp_path):
    """Cover public annotation, pruning, ONNX export, and ordered bundle replay.

    This combines deep shared dependencies, a view mutation, and unused-input
    pruning with export and fresh-input inference in one representative case.
    """
    recursion_limit = sys.getrecursionlimit()
    depth = max(1500, recursion_limit + 100)
    leapp.start(name="deep_chain", save_path=str(tmp_path))
    try:
        x, unused = leapp.annotate.input_tensors(
            "chain", {"x": torch.ones(2), "unused": torch.zeros(2)}
        )
        unused * 100
        mutated = x.clone()
        view = mutated[:1]
        result = x
        for _ in range(depth):
            result = result + 0.001
        view.add_(result[:1])
        left = mutated * 2
        right = result - x
        leapp.annotate.output_tensors(
            "chain",
            {"right": right, "merged": left + right, "left": left},
            export_with="onnx",
        )
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = leapp.InferenceManager(str(tmp_path / "deep_chain" / "deep_chain.yaml"))
    assert list(manager.inputs) == ["chain/x"]
    assert list(manager.outputs) == ["chain/right", "chain/merged", "chain/left"]
    for fresh in (torch.tensor([-3.0, 0.0]), torch.tensor([0.25, 7.0])):
        expected = fresh.clone()
        for _ in range(depth):
            expected = expected + 0.001
        expected_mutated = fresh.clone()
        expected_mutated[:1] += expected[:1]
        actual = manager.run_policy({"chain/x": fresh})
        expected_outputs = {
            "chain/right": expected - fresh,
            "chain/merged": expected_mutated * 2 + (expected - fresh),
            "chain/left": expected_mutated * 2,
        }
        # Repeated float32 adds can be reassociated by backend optimizers.
        for name, reference in expected_outputs.items():
            torch.testing.assert_close(actual[name], reference, rtol=1e-4, atol=1e-5)
    assert sys.getrecursionlimit() == recursion_limit
