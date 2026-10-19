#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Public stateful and Warp artifact replay for captured view mutations."""

import pytest
import torch

import leapp
from leapp import annotate, InferenceManager


def stateful_step(observation, initial):
    observation = annotate.input_tensors("policy", {"obs": observation})
    state = annotate.state_tensors("policy", {"history": initial})
    view = state.reshape(3, 4)[:, 1:3]
    view += observation
    annotate.update_state("policy", {"history": state})
    annotate.output_tensors("policy", {"action": state * 2}, export_with="pt2")


def test_view_mutation_feedback(tmp_path):
    leapp.start("state_views", save_path=str(tmp_path), max_cached_io=2)
    try:
        for amount in (1.0, 3.0):
            stateful_step(torch.tensor([amount]), torch.zeros(12))
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(tmp_path / "state_views/state_views.yaml"))
    expected = torch.zeros(12)
    for amount in (2.0, -1.0, 4.0):
        expected.reshape(3, 4)[:, 1:3] += amount
        device = manager.nodes["policy"].device
        result = manager.run_policy(
            {"policy/obs": torch.tensor([amount], device=device)}
        )
        torch.testing.assert_close(result["policy/action"].cpu(), expected * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Warp/APIC requires CUDA")
@pytest.mark.parametrize("backend", ["pt2", "onnx"])
@pytest.mark.parametrize("allocate_output", [False, True])
def test_warp_view_mutation_reload(tmp_path, backend, allocate_output):
    wp = pytest.importorskip("warp")
    wp.init()

    @wp.kernel
    def add(
        src: wp.array(dtype=wp.float32),
        delta: wp.array(dtype=wp.float32),
        dst: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        dst[i] = src[i] + delta[0]

    def pipeline(base, delta):
        leaf = base[1:10][1:5]
        src, operand = wp.from_torch(leaf), wp.from_torch(delta)
        with annotate.warp_op("policy"):
            dst = wp.empty_like(src) if allocate_output else src
            wp.launch(
                add, dim=src.size, inputs=[src, operand], outputs=[dst], device="cuda:0"
            )
        result = wp.to_torch(dst)
        sibling = base[3:8]
        sibling += delta * 2
        return {"base": base * 2, "leaf": result * 3, "sibling": sibling * 4}

    name = f"warp_views_{backend}_{allocate_output}"
    leapp.start(name, save_path=str(tmp_path), max_cached_io=2)
    try:
        for offset, amount in ((0, 1), (100, 7)):
            base, delta = annotate.input_tensors(
                "policy",
                {
                    "x": torch.arange(12.0, device="cuda") + offset,
                    "delta": torch.tensor([float(amount)], device="cuda"),
                },
            )
            annotate.output_tensors(
                "policy", pipeline(base, delta), export_with=backend
            )
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(tmp_path / name / f"{name}.yaml"))
    for offset, amount in ((0, 1), (100, 1), (0, 7), (-20, -3)):
        given = torch.arange(12.0, device="cuda") + offset
        expected = given.clone()
        leaf = expected[2:6]
        if allocate_output:
            leaf = leaf.clone()
        leaf += amount
        expected[3:8] += amount * 2
        outputs = manager.run_policy(
            {
                "policy/x": given.clone(),
                "policy/delta": torch.tensor([float(amount)], device="cuda"),
            }
        )
        for key, reference in {
            "base": expected * 2,
            "leaf": leaf * 3,
            "sibling": expected[3:8] * 4,
        }.items():
            torch.testing.assert_close(outputs[f"policy/{key}"].to("cuda"), reference)
