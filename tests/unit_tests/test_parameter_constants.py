#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Parameter constants must retain sharing without merging different views."""

import pytest
import torch

from leapp.leapp_graph.traced_node import TracedTensorNode


def constant_count(node):
    return len({n.target for n in node.m.graph.nodes if n.op == "get_attr"})


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_reused_parameter_is_one_fx_constant(requires_grad, tied):
    weight = torch.nn.Parameter(torch.randn(4, 4), requires_grad=requires_grad)
    other = (
        torch.nn.Parameter(weight.detach(), requires_grad=requires_grad)
        if tied
        else weight
    )
    node = TracedTensorNode("policy")
    sample = torch.randn(2, 4)
    x = node.create_input(sample, name="x")
    node.compile_trace({"y": (x @ weight) @ other})

    assert constant_count(node) == 1
    fresh = torch.randn_like(sample)
    torch.testing.assert_close(node.m(fresh), (fresh @ weight) @ other)
    exported = torch.export.export(node.m, (torch.randn(2, 4),))
    assert len(exported.constants) + len(exported.state_dict) == 1


@pytest.mark.parametrize("view", ["transpose", "offset", "shape", "dtype", "negative"])
def test_different_parameter_views_remain_distinct(view):
    storage = torch.arange(8, dtype=torch.float32)
    first = storage[:4].reshape(2, 2)
    second = {
        "transpose": first.t(),
        "offset": storage[4:].reshape(2, 2),
        "shape": first.reshape(4),
        "dtype": first.view(torch.int32),
        "negative": torch._neg_view(first),
    }[view]
    first = torch.nn.Parameter(first, requires_grad=False)
    second = torch.nn.Parameter(second, requires_grad=False)

    def forward(x):
        return (x.reshape(first.shape) + first).flatten() + (
            x.reshape(second.shape) + second
        ).flatten()

    node = TracedTensorNode("policy")
    x = node.create_input(torch.randn(4), name="x")
    node.compile_trace({"y": forward(x)})
    assert constant_count(node) == 2
    fresh = torch.randn(4)
    torch.testing.assert_close(node.m(fresh), forward(fresh))


def test_rebinding_parameter_data_preserves_both_values():
    weight = torch.nn.Parameter(torch.ones(4))
    node = TracedTensorNode("policy")
    x = node.create_input(torch.randn(4), name="x")
    y = x * weight
    weight.data = torch.full((4,), 3.0)
    node.compile_trace({"y": y + x * weight})

    assert constant_count(node) == 2
    fresh = torch.randn(4)
    torch.testing.assert_close(node.m(fresh), fresh * 4)


def test_parameter_constants_are_reset_with_trace():
    node = TracedTensorNode("policy")
    weight = torch.nn.Parameter(torch.ones(4))
    for value in (1.0, 3.0):
        node.reset_trace_state()
        weight.data = torch.full((4,), value)
        x = node.create_input(torch.randn(4), name="x")
        node.compile_trace({"y": x * weight + x * weight})

        assert constant_count(node) == 1
        fresh = torch.randn(4)
        torch.testing.assert_close(node.m(fresh), fresh * (2 * value))


def test_parameter_constants_are_local_to_each_node():
    weight = torch.nn.Parameter(torch.randn(4))
    for name in ("first", "second"):
        node = TracedTensorNode(name)
        x = node.create_input(torch.randn(4), name="x")
        node.compile_trace({"y": x * weight + x * weight})
        assert constant_count(node) == 1
        fresh = torch.randn(4)
        torch.testing.assert_close(node.m(fresh), fresh * weight * 2)
