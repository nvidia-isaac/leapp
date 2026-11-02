#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Dependency pruning for mutation roots outside the output data path."""

import sys

import pytest
import torch

from leapp.leapp_graph.traced_node import TracedTensorNode


@pytest.mark.parametrize("preserve_mutations", [False, True])
@pytest.mark.parametrize("mutation_kind", ["metadata", "impure"])
def test_deep_mutation_dependencies(preserve_mutations, mutation_kind):
    ctx = TracedTensorNode("mutation")
    x = ctx.create_input(torch.zeros(2), "x")
    source = ctx.create_input(torch.ones(2), "source")
    deep = source.proxy.node
    for _ in range(max(1500, sys.getrecursionlimit() + 100)):
        deep = ctx.graph.call_function(torch.add, (deep, 0.001))
    if mutation_kind == "metadata":
        mutation = ctx.graph.call_method("copy_", (x.proxy.node, deep))
        mutation.meta["leapp_mutation"] = True
    else:
        mutation = ctx.graph.call_function(
            torch.ops.aten.copy_.default, (x.proxy.node, deep)
        )
        assert mutation.is_impure()
    ctx.graph.output(x.proxy.node)

    ctx._prune_graph([x.proxy.node], preserve_mutations=preserve_mutations)

    ctx.graph.lint()
    assert (deep in ctx.graph.nodes) == preserve_mutations
    assert (mutation in ctx.graph.nodes) == preserve_mutations
    expected_inputs = ["x", "source"] if preserve_mutations else ["x"]
    assert [desc.name for desc in ctx.inputs] == expected_inputs
